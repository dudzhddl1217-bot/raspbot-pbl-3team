#!/usr/bin/env python3
# 지시서: specs/04-obstacle.md
"""초음파 중앙값으로 전진·정지·재출발. 가져오기만 해서는 장치에 접근하지 않는다."""
import argparse
from collections import deque
from contextlib import contextmanager
import csv
from datetime import datetime
import math
from pathlib import Path
import select
from statistics import median
import sys
import termios
import time
import tty
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
KST = ZoneInfo("Asia/Seoul")
INTERVAL = 0.05
BEEP_SECONDS = 0.2
MOTOR_REFRESH = 0.2
MOTOR_GAP = 0.01
STOP_REPEATS = 3


class ObstacleController:
    """장치 없이 시험할 수 있는 거리 판단. 반환값은 (속도, 이번 정지 부저)."""

    def __init__(self):
        self.samples = deque(maxlen=3)
        self.invalid_count = 0
        self.moving = False
        self.filtered_cm = None
        self.status = "정상값 대기"

    def update(self, distance_cm):
        valid = (isinstance(distance_cm, (int, float))
                 and not isinstance(distance_cm, bool)
                 and math.isfinite(distance_cm) and 0 < distance_cm <= 400)
        if not valid:
            self.invalid_count += 1
            if self.invalid_count >= 3:
                self.moving = False
                self.samples.clear()
                self.filtered_cm = None
                self.status = "연속 오류 정지"
            else:
                self.status = "오류 표본 제외·이전 상태 유지"
            return (50 if self.moving else 0), False

        self.invalid_count = 0
        self.samples.append(distance_cm)
        if len(self.samples) < 3:
            self.moving = False
            self.filtered_cm = None
            self.status = "정상값 대기"
            return 0, False
        self.filtered_cm = float(median(self.samples))
        was_moving = self.moving
        if self.moving and self.filtered_cm <= 25:
            self.moving = False
        elif not self.moving and self.filtered_cm >= 35:
            self.moving = True
        self.status = "전진" if self.moving else "거리 조건 정지"
        return (50 if self.moving else 0), was_moving and not self.moving


def read_distance_cm(bot):
    values = []
    for reg in (0x1b, 0x1a):
        data = bot.read_data_array(reg, 1)
        if (data is None or len(data) != 1 or not isinstance(data[0], int)
                or not 0 <= data[0] <= 255):
            raise RuntimeError("초음파 I2C 읽기 실패. 연결과 전원을 확인하세요.")
        values.append(data[0])
    return ((values[0] << 8) | values[1]) / 10.0


def set_speed(bot, speed):
    if speed not in (0, 50):
        raise ValueError("이 지시서의 바퀴 속도는 0 또는 50입니다.")
    for number in range(4):
        bot.Ctrl_Muto(number, speed)
        if number < 3:
            time.sleep(MOTOR_GAP)


def cleanup(bot):
    # 한 명령에서 예외가 나도 나머지 바퀴·부저·초음파 끄기를 모두 시도한다.
    errors = []
    for repeat in range(STOP_REPEATS):
        sent_at = time.monotonic()
        for number in range(4):
            try:
                bot.Ctrl_Muto(number, 0)
            except Exception as exc:
                errors.append(exc)
            if number < 3:
                time.sleep(MOTOR_GAP)
        for function in (bot.Ctrl_BEEP_Switch, bot.Ctrl_Ulatist_Switch):
            try:
                function(0)
            except Exception as exc:
                errors.append(exc)
        if repeat < STOP_REPEATS - 1:
            time.sleep(max(0, sent_at + MOTOR_REFRESH - time.monotonic()))
    if errors:
        raise RuntimeError("장치 정리 명령 실패. 라즈봇 상태를 확인하세요.") from errors[0]


@contextmanager
def keyboard_stop():
    if not sys.stdin.isatty():
        raise RuntimeError("아무 키 정지를 위해 터미널에서 실행해야 합니다.")
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield lambda: bool(select.select([sys.stdin], [], [], 0)[0])
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, original)


def run_obstacle(bot, key_pressed, stream):
    writer = csv.writer(stream, lineterminator="\n")
    controller = ObstacleController()
    beep_until = None
    last_speed = 0
    last_display = None
    start = time.monotonic()
    next_read = start + 1.0  # 프로젝트 수칙대로 초음파를 켜고 약 1초 기다린다.
    try:
        writer.writerow(("time", "elapsed_s", "raw_cm", "median_cm", "invalid_count",
                         "speed", "state", "beep_started", "ruler_cm", "motor_batches"))
        sent_at = time.monotonic()
        set_speed(bot, 0)
        next_motor = sent_at + MOTOR_REFRESH
        motor_batches = 1
        bot.Ctrl_BEEP_Switch(0)
        bot.Ctrl_Ulatist_Switch(1)
        while True:
            if key_pressed():
                print("키 입력: 정지합니다.", flush=True)
                break
            now = time.monotonic()
            if beep_until is not None and now >= beep_until:
                bot.Ctrl_BEEP_Switch(0)
                beep_until = None
            if now < next_read:
                batch_end = now + 3 * MOTOR_GAP
                # 4개 명령 사이 간격 때문에 예정된 측정·부저 끄기가 밀리지 않게 한다.
                room_for_batch = (batch_end <= next_read
                                  and (beep_until is None or batch_end <= beep_until))
                if now >= next_motor and room_for_batch:
                    sent_at = time.monotonic()
                    set_speed(bot, last_speed)
                    motor_batches += 1
                    next_motor = sent_at + MOTOR_REFRESH
                    continue
                delay = next_read - now
                if next_motor > now:
                    delay = min(delay, next_motor - now)
                if beep_until is not None:
                    delay = min(delay, max(0, beep_until - now))
                time.sleep(min(delay, 0.01))
                continue
            sampled_at = now
            raw_cm = read_distance_cm(bot)
            if key_pressed():
                break
            speed, beep = controller.update(raw_cm)
            motor_due = time.monotonic() >= next_motor
            beep_has_time = (beep_until is None
                             or time.monotonic() + 3 * MOTOR_GAP <= beep_until)
            if speed != last_speed or (motor_due and beep_has_time):
                sent_at = time.monotonic()
                set_speed(bot, speed)
                last_speed = speed
                motor_batches += 1
                next_motor = sent_at + MOTOR_REFRESH
            if beep:
                bot.Ctrl_BEEP_Switch(1)
                beep_until = time.monotonic() + BEEP_SECONDS
            row = (datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S.%f"),
                   round(sampled_at - start, 4), raw_cm,
                   controller.filtered_cm if controller.filtered_cm is not None else "",
                   controller.invalid_count, speed, controller.status, int(beep), "", motor_batches)
            writer.writerow(row)
            stream.flush()
            display = (controller.status, speed, beep)
            if display != last_display or beep:
                filtered = "대기" if controller.filtered_cm is None else f"{controller.filtered_cm:.1f}"
                print(f"거리 {raw_cm:.1f}cm / 중앙값 {filtered}cm / "
                      f"{controller.status} / 속도 {speed}", flush=True)
                last_display = display
            next_read = sampled_at + INTERVAL
    finally:
        cleanup(bot)


def main():
    parser = argparse.ArgumentParser(description="초음파 장애물 정지와 재출발")
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S_%f")
    parser.add_argument("--log", type=Path, default=ROOT / "results" / f"obstacle_{stamp}.csv",
                        help="거리와 상태 기록 경로")
    args = parser.parse_args()
    path = args.log.resolve()
    if not path.is_relative_to((ROOT / "results").resolve()):
        parser.error("기록 경로는 results/ 안이어야 합니다.")
    from Raspbot_Lib import Raspbot
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"기록: {path}. 아무 키나 누르면 정지합니다.", flush=True)
    try:
        with keyboard_stop() as key_pressed, path.open("w", newline="", encoding="utf-8") as stream:
            run_obstacle(Raspbot(), key_pressed, stream)
    except KeyboardInterrupt:
        print("Ctrl+C: 정지했습니다.", flush=True)


if __name__ == "__main__":
    main()
