#!/usr/bin/env python3
# 지시서: specs/02-line.md
"""라인센서 주행. --check-sensors는 읽기만 하며 모터 명령을 보내지 않는다."""
import argparse
from contextlib import contextmanager
import select
import sys
import termios
import time
import tty


# 읽은 x1,x2,x3,x4를 왼쪽부터 재배열한다. 2026-11-03 개별 센서 실측 확인.
SENSOR_ORDER = (1, 0, 2, 3)
STOP = (0, 0, 0, 0)


def read_sensors(bot):
    data = bot.read_data_array(0x0a, 1)
    if data is None or len(data) != 1 or not isinstance(data[0], int):
        raise RuntimeError("라인센서 읽기 실패. I2C 연결과 전원을 확인하세요.")
    value = data[0]
    return tuple((value >> shift) & 1 for shift in (3, 2, 1, 0))


def turn(direction, inside, outside):
    left, right = (inside, outside) if direction == "LEFT" else (outside, inside)
    return (left, left, right, right)


class LineFollower:
    """장치 접근 없이 센서 값과 시각으로 바퀴 명령을 정한다."""

    def __init__(self):
        self.last_turn = None
        self.command = STOP
        self.lost_since = None
        self.stopped = False
        self.status = "READY"

    def step(self, bits, now):
        if len(bits) != 4 or any(bit not in (0, 1) for bit in bits):
            raise ValueError("센서 값은 0 또는 1 네 칸이어야 합니다.")
        if self.stopped:
            return STOP
        physical = tuple(bits[i] for i in SENSOR_ORDER)
        black = {i for i, bit in enumerate(physical) if bit == 0}
        if not black:
            if self.lost_since is None:
                self.lost_since = now
            if self.last_turn is None or now - self.lost_since > 1.0:
                self.stopped = True
                self.status = "LOST_STOP"
                self.command = STOP
            else:
                self.status = "SEARCH_" + self.last_turn
                self.command = turn(self.last_turn, 0, 40)
            return self.command

        self.lost_since = None
        if {1, 2}.issubset(black):
            self.status = "FORWARD"
            self.command = (50, 50, 50, 50)
        elif black.issubset({0, 1}) or black.issubset({2, 3}):
            direction = "LEFT" if black.issubset({0, 1}) else "RIGHT"
            if len(black) == 2:
                inside, strength = 15, "MEDIUM"
            elif black & {1, 2}:
                inside, strength = 30, "GENTLE"
            else:
                inside, strength = 0, "SHARP"
            self.last_turn = direction
            self.status = direction + "_" + strength
            self.command = turn(direction, inside, 60)
        else:
            # 양쪽 감지 시 직전 명령 유지. 시작 시 직전 명령은 정지다.
            self.status = "HOLD"
        return self.command


def drive(bot, speeds):
    if len(speeds) != 4 or any(abs(speed) > 150 for speed in speeds):
        raise ValueError("바퀴 속도 한계 150 초과")
    for number, speed in enumerate(speeds):
        bot.Ctrl_Muto(number, speed)


def stop_all(bot):
    errors = []
    for number in range(4):
        try:
            bot.Ctrl_Muto(number, 0)
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError("정지 명령 실패. 라즈봇 전원을 확인하세요.") from errors[0]


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


def check_sensors(bot, seconds):
    print("읽기 전용 점검: 바퀴 명령 없음. 안내받은 종이 위치를 유지하세요.", flush=True)
    last = None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        bits = read_sensors(bot)
        if bits != last:
            print("x1 x2 x3 x4 = " + " ".join(map(str, bits)), flush=True)
            last = bits
        time.sleep(0.05)


def follow(bot, key_pressed):
    follower = LineFollower()
    previous = None
    try:
        while True:
            if key_pressed():
                print("키 입력: 정지합니다.", flush=True)
                break
            bits = read_sensors(bot)
            if key_pressed():
                break
            speeds = follower.step(bits, time.monotonic())
            drive(bot, speeds)
            current = (bits, follower.status, speeds)
            if current != previous:
                print(f"x1..x4={bits} {follower.status} wheels={speeds}", flush=True)
                previous = current
            if follower.stopped:
                print("줄 놓침 정지. 다시 출발하려면 프로그램을 재실행하세요.", flush=True)
                break
            time.sleep(0.02)
    finally:
        stop_all(bot)


def main():
    parser = argparse.ArgumentParser(description="줄 따라가기 / 읽기 전용 센서 점검")
    parser.add_argument("--check-sensors", action="store_true", help="바퀴 명령 없이 센서만 읽기")
    parser.add_argument("--seconds", type=float, default=10, help="읽기 전용 점검 시간")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("점검 시간은 양수여야 합니다.")
    from Raspbot_Lib import Raspbot
    if args.check_sensors:
        # 생성자는 I2C 연결만 만든다. 이 경로는 정지 명령도 포함해 모터 호출이 없다.
        check_sensors(Raspbot(), args.seconds)
        return
    try:
        with keyboard_stop() as key_pressed:
            follow(Raspbot(), key_pressed)
    except KeyboardInterrupt:
        print("Ctrl+C: 정지했습니다.", flush=True)


if __name__ == "__main__":
    main()
