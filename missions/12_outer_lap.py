#!/usr/bin/env python3
# 지시서: specs/12-outer-lap.md
"""바깥 고리 한 바퀴: 바깥(왼쪽) 노란 선 기준 고정 간격으로 시계 방향 한 바퀴 돌고, 모서리 4회 뒤 START 꺾쇠에서 정지.
아래쪽 변의 빨강 칸에서는 감속, 파랑 칸에서는 가속한다(바퀴 값에 배율).

07(카메라 노란 차선)의 선 찾기·비례 조향·속도차 제한·녹화 부품을 그대로 쓴다. --check는 모터 명령을 보내지 않는다.
단계: WAITLINES(첫 선 대기) → LAP(주행·모서리 세기) → ARMED(모서리 4회, START 감시) → FINISH / TIMEOUT / LOSTSTOP / FRAMESTOP
"""
import argparse
from collections import deque
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'missions' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


line = load('outer_lap_line', '02_line.py')
video = load('outer_lap_video', '03_video.py')
lane = load('outer_lap_camera', '07_camera_lane.py')
STOP = lane.STOP
KST = ZoneInfo('Asia/Seoul')
ROI_TOP, ROI_BOTTOM = lane.ROI_TOP, lane.ROI_BOTTOM      # 07과 같은 띠 (120~299행)
FLOOR_TOP, FLOOR_BOTTOM = 300, 400                       # START 무늬를 보는 아래 띠
FLOOR_CENTER = (280, 360)                                # 바로 앞 창(열)
WHITE_S_MAX, WHITE_V_MIN = 60, 150                       # 흰색 기준
YELLOW_GUARD = 7                                         # 노란 선 번짐을 흰색에서 빼는 두께(px)
EDGE_MARGIN = 12
DEFAULT_WIDTH = 460                                      # 두 선을 보기 전 도로 폭 기본값(px) — 07 실측 474 참고, --check로 조정
MIN_WIDTH, MAX_WIDTH = 150, 600                          # 폭으로 저장할 범위
LEFT_NAME_X = 380                                        # 선이 하나뿐일 때 아랫변 x가 이 값 이하면 왼쪽(바깥) 선
LEFT_TRACK_PX, LEFT_TRACK_SEC = 150, 0.5                 # 직전 왼쪽 선 근처면 같은 선
CONFIRM = 2
START_GRACE = 3.0                                        # 시작 뒤 첫 노란 선 대기
HOLD_MAX = 0.5                                           # 꺾은 적 없이 선을 놓쳤을 때 직전 명령 유지
SEARCH_MAX = 1.0                                         # 07과 같은 탐색 시간
SEARCH_RIGHT, SEARCH_LEFT = (60, 60, 0, 0), (0, 0, 60, 60)
CORNER_WINDOW = 5.0                                      # 회전 누적 창(초)
CORNER_TURN_SUM = 40.0                                   # 창 안 (왼쪽−오른쪽)×초 합이 이 값 이상이면 우회전 모서리 1회 — 실측 보정
CORNER_COOLDOWN = 4.0                                    # 모서리 1회 뒤 다시 세지 않는 시간
LAP_CORNERS = 4
START_CENTER, START_WIDE = 0.5, 0.25                     # START 판단: 바로 앞 창·넓은 창 흰 비율
FINISH_MAX = 15.0                                        # 모서리 4회 뒤 START를 기다리는 최대 시간
ZONE_MIN_RATIO = 0.3                                     # 아래 띠에서 이 비율 이상 빨강/파랑이면 칸 위
ZONE_HOLD = 0.5                                          # 칸이 안 보여도 이만큼 배율 유지(깜빡임 방지)
SLOW_FACTOR, FAST_FACTOR = 0.8, 1.3                      # 빨강 칸 감속·파랑 칸 가속 배율 (조 확정 전 제안값)
ZONE_LIGHT = {'RED': 0, 'BLUE': 2}                       # 칸 위에서 불빛 색 (0 빨강 · 2 파랑)
FRAME_STALE = 0.5


# ---------- 카메라가 보는 것 ----------
def _check_frame(frame):
    if frame is None or frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise ValueError('입력 화면은 640x480 uint8 BGR이어야 합니다.')


def _white_mask(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 100), (40, 255, 255))
    white = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (180, WHITE_S_MAX, 255))
    white[cv2.dilate(yellow, np.ones((1, 2 * YELLOW_GUARD + 1), np.uint8)) > 0] = 0
    return white > 0


def floor_white_ratio(frame, lines):
    """아래 띠(300~399행) 비율들: 흰색 넓은 창(160~오른쪽 선 안쪽)·바로 앞 창(280~359열) — START 꺾쇠는 둘 다 높다 —
    과 같은 넓은 창의 빨강·파랑 비율(아래쪽 변의 감속·가속 칸)."""
    _check_frame(frame)
    region = frame[FLOOR_TOP:FLOOR_BOTTOM]
    mask = _white_mask(region)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, (0, 120, 80), (8, 255, 255)) | cv2.inRange(hsv, (168, 120, 80), (180, 255, 255))) > 0
    blue = cv2.inRange(hsv, (95, 120, 80), (125, 255, 255)) > 0
    hi = 480 if lines['right'] is None else min(480, int(round(lines['right'])) - EDGE_MARGIN)
    ok = hi - 160 >= 20
    return {'wide': round(float(mask[:, 160:hi].mean()), 4) if ok else 0.0,
            'center': round(float(mask[:, FLOOR_CENTER[0]:FLOOR_CENTER[1]].mean()), 4),
            'red': round(float(red[:, 160:hi].mean()), 4) if ok else 0.0,
            'blue': round(float(blue[:, 160:hi].mean()), 4) if ok else 0.0}


def _bottom_x(component):
    return float(component.xs[component.ys == component.ys.max()].mean())


def analyze(frame, previous=None, width=None, last_name=None, last_left=None):
    """노란 선 두 개면 07처럼 x순으로 왼쪽·오른쪽. 하나면 12번 규칙: 직전 왼쪽 선 근처(0.5초 안·150px 안)면 왼쪽,
    아니면 가장 아래 행 x가 380 이하면 왼쪽, 그 외 오른쪽. 조향용으로 각 선의 가장 아래 행 x(left_bottom/right_bottom)도 준다.
    last_left=(x, 본 지 몇 초)."""
    components = lane.find_components(frame)
    lines = lane.assign_lines(components, previous, width, last_name)
    lines = dict(lines, left_bottom=None, right_bottom=None)
    if not components:
        return lines, floor_white_ratio(frame, lines)
    parts = sorted(components, key=lambda c: c[0])
    both = lines['left'] is not None and lines['right'] is not None
    if both and len(parts) >= 2:
        lines['left_bottom'], lines['right_bottom'] = _bottom_x(parts[0]), _bottom_x(parts[-1])
    else:
        xs = np.concatenate([c.xs for c in components])
        ys = np.concatenate([c.ys for c in components])
        x = float(xs[ys == ys.max()].mean())
        if last_left is not None and last_left[1] <= LEFT_TRACK_SEC and abs(x - last_left[0]) <= LEFT_TRACK_PX:
            name = 'left'
        else:
            name = 'left' if x <= LEFT_NAME_X else 'right'
        lines = dict(left=None, right=None, left_count=0, right_count=0, left_bottom=None, right_bottom=None, named_by='12')
        lines[name], lines[name + '_count'], lines[name + '_bottom'] = x, int(len(xs)), x
    return lines, floor_white_ratio(frame, lines)


# ---------- 장치 없는 판단 ----------
class OuterLapController:
    """선 위치·흰 비율·시각만 받아 단계·모드와 바퀴 값을 정한다. 장치 호출은 없다."""

    def __init__(self):
        self.state = 'WAITLINES'
        self.mode = 'WAIT'
        self.speeds = STOP
        self.stopped = False
        self.reason = ''
        self.notes = []
        self.previous = None
        self.single_name = None
        self.width = DEFAULT_WIDTH
        self.width_measured = False
        self.center = None
        self.centers = deque(maxlen=3)
        self.last_turn = None
        self.lines_last = None
        self.left_last = None
        self.started_at = None
        self.phase_since = None
        self.last_frame = None
        self.last_update = None
        self.turn_window = deque()
        self.turn_sum = 0.0
        self.corners = 0
        self.corner_last = None
        self.start_count = 0
        self.zone = None                # 'RED'(감속) · 'BLUE'(가속) · None
        self.zone_last = None
        self.factor = 1.0
        self.lines = dict(left=None, right=None, left_count=0, right_count=0, left_bottom=None, right_bottom=None)
        self.white = {'wide': 0.0, 'center': 0.0, 'red': 0.0, 'blue': 0.0}

    def tick(self, now):
        if self.stopped:
            return STOP
        if self.last_frame is not None and now - self.last_frame > FRAME_STALE:
            return self._stop('FRAMESTOP')
        return self.speeds

    def left_hint(self, now):
        return None if self.left_last is None else (self.left_last[0], now - self.left_last[1])

    def update(self, lines, white, now):
        self.tick(now)
        if self.stopped:
            return STOP
        dt = 0.0 if self.last_update is None else max(0.0, now - self.last_update)
        self.last_update = self.last_frame = now
        if not isinstance(white, dict):
            white = {'wide': float(white), 'center': float(white)}
        white = dict({'red': 0.0, 'blue': 0.0}, **white)
        self.lines, self.white = dict(lines), white
        self._update_zone(white, now)
        left, right = lines['left'], lines['right']
        if left is not None and right is not None:
            self.previous = dict(left=left, right=right)
        elif self.previous is not None:
            for name in ('left', 'right'):
                if lines[name] is not None:
                    self.previous[name] = lines[name]
        if (left is None) != (right is None):
            self.single_name = 'left' if left is not None else 'right'
        if self.started_at is None:
            self.started_at = now
        if self.phase_since is None:
            self.phase_since = now
        if left is not None or right is not None or lines.get('unassigned'):
            self.lines_last = now
        if left is not None:
            self.left_last = (lines.get('left_bottom') or left, now)
        if left is not None and right is not None and MIN_WIDTH <= right - left <= MAX_WIDTH:
            self.width = right - left
            self.width_measured = True
        handler = {'WAITLINES': self._waitlines, 'LAP': self._lap, 'ARMED': self._armed}[self.state]
        self.speeds = STOP if self.stopped else handler(lines, white, now)
        self._count_corners(now, dt)
        return self.speeds

    # ---- 공통 ----
    def _update_zone(self, white, now):
        """빨강 칸이면 감속, 파랑 칸이면 가속. 칸이 안 보여도 0.5초는 유지한다."""
        seen = 'RED' if white['red'] >= ZONE_MIN_RATIO else 'BLUE' if white['blue'] >= ZONE_MIN_RATIO else None
        if seen is not None:
            self.zone, self.zone_last = seen, now
        elif self.zone is not None and now - self.zone_last > ZONE_HOLD:
            self.zone = None
        self.factor = {'RED': SLOW_FACTOR, 'BLUE': FAST_FACTOR}.get(self.zone, 1.0)

    def _scaled(self, speeds):
        return tuple(max(0, min(lane.MAX_SPEED, round(v * self.factor))) for v in speeds)

    def _limited(self, target):
        return lane.limit_difference(target, self.speeds)

    def _enter(self, state, now):
        self.state = state
        self.phase_since = now
        self.start_count = 0

    def _stop(self, reason):
        self.state = reason
        self.mode = 'STOP'
        self.reason = reason
        self.stopped = True
        self.speeds = STOP
        self.center = None
        return STOP

    def _follow(self, lines, now):
        """바깥(왼쪽) 선 기준: 중심 = 왼쪽 선 아랫변 x + 폭/2. 왼쪽 선이 없으면 오른쪽 선 − 폭/2. 둘 다 없으면 탐색."""
        left = lines.get('left_bottom') if lines.get('left_bottom') is not None else lines['left']
        right = lines.get('right_bottom') if lines.get('right_bottom') is not None else lines['right']
        if left is not None:
            raw, self.mode = left + self.width / 2, 'TRACKLEFT' if right is None else 'TRACKBOTH'
        elif right is not None:
            raw, self.mode = right - self.width / 2, 'TRACKRIGHT'
        else:
            return self._lost(now)
        self.centers.append(max(0.0, min(640.0, raw)))
        self.center = sum(self.centers) / len(self.centers)
        speeds = self._limited(self._scaled(lane.wheel_speeds(self.center)))
        if speeds[0] > speeds[2]:
            self.last_turn = 'RIGHT'
        elif speeds[0] < speeds[2]:
            self.last_turn = 'LEFT'
        return speeds

    def _lost(self, now):
        self.center = None
        age = now - self.lines_last
        if self.last_turn is None:
            if age <= HOLD_MAX:
                self.mode = 'HOLD'
                return self.speeds
            return self._stop('LOSTSTOP')
        if age <= SEARCH_MAX:
            self.mode = 'SEARCH' + self.last_turn
            return self._limited(SEARCH_RIGHT if self.last_turn == 'RIGHT' else SEARCH_LEFT)
        return self._stop('LOSTSTOP')

    def _count_corners(self, now, dt):
        """바퀴 명령 (왼쪽−오른쪽)×초를 최근 창에서 더해 우회전 모서리를 센다."""
        if dt > 0:
            self.turn_window.append((now, (self.speeds[0] - self.speeds[2]) * dt))
        while self.turn_window and now - self.turn_window[0][0] > CORNER_WINDOW:
            self.turn_window.popleft()
        self.turn_sum = round(sum(v for _, v in self.turn_window), 2)
        cooled = self.corner_last is None or now - self.corner_last >= CORNER_COOLDOWN
        if self.turn_sum >= CORNER_TURN_SUM and cooled and self.state in ('LAP', 'ARMED'):
            self.corners += 1
            self.corner_last = now
            self.turn_window.clear()
            self.turn_sum = 0.0

    # ---- 단계 ----
    def _waitlines(self, lines, white, now):
        if lines['left'] is not None or lines['right'] is not None or lines.get('unassigned'):
            self._enter('LAP', now)
            return self._lap(lines, white, now)
        if now - self.phase_since >= START_GRACE:
            self.notes.append('NO_LINES_AT_START')
            return self._stop('LOSTSTOP')
        self.mode = 'WAIT'
        return STOP

    def _lap(self, lines, white, now):
        if self.corners >= LAP_CORNERS:
            self._enter('ARMED', now)
            return self._armed(lines, white, now)
        return self._follow(lines, now)

    def _armed(self, lines, white, now):
        hit = white['center'] >= START_CENTER and white['wide'] >= START_WIDE
        self.start_count = self.start_count + 1 if hit else 0
        if self.start_count >= CONFIRM:
            return self._stop('FINISH')
        if now - self.phase_since >= FINISH_MAX:
            self.notes.append('FINISH_TIMEOUT')
            return self._stop('TIMEOUT')
        return self._follow(lines, now)

    def report(self):
        return dict(self.lines, width=self.width, width_measured=self.width_measured, center=self.center,
                    speeds=self.speeds, state=self.state, mode=self.mode, stopped=self.stopped, reason=self.reason,
                    corners=self.corners, turn_sum=self.turn_sum, white=self.white['wide'],
                    white_center=self.white['center'], red=self.white['red'], blue=self.white['blue'],
                    zone=self.zone, factor=self.factor, last_turn=self.last_turn, notes=list(self.notes))


# ---------- 화면 · 기록 ----------
def annotated(frame, report):
    out = lane.annotated(frame, report)
    cv2.rectangle(out, (0, FLOOR_TOP), (639, FLOOR_BOTTOM - 1), (200, 200, 200), 1)
    cv2.rectangle(out, (FLOOR_CENTER[0], FLOOR_TOP), (FLOOR_CENTER[1] - 1, FLOOR_BOTTOM - 1), (200, 200, 200), 1)
    cv2.putText(out, f'{report["mode"]} CORNERS {report["corners"]} TURN {report["turn_sum"]:.0f} '
                     f'WHITE {report["white"]:.2f} C {report["white_center"]:.2f} '
                     f'ZONE {report["zone"] or "-"} x{report["factor"]:.1f}', (8, 380),
                cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class Display(lane.LatestDisplay):
    def update(self, frame, report):
        marked = annotated(frame, report)
        with self.lock:
            self.frame = marked
            self.sequence += 1


FIELDS = ('경과초', '화면순번', '단계', '모드', '왼쪽선_x', '오른쪽선_x', '차선폭_px', '차선중심_x', '모서리횟수',
          '회전누적', '바닥흰비율', '바로앞흰비율', '빨강비율', '파랑비율', '구간', '배율', '바퀴값', '실행모드')


def row_values(report, elapsed, sequence, check):
    return dict(zip(FIELDS, (round(elapsed, 4), sequence, report['state'], report['mode'], report['left'],
                             report['right'], report['width'], report['center'], report['corners'],
                             report['turn_sum'], report['white'], report['white_center'], report['red'],
                             report['blue'], report['zone'] or '', report['factor'],
                             json.dumps(report['speeds']), '읽기전용' if check else '주행')))


def run_loop(camera, recorder, display, key_pressed, stream, check=False, bot=None):
    if not check and bot is None:
        raise ValueError('주행 모드에는 바퀴 장치가 필요합니다.')
    controller = OuterLapController()
    started = time.monotonic()
    next_print = started
    sequence = None
    frame = None
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    last_record = None
    last_zone = None
    reason = '키 입력'
    try:
        while True:
            if key_pressed():
                break
            controller.tick(time.monotonic())
            candidate, new_sequence = camera.snapshot()
            received_at = time.monotonic()
            if new_sequence != sequence:
                lines, white = analyze(candidate, controller.previous, controller.width, controller.single_name,
                                       controller.left_hint(received_at))
                controller.update(lines, white, received_at)
                frame, sequence = candidate, new_sequence
            if key_pressed():
                break
            now = time.monotonic()
            controller.tick(now)
            report = controller.report()
            if not check:
                line.drive(bot, report['speeds'])
                if report['zone'] != last_zone:                   # 칸에 들어가고 나갈 때만 불빛 명령
                    if report['zone'] is None:
                        bot.Ctrl_WQ2812_ALL(0, 0)
                    else:
                        bot.Ctrl_WQ2812_ALL(1, ZONE_LIGHT[report['zone']])
                    last_zone = report['zone']
            if frame is not None:
                display.update(frame, report)
            if recorder is not None:
                recorder.set_status(report['state'])
                recorder.check()
            current = (sequence, report['state'], report['mode'], report['corners'], report['zone'])
            if current != last_record:
                writer.writerow(row_values(report, now - started, sequence, check))
                stream.flush()
                last_record = current
            if now >= next_print:
                print('한 바퀴 계산: ' + json.dumps(report, ensure_ascii=False), flush=True)
                next_print = now + 1
            if controller.stopped:
                if not check:
                    reason = report['state']
                    break
                print(f'읽기 전용: 정지 조건 {report["state"]} (기록 {controller.notes}) → 새로 판단을 시작해 계속 관찰합니다.', flush=True)
                controller = OuterLapController()
                last_record = None
            time.sleep(.01)
    except KeyboardInterrupt:
        reason = 'Ctrl+C'
    finally:
        if not check:
            line.stop_all(bot)
    return reason, controller.report()


def beep(bot, times=1):
    for _ in range(times):
        bot.Ctrl_BEEP_Switch(1)
        time.sleep(0.2)
        bot.Ctrl_BEEP_Switch(0)
        time.sleep(0.2)


def append_run(path, trial, reason, elapsed, raw, paths, report):
    path = Path(path)
    default = ['회차', '코드 버전(커밋 번호)', '시도', '완주', '이탈', '시간(초)', '손댄 횟수', '메모']
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline='', encoding='utf-8') as f:
            fields = csv.DictReader(f).fieldnames
    else:
        fields = default
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
        if subprocess.check_output(['git', 'status', '--porcelain', '--', 'missions/12_outer_lap.py'],
                                   cwd=ROOT, text=True).strip():
            commit += '+미커밋'
    except (subprocess.CalledProcessError, OSError):
        commit = '미확인'
    row = dict.fromkeys(fields, '')
    values = {'회차': trial, '코드 버전(커밋 번호)': commit, '시도': '12 바깥 고리 한 바퀴',
              '완주': '사용자 확인 대기', '이탈': '사용자 확인 대기', '시간(초)': elapsed, '손댄 횟수': '사용자 확인 대기',
              '메모': f'{reason}; 모서리 {report.get("corners")}회; 기록 {report.get("notes") or "없음"}; '
                      f'실제 동작 사용자 확인 대기; 원시 기록 {raw}; 영상 {paths}',
              '날짜': datetime.now(KST).strftime('%Y-%m-%d'), '프로그램': 'missions/12_outer_lap.py',
              '속도': lane.BASE_SPEED, '원시기록': str(raw), '종료후전체정지확인': '사용자 확인 대기'}
    row.update({k: v for k, v in values.items() if k in fields})
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='바깥 고리 한 바퀴 (바깥 노란 선 기준) / --check는 모터 명령 없음')
    parser.add_argument('--check', action='store_true',
                        help='모터 없이 선·폭·중심·바퀴 계산·회전 누적·모서리·흰 비율을 1초마다 출력 (정지 조건이 와도 계속 관찰)')
    parser.add_argument('--trial', default='', help='runs.csv에 남길 회차 이름 (주행 모드)')
    args = parser.parse_args()
    camera = recorder = bot = None
    entered = False
    started = time.monotonic()
    reason, report = '시작 준비 오류', {}
    stamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S_%f')
    path = ROOT / 'results' / f'outer_lap_{stamp}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    display = Display()
    try:
        with line.keyboard_stop() as key_pressed:
            if not args.check:
                from Raspbot_Lib import Raspbot
                bot = Raspbot()
                line.stop_all(bot)
            camera = video.LatestCamera()
            camera.start()
            first, _ = camera.snapshot()
            display.update(first, OuterLapController().report())
            recorder = lane.LaneVideo(display)
            recorder.start()
            if bot is not None:
                beep(bot, 1)                                  # 출발 전 삐 (06 수칙)
            with path.open('x', newline='', encoding='utf-8') as stream:
                entered = True
                reason, report = run_loop(camera, recorder, display, key_pressed, stream, args.check, bot)
            if reason == 'FINISH' and bot is not None:
                bot.Ctrl_WQ2812_ALL(1, 1)
                beep(bot, 2)
            print(f'종료: {reason}, 모서리 {report.get("corners")}회, 기록: {path}. 실제 정지 위치는 사용자 확인이 필요합니다.', flush=True)
    except KeyboardInterrupt:
        reason = 'Ctrl+C'
    finally:
        errors = []
        if bot is not None:
            cleanups = [lambda: bot.Ctrl_BEEP_Switch(0), lambda: bot.Ctrl_WQ2812_ALL(0, 0)]
            if not entered:
                cleanups.insert(0, lambda: line.stop_all(bot))
            for cleanup in cleanups:
                try:
                    cleanup()
                except Exception as exc:
                    errors.append(exc)
        for resource in (recorder, camera):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(exc)
        if not args.check:
            append_run(ROOT / 'results/runs.csv', args.trial or stamp, reason, round(time.monotonic() - started, 4),
                       path, [] if recorder is None else [str(p) for p in recorder.paths], report)
        if errors:
            raise RuntimeError('종료 정리 오류: ' + '; '.join(map(str, errors)))


if __name__ == '__main__':
    main()
