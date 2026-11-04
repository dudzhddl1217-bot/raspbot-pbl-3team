#!/usr/bin/env python3
# 지시서: specs/11-right-lane-turn.md
"""로터리 출구 뒤 두 차선 길: 오른쪽 차선 유지 → 우회전 → START 정지. --check는 모터 명령을 보내지 않는다.

07(카메라 노란 차선)의 선 찾기·비례 조향·녹화 부품을 그대로 쓰고, 흰 실선(차선 구분선)과 바닥 흰 비율을 더해
WAITLANE → RIGHTLANE → TURNRIGHT → FINISHRUN → FINISH 단계로 판단한다.
"""
import argparse
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'missions' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


line = load('right_lane_line', '02_line.py')
video = load('right_lane_video', '03_video.py')
lane = load('right_lane_camera', '07_camera_lane.py')
STOP = lane.STOP
KST = ZoneInfo('Asia/Seoul')
ROI_TOP, ROI_BOTTOM = lane.ROI_TOP, lane.ROI_BOTTOM      # 07과 같은 띠 (120~299행)
FLOOR_TOP, FLOOR_BOTTOM = 300, 400                       # START 무늬를 보는 아래 띠
WHITE_S_MAX, WHITE_V_MIN = 50, 170                       # 흰색 기준
YELLOW_GUARD = 7                                         # 노란 선 가장자리 번짐을 흰색에서 빼는 두께(px)
EDGE_MARGIN = 12                                         # 노란 선 안쪽으로 이만큼부터 흰 실선을 찾는다
DIVIDER_MAX_RUN = 30                                     # 한 행에서 흰 실선로 볼 최대 가로 길이(px)
DIVIDER_MIN_ROWS = 0.6                                   # 띠의 이 비율 이상 행에 보여야 흰 실선
MIN_RIGHT_WIDTH = 40                                     # 오른쪽 차선 폭으로 인정할 최소값(px)
CONFIRM = 2                                              # 새 화면 연속 횟수
DIVIDER_GONE = 0.3                                       # 흰 실선이 이만큼 안 보이면 끝난 것
RIGHTLANE_MIN = 1.0                                      # RIGHTLANE 최소 유지 시간
TURN_ARC = (55, 55, 10, 10)                              # 오른쪽 선이 안 보일 때 오른쪽으로 호 그리기
TURN_MAX = 4.0                                           # 우회전 최대 시간
TURN_LOST_MAX = 2.0                                      # 우회전 중 어떤 선도 안 보이면 정지
WIDE_RATIO = 1.4                                         # 두 선 폭이 오른쪽 차선 폭의 이 배수면 큰 길
START_WHITE = 0.25                                       # START 무늬 판단 흰 비율
FINISH_MAX = 8.0                                         # FINISHRUN 최대 시간
FRAME_STALE = 0.5                                        # 07과 같은 화면 갱신 중단 한계


# ---------- 카메라가 보는 것 ----------
def _masks(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 100), (40, 255, 255))
    white = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (180, WHITE_S_MAX, 255))
    kernel = np.ones((1, 2 * YELLOW_GUARD + 1), np.uint8)
    white[cv2.dilate(yellow, kernel) > 0] = 0                 # 노란 선 번짐은 흰색에서 뺀다
    return white > 0


def _check_frame(frame):
    if frame is None or frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise ValueError('입력 화면은 640x480 uint8 BGR이어야 합니다.')


def _window(lines, lo_default, hi_default):
    lo = lo_default if lines['left'] is None else int(round(lines['left'])) + EDGE_MARGIN
    hi = hi_default if lines['right'] is None else int(round(lines['right'])) - EDGE_MARGIN
    return max(0, lo), min(640, hi)


def find_divider(frame, lines):
    """두 노란 선 사이에서 세로로 긴 가는 흰 실선을 찾는다. 행마다 가장 왼쪽의 가는 흰 묶음을 모아 중앙값을 낸다."""
    _check_frame(frame)
    result = {'divider': None, 'divider_rows': 0.0}
    lo, hi = _window(lines, 0, 640)
    if hi - lo < 4:
        return result
    window = _masks(frame[ROI_TOP:ROI_BOTTOM])[:, lo:hi]
    centers = []
    for row in window:
        if not row.any():
            continue
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[0::2], edges[1::2]
        lengths = ends - starts
        thin = np.flatnonzero((lengths >= 2) & (lengths <= DIVIDER_MAX_RUN))
        if len(thin):
            i = thin[0]                                   # 가장 왼쪽 = 오른쪽 차선의 왼쪽 경계
            centers.append((starts[i] + ends[i] - 1) / 2)
    coverage = len(centers) / window.shape[0]
    if coverage >= DIVIDER_MIN_ROWS:
        result.update(divider=float(np.median(centers) + lo), divider_rows=round(coverage, 3))
    else:
        result['divider_rows'] = round(coverage, 3)
    return result


def floor_white_ratio(frame, lines):
    """아래 띠(300~399행)에서 두 노란 선 사이 흰 점 비율. START 갈매기 무늬에 닿으면 크게 오른다."""
    _check_frame(frame)
    lo, hi = _window(lines, 160, 480)
    if hi - lo < 20:
        return 0.0
    return float(_masks(frame[FLOOR_TOP:FLOOR_BOTTOM])[:, lo:hi].mean())


def analyze(frame):
    lines = lane.find_lines(frame)
    divider = find_divider(frame, lines)
    return lines, divider, floor_white_ratio(frame, lines)


# ---------- 장치 없는 판단 ----------
class RightLaneController:
    """선 위치·흰 실선·흰 비율·시각만 받아 단계와 바퀴 값을 정한다. 장치 호출은 없다."""

    def __init__(self):
        self.lane = lane.LaneController()
        self.state = 'WAITLANE'
        self.speeds = STOP
        self.stopped = False
        self.reason = ''
        self.notes = []
        self.right_width = None
        self.full_width = None
        self.center = None
        self.divider_seen = 0
        self.divider_last = None
        self.lines_last = None
        self.phase_since = None
        self.wide_count = 0
        self.start_count = 0
        self.last_frame = None
        self.lines = dict(left=None, right=None, left_count=0, right_count=0)
        self.divider = None
        self.white = 0.0

    # 07과 같은 화면 갱신 중단 규칙
    def tick(self, now):
        if self.stopped:
            return STOP
        if self.last_frame is not None and now - self.last_frame > FRAME_STALE:
            return self._stop('FRAMESTOP')
        return self.speeds

    def update(self, lines, divider, white, now):
        self.tick(now)
        if self.stopped:
            return STOP
        self.last_frame = now
        self.lines, self.divider, self.white = dict(lines), divider, white
        if self.phase_since is None:
            self.phase_since = now
        if lines['left'] is not None or lines['right'] is not None or divider is not None:
            self.lines_last = now
        if divider is not None:
            self.divider_seen += 1
            self.divider_last = now
        else:
            self.divider_seen = 0
        handler = {'WAITLANE': self._waitlane, 'RIGHTLANE': self._rightlane,
                   'TURNRIGHT': self._turnright, 'FINISHRUN': self._finishrun}[self.state]
        self.speeds = STOP if self.stopped else handler(lines, divider, white, now)
        return self.speeds

    def _enter(self, state, now):
        self.state = state
        self.phase_since = now
        self.wide_count = 0
        self.start_count = 0

    def _stop(self, reason):
        self.state = reason
        self.reason = reason
        self.stopped = True
        self.speeds = STOP
        self.center = None
        return STOP

    def _waitlane(self, lines, divider, white, now):
        speeds = self.lane.update(lines, now)
        self.center = self.lane.center
        if self.lane.stopped:
            return self._stop(self.lane.state)
        if self.divider_seen >= CONFIRM:
            right = lines['right']
            if right is not None and right - divider >= MIN_RIGHT_WIDTH:
                self.right_width = right - divider
            elif self.lane.width is not None:
                self.right_width = self.lane.width / 2
            else:
                return speeds                               # 폭을 모르면 아직 넘어가지 않는다
            self.full_width = self.lane.width
            self._enter('RIGHTLANE', now)
            return self._rightlane(lines, divider, white, now)
        return speeds

    def _rightlane(self, lines, divider, white, now):
        right = lines['right']
        if divider is not None and right is not None and right - divider >= MIN_RIGHT_WIDTH:
            self.right_width = right - divider
            self.center = (divider + right) / 2
        elif divider is not None:
            self.center = divider + self.right_width / 2
        elif right is not None:
            self.center = right - self.right_width / 2
        else:
            self.center = None
        gone = now - self.divider_last > DIVIDER_GONE
        if gone and now - self.phase_since >= RIGHTLANE_MIN:
            self._enter('TURNRIGHT', now)
            return self._turnright(lines, divider, white, now)
        if self.center is None:
            return self.speeds                              # 잠깐 놓침: 직전 명령 유지
        return lane.wheel_speeds(self.center)

    def _turnright(self, lines, divider, white, now):
        left, right = lines['left'], lines['right']
        elapsed = now - self.phase_since
        wide = left is not None and right is not None and right - left >= WIDE_RATIO * self.right_width
        self.wide_count = self.wide_count + 1 if wide else 0
        if self.wide_count >= CONFIRM or elapsed >= TURN_MAX:
            if self.wide_count < CONFIRM:
                self.notes.append('TURN_TIMEOUT')
            self.lane = lane.LaneController()                # 큰 길 폭을 다시 잰다
            self.lane.width = self.full_width                # 한쪽 선만 보여도 바로 갈 수 있게
            self._enter('FINISHRUN', now)
            return self._finishrun(lines, divider, white, now)
        if self.lines_last is None or now - self.lines_last > TURN_LOST_MAX:
            return self._stop('LOSTSTOP')
        if right is not None:
            self.center = right - self.right_width / 2
            return lane.wheel_speeds(self.center)
        self.center = None
        return TURN_ARC

    def _finishrun(self, lines, divider, white, now):
        speeds = self.lane.update(lines, now)
        self.center = self.lane.center
        if self.lane.stopped:
            return self._stop(self.lane.state)
        self.start_count = self.start_count + 1 if white >= START_WHITE else 0
        if self.start_count >= CONFIRM:
            return self._stop('FINISH')
        if now - self.phase_since >= FINISH_MAX:
            self.notes.append('FINISH_TIMEOUT')
            return self._stop('TIMEOUT')
        return speeds

    def report(self):
        return dict(self.lines, divider=self.divider, white=round(self.white, 4), right_width=self.right_width,
                    full_width=self.full_width, width=self.right_width if self.state in ('RIGHTLANE', 'TURNRIGHT')
                    else self.lane.width, center=self.center, speeds=self.speeds, state=self.state,
                    stopped=self.stopped, reason=self.reason, notes=list(self.notes))


# ---------- 화면 · 기록 ----------
def annotated(frame, report):
    out = lane.annotated(frame, report)
    if report.get('divider') is not None:
        x = round(report['divider'])
        cv2.line(out, (x, ROI_TOP), (x, ROI_BOTTOM - 1), (255, 255, 255), 2)
        cv2.putText(out, f'DIV {report["divider"]:.1f}', (max(2, min(520, x)), ROI_BOTTOM + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, FLOOR_TOP), (639, FLOOR_BOTTOM - 1), (200, 200, 200), 1)
    cv2.putText(out, f'WHITE {report["white"]:.2f} RW {report["right_width"]}', (8, 380),
                cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class Display(lane.LatestDisplay):
    def update(self, frame, report):
        marked = annotated(frame, report)
        with self.lock:
            self.frame = marked
            self.sequence += 1


FIELDS = ('경과초', '화면순번', '단계', '왼쪽선_x', '오른쪽선_x', '흰실선_x', '흰실선_행비율',
          '오른쪽차선폭_px', '차선중심_x', '바닥흰비율', '바퀴값', '모드')


def row_values(report, divider, elapsed, sequence, check):
    return dict(zip(FIELDS, (round(elapsed, 4), sequence, report['state'], report['left'], report['right'],
                             report['divider'], divider['divider_rows'], report['right_width'],
                             report['center'], report['white'], json.dumps(report['speeds']),
                             '읽기전용' if check else '주행')))


def run_loop(camera, recorder, display, key_pressed, stream, check=False, bot=None):
    if not check and bot is None:
        raise ValueError('주행 모드에는 바퀴 장치가 필요합니다.')
    controller = RightLaneController()
    started = time.monotonic()
    next_print = started
    sequence = None
    frame = None
    divider = {'divider': None, 'divider_rows': 0.0}
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    last_record = None
    reason = '키 입력'
    try:
        while True:
            if key_pressed():
                break
            controller.tick(time.monotonic())
            candidate, new_sequence = camera.snapshot()
            received_at = time.monotonic()
            if new_sequence != sequence:
                lines, divider, white = analyze(candidate)
                controller.update(lines, divider['divider'], white, received_at)
                frame, sequence = candidate, new_sequence
            if key_pressed():
                break
            now = time.monotonic()
            controller.tick(now)
            report = controller.report()
            if not check:
                line.drive(bot, report['speeds'])
            if frame is not None:
                display.update(frame, report)
            if recorder is not None:
                recorder.set_status(report['state'])
                recorder.check()
            current = (sequence, report['state'])
            if current != last_record:
                writer.writerow(row_values(report, divider, now - started, sequence, check))
                stream.flush()
                last_record = current
            if now >= next_print:
                print('차선 계산: ' + json.dumps(report, ensure_ascii=False), flush=True)
                next_print = now + 1
            if controller.stopped:
                reason = report['state']
                break
            time.sleep(.01)
    except KeyboardInterrupt:
        reason = 'Ctrl+C'
    finally:
        if not check:
            line.stop_all(bot)
    return reason, controller.report()


def celebrate(bot):
    """START 정지 뒤: 삐 0.2초 두 번, 불빛 초록. 불빛은 종료 정리에서 끈다."""
    bot.Ctrl_WQ2812_ALL(1, 1)
    for _ in range(2):
        bot.Ctrl_BEEP_Switch(1)
        time.sleep(0.2)
        bot.Ctrl_BEEP_Switch(0)
        time.sleep(0.2)


def append_run(path, trial, reason, elapsed, raw, paths, notes):
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
        if subprocess.check_output(['git', 'status', '--porcelain', '--', 'missions/11_right_lane_turn.py'],
                                   cwd=ROOT, text=True).strip():
            commit += '+미커밋'
    except (subprocess.CalledProcessError, OSError):
        commit = '미확인'
    row = dict.fromkeys(fields, '')
    values = {'회차': trial, '코드 버전(커밋 번호)': commit, '시도': '11 오른쪽 차선·우회전·START 정지',
              '완주': '사용자 확인 대기', '이탈': '사용자 확인 대기', '시간(초)': elapsed, '손댄 횟수': '사용자 확인 대기',
              '메모': f'{reason}; 기록 {notes or "없음"}; 실제 동작 사용자 확인 대기; 원시 기록 {raw}; 영상 {paths}',
              '날짜': datetime.now(KST).strftime('%Y-%m-%d'), '프로그램': 'missions/11_right_lane_turn.py',
              '속도': 35, '원시기록': str(raw), '종료후전체정지확인': '사용자 확인 대기'}
    row.update({k: v for k, v in values.items() if k in fields})
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='두 차선 길 오른쪽 차선 → 우회전 → START 정지 / --check는 모터 명령 없음')
    parser.add_argument('--check', action='store_true', help='모터 없이 선·흰 실선·흰 비율·단계·바퀴 계산 값을 1초마다 출력')
    parser.add_argument('--trial', default='', help='runs.csv에 남길 회차 이름 (주행 모드)')
    args = parser.parse_args()
    camera = recorder = bot = None
    entered = False
    started = time.monotonic()
    reason, report = '시작 준비 오류', {}
    stamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S_%f')
    path = ROOT / 'results' / f'right_lane_turn_{stamp}.csv'
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
            display.update(first, RightLaneController().report())
            recorder = lane.LaneVideo(display)
            recorder.start()
            with path.open('x', newline='', encoding='utf-8') as stream:
                entered = True
                reason, report = run_loop(camera, recorder, display, key_pressed, stream, args.check, bot)
            if reason == 'FINISH' and bot is not None:
                celebrate(bot)
            print(f'종료: {reason}, 기록: {path}. 실제 정지 위치는 사용자 확인이 필요합니다.', flush=True)
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
                       path, [] if recorder is None else [str(p) for p in recorder.paths], report.get('notes'))
        if errors:
            raise RuntimeError('종료 정리 오류: ' + '; '.join(map(str, errors)))


if __name__ == '__main__':
    main()
