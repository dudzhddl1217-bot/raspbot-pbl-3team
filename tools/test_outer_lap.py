#!/usr/bin/env python3
# 지시서: specs/12-outer-lap.md
"""실제 카메라·I2C·모터 없이 갈림길(한쪽 선)·모서리 세기·빨강/파랑 칸·흰 표시·선 놓침·키 정지·종료 정지·START 재인식을 검증한다."""
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import time
import unittest
import unittest.mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('outer_lap', ROOT / 'missions/12_outer_lap.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
lane = m.lane
YELLOW, WHITE, ROAD = (0, 255, 255), (255, 255, 255), (20, 20, 20)
BLUE, RED = (255, 0, 0), (0, 0, 255)


def limited(target, previous):
    return lane.limit_difference(target, previous)


def frame(yellow=((100, 106), (554, 560)), white=(), fill=(), yellow_line=None):
    img = np.full((480, 640, 3), ROAD, np.uint8)
    for x0, x1, y0, y1, color in fill:
        img[y0:y1, x0:x1] = color
    for a, b in yellow:
        img[:, a:b] = YELLOW
    if yellow_line is not None:
        cv2.line(img, yellow_line[0], yellow_line[1], YELLOW, 6)
    for x0, x1, y0, y1 in white:
        img[y0:y1, x0:x1] = WHITE
    return img


def L(left=None, right=None):
    return dict(left=left, right=right, left_count=0 if left is None else 500, right_count=0 if right is None else 500,
                left_bottom=left, right_bottom=right)


W0 = {'wide': 0.0, 'center': 0.0}
WSTART = {'wide': 0.6, 'center': 0.9}
WRED = {'wide': 0.0, 'center': 0.0, 'red': 0.8, 'blue': 0.0}
WBLUE = {'wide': 0.0, 'center': 0.0, 'red': 0.0, 'blue': 0.8}


class VisionTests(unittest.TestCase):
    def test_two_lines_and_bottom_x(self):
        lines, white = m.analyze(frame())
        self.assertAlmostEqual(lines['left'], 102.5, delta=1)
        self.assertAlmostEqual(lines['right'], 556.5, delta=1)
        self.assertAlmostEqual(lines['left_bottom'], 102.5, delta=1)
        self.assertEqual((white['wide'], white['center'], white['red'], white['blue']), (0.0, 0.0, 0.0, 0.0))

    def test_single_line_named_by_position_or_continuity(self):
        lines, _ = m.analyze(frame(yellow=[(100, 106)]))         # 수직 한 줄, 왼쪽
        self.assertAlmostEqual(lines['left'], 102.5, delta=1)
        self.assertIsNone(lines['right'])
        lines, _ = m.analyze(frame(yellow=[(554, 560)]))         # 수직 한 줄, 오른쪽
        self.assertIsNone(lines['left'])
        self.assertAlmostEqual(lines['right'], 556.5, delta=1)
        lines, _ = m.analyze(frame(yellow=[(420, 426)]), last_left=(400, 0.2))   # 직전 왼쪽 선 근처면 왼쪽
        self.assertAlmostEqual(lines['left'], 422.5, delta=1)
        lines, _ = m.analyze(frame(yellow=[(420, 426)]), last_left=(400, 2.0))   # 오래됐으면 위치로 → 오른쪽
        self.assertIsNone(lines['left'])
        curved, _ = m.analyze(frame(yellow=[], yellow_line=((110, 299), (200, 120))))   # 코너에서 휜 바깥 선 (/ 모양)
        self.assertIsNotNone(curved['left'])

    def test_junction_island_fragment_does_not_steal_left(self):
        f = frame(yellow=[(100, 106)], yellow_line=((600, 200), (639, 160)))   # 왼쪽 선 + 갈림길 섬 모서리 조각
        lines, _ = m.analyze(f)
        self.assertAlmostEqual(lines['left'], 102.5, delta=1)                   # 두 덩어리 x순: 왼쪽은 그대로

    def test_blue_red_zones_and_white_marks_ignored(self):
        zones = frame(fill=[(110, 330, 0, 480, BLUE), (330, 550, 0, 480, RED)])
        lines, white = m.analyze(zones)
        self.assertAlmostEqual(lines['left'], 102.5, delta=1)
        self.assertAlmostEqual(lines['right'], 556.5, delta=1)
        comps = lane.find_components(zones)
        self.assertEqual(len(comps), 2)
        self.assertGreater(white['blue'], 0.4)                    # 아래 띠 넓은 창의 절반쯤 파랑·절반쯤 빨강
        self.assertGreater(white['red'], 0.4)
        self.assertEqual(white['center'], 0.0)
        blue_only = frame(fill=[(110, 550, 0, 480, BLUE)])
        _, white = m.analyze(blue_only)
        self.assertGreaterEqual(white['blue'], m.ZONE_MIN_RATIO)
        self.assertEqual(white['red'], 0.0)
        plain = frame()
        _, white = m.analyze(plain)
        self.assertEqual((white['red'], white['blue']), (0.0, 0.0))
        marks = frame(white=[(120, 540, 150, 170), (120, 540, 190, 210), (300, 340, 120, 300)])   # 횡단보도·화살표
        lines, _ = m.analyze(marks)
        self.assertEqual(len(lane.find_components(marks)), 2)
        self.assertAlmostEqual(lines['left'], 102.5, delta=1)

    def test_start_floor_ratio(self):
        chevrons = frame(white=[(120, 540, 320, 380)])
        _, white = m.analyze(chevrons)
        self.assertGreaterEqual(white['center'], m.START_CENTER)
        self.assertGreaterEqual(white['wide'], m.START_WIDE)
        side = frame(white=[(400, 540, 300, 400)])                # 옆쪽만 흰색(섬) → 바로 앞은 0
        _, white = m.analyze(side)
        self.assertLess(white['center'], m.START_CENTER)
        with self.assertRaises(ValueError):
            m.analyze(np.zeros((10, 10, 3), np.uint8))

    def test_annotated(self):
        f = frame()
        c = m.OuterLapController()
        lines, white = m.analyze(f)
        c.update(lines, white, 0)
        out = m.annotated(f, c.report())
        self.assertEqual(out.shape, f.shape)


class ControllerTests(unittest.TestCase):
    @staticmethod
    def advance(c, lines, white, t_from, t_to, step=0.1):
        t = t_from
        while t + step <= t_to + 1e-9:
            t = round(t + step, 6)
            c.update(lines, white, t)
        return t

    def test_waitlines_grace(self):
        c = m.OuterLapController()
        t = 0.0
        for _ in range(30):
            c.update(L(), W0, t)
            self.assertEqual((c.state, c.speeds, c.stopped), ('WAITLINES', m.STOP, False))
            t = round(t + 0.1, 6)
        c.update(L(), W0, 3.0)
        self.assertEqual(c.state, 'LOSTSTOP')
        self.assertIn('NO_LINES_AT_START', c.notes)

    def test_follow_left_line_with_default_or_measured_width(self):
        c = m.OuterLapController()
        speeds = c.update(L(100, None), W0, 0)
        self.assertEqual((c.state, c.mode), ('LAP', 'TRACKLEFT'))
        self.assertEqual(c.center, 100 + m.DEFAULT_WIDTH / 2)
        self.assertEqual(speeds, limited(lane.wheel_speeds(330), m.STOP))
        c.update(L(100, 560), W0, 0.1)                           # 두 선 → 폭 460 측정
        self.assertTrue(c.width_measured)
        self.assertEqual(c.width, 460)
        self.assertEqual(c.mode, 'TRACKBOTH')
        c.update(L(90, None), W0, 0.2)                           # 갈림길: 오른쪽 선 사라짐 → 왼쪽 선 + 폭/2
        self.assertEqual(c.mode, 'TRACKLEFT')
        self.assertAlmostEqual(c.center, (330 + 330 + 320) / 3, places=5)   # 최근 3개 평균
        c.update(L(None, 500), W0, 0.3)                          # 오른쪽 선만 → 오른쪽 − 폭/2
        self.assertEqual(c.mode, 'TRACKRIGHT')
        c.update(L(100, 800), W0, 0.4)                           # 폭 700은 범위 밖 → 저장 안 함
        self.assertEqual(c.width, 460)

    def test_lost_hold_then_stop_without_turn_history(self):
        c = m.OuterLapController()
        c.update(L(100, 540), W0, 0)                             # 중심 320 → 직진, 꺾은 적 없음
        held = c.speeds
        c.update(L(), W0, 0.4)
        self.assertEqual((c.mode, c.speeds), ('HOLD', held))
        c.update(L(), W0, 0.6)
        self.assertEqual(c.state, 'LOSTSTOP')

    def test_lost_search_in_last_turn_direction_then_stop(self):
        c = m.OuterLapController()
        c.update(L(200, None), W0, 0)                            # 중심 430 → 오른쪽 꺾기
        self.assertEqual(c.last_turn, 'RIGHT')
        prev = c.speeds
        c.update(L(), W0, 0.1)
        self.assertEqual(c.mode, 'SEARCHRIGHT')
        self.assertEqual(c.speeds, limited(m.SEARCH_RIGHT, prev))
        self.advance(c, L(), W0, 0.1, 1.0)
        self.assertEqual(c.state, 'LAP')
        c.update(L(), W0, 1.1)
        self.assertEqual(c.state, 'LOSTSTOP')
        c2 = m.OuterLapController()
        c2.update(L(None, 440), W0, 0)                           # 중심 210 → 왼쪽 꺾기
        c2.update(L(), W0, 0.1)
        self.assertEqual(c2.mode, 'SEARCHLEFT')

    def test_speed_difference_change_limited(self):
        c = m.OuterLapController()
        prev, t = m.STOP, 0.0
        for lines in [L(100, 560)] * 3 + [L(300, None)] * 6 + [L()] * 3 + [L(100, 560)] * 3:
            speeds = c.update(lines, W0, t)
            self.assertLessEqual(abs((speeds[0] - speeds[2]) - (prev[0] - prev[2])), 10)
            self.assertTrue(all(0 <= v <= 80 for v in speeds))
            prev, t = speeds, round(t + 0.1, 6)

    def test_corner_counting_window_and_cooldown(self):
        c = m.OuterLapController()
        t = 0.0
        c.update(L(100, 560), W0, t)
        while c.corners == 0 and t < 10:                         # 왼쪽 선이 300 → 강한 우회전 명령 누적
            t = round(t + 0.1, 6)
            c.update(L(300, None), W0, t)
        self.assertEqual(c.corners, 1)
        self.assertLess(t, 3.0)
        first = t
        self.assertEqual(c.turn_sum, 0.0)                        # 센 뒤 창 비움
        t = self.advance(c, L(300, None), W0, t, first + m.CORNER_COOLDOWN - 0.2)
        self.assertEqual(c.corners, 1)                           # 쉬는 시간 동안 다시 세지 않음
        t = self.advance(c, L(300, None), W0, t, first + m.CORNER_COOLDOWN + 2.0)
        self.assertEqual(c.corners, 2)
        c2 = m.OuterLapController()                              # 직선 잔 흔들림(좌우 교대)은 합이 0 근처
        t = 0.0
        for i in range(60):
            c2.update(L(90 if i % 2 else 110, None), W0, t)
            t = round(t + 0.1, 6)
        self.assertEqual(c2.corners, 0)
        self.assertLess(abs(c2.turn_sum), m.CORNER_TURN_SUM)

    def test_armed_after_four_corners_then_finish_or_timeout(self):
        with unittest.mock.patch.object(m, 'CORNER_TURN_SUM', 5.0), unittest.mock.patch.object(m, 'CORNER_COOLDOWN', 0.3):
            c = m.OuterLapController()
            t = 0.0
            c.update(L(100, 560), W0, t)
            while c.state != 'ARMED' and t < 20:
                t = round(t + 0.1, 6)
                c.update(L(300, None), W0, t)
            self.assertEqual(c.state, 'ARMED')
            self.assertEqual(c.corners, 4)
            c.update(L(100, 560), {'wide': 0.1, 'center': 0.9}, t + 0.1)   # 바로 앞만 희면 아직 아님
            c.update(L(100, 560), {'wide': 0.3, 'center': 0.2}, t + 0.2)   # 넓게만 희면 아직 아님
            self.assertEqual(c.state, 'ARMED')
            c.update(L(100, 560), WSTART, t + 0.3)
            self.assertEqual(c.state, 'ARMED')
            c.update(L(100, 560), WSTART, t + 0.4)                           # 연속 2회 → FINISH
            self.assertEqual(c.state, 'FINISH')
            self.assertTrue(c.stopped)
            self.assertEqual(c.update(L(100, 560), WSTART, t + 0.5), m.STOP)
            c2 = m.OuterLapController()
            t = 0.0
            c2.update(L(100, 560), W0, t)
            while c2.state != 'ARMED' and t < 20:
                t = round(t + 0.1, 6)
                c2.update(L(300, None), W0, t)
            base = c2.phase_since
            self.advance(c2, L(100, 560), W0, t, base + m.FINISH_MAX - 0.1)
            self.assertEqual(c2.state, 'ARMED')
            c2.update(L(100, 560), W0, base + m.FINISH_MAX + 0.05)
            self.assertEqual(c2.state, 'TIMEOUT')
            self.assertIn('FINISH_TIMEOUT', c2.notes)

    def test_zone_factor_slow_fast_and_hold(self):
        c = m.OuterLapController()
        c.update(L(100, 540), W0, 0)                             # 중심 320 → 35/35
        self.assertEqual(c.speeds, (35, 35, 35, 35))
        c.update(L(100, 540), WRED, 0.1)                         # 빨강 칸 → 감속
        self.assertEqual((c.zone, c.factor), ('RED', m.SLOW_FACTOR))
        self.assertEqual(c.speeds, limited((28, 28, 28, 28), (35, 35, 35, 35)))
        c.update(L(100, 540), W0, 0.4)                           # 0.5초 안 → 유지
        self.assertEqual(c.zone, 'RED')
        c.update(L(100, 540), W0, 0.7)                           # 0.5초 넘음 → 해제
        self.assertEqual((c.zone, c.factor), (None, 1.0))
        c.update(L(100, 540), WBLUE, 0.8)                        # 파랑 칸 → 가속
        self.assertEqual((c.zone, c.factor), ('BLUE', m.FAST_FACTOR))
        self.advance(c, L(100, 540), WBLUE, 0.8, 1.5)
        self.assertEqual(c.speeds, (46, 46, 46, 46))             # 35×1.3 = 45.5 → 46
        c.update(L(200, None), WBLUE, 1.6)                       # 가속 중에도 최대 80·속도차 제한 유지
        self.assertTrue(all(0 <= v <= 80 for v in c.speeds))
        self.assertLessEqual(abs((c.speeds[0] - c.speeds[2]) - 0), 10)
        r = c.report()
        self.assertEqual((r['zone'], r['factor']), ('BLUE', m.FAST_FACTOR))

    def test_start_marks_before_arming_are_ignored(self):
        c = m.OuterLapController()
        for i in range(10):                                      # 출발 직후 꺾쇠·횡단보도가 보여도 무시
            c.update(L(100, 560), WSTART, i * 0.1)
        self.assertEqual(c.state, 'LAP')
        self.assertFalse(c.stopped)

    def test_frame_stale_stops(self):
        c = m.OuterLapController()
        c.update(L(100, 560), W0, 0)
        self.assertEqual(c.tick(0.5), c.speeds)
        self.assertEqual(c.tick(0.51), m.STOP)
        self.assertEqual(c.state, 'FRAMESTOP')

    def test_report_fields(self):
        c = m.OuterLapController()
        c.update(L(100, 560), W0, 0)
        r = c.report()
        for key in ('state', 'mode', 'speeds', 'width', 'center', 'corners', 'turn_sum', 'white', 'white_center', 'notes'):
            self.assertIn(key, r)


class FakeCamera:
    def __init__(self, scenes):
        self.scenes, self.start, self.sequence = scenes, time.monotonic(), 0

    def snapshot(self):
        elapsed = time.monotonic() - self.start
        img = self.scenes[-1][1]
        for until, candidate in self.scenes:
            if elapsed < until:
                img = candidate
                break
        self.sequence += 1
        return img.copy(), self.sequence


class FakeBot:
    def __init__(self):
        self.calls = []

    def Ctrl_Muto(self, number, speed):
        self.calls.append(('motor', number, speed))

    def Ctrl_BEEP_Switch(self, state):
        self.calls.append(('beep', state))

    def Ctrl_WQ2812_ALL(self, state, color):
        self.calls.append(('light', state, color))


class LoopTests(unittest.TestCase):
    def test_check_mode_sends_no_motor_commands_and_keeps_observing(self):
        camera = FakeCamera([(0.4, frame()), (99, frame(yellow=[]))])    # 선 보이다가 사라짐
        presses = iter([False] * 120 + [True] * 100)
        stream = io.StringIO()
        with unittest.mock.patch.object(m, 'START_GRACE', 0.1), unittest.mock.patch.object(m, 'SEARCH_MAX', 0.2):
            reason, report = m.run_loop(camera, None, m.Display(), lambda: next(presses), stream, check=True, bot=None)
        self.assertEqual(reason, '키 입력')
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(rows[0]['실행모드'], '읽기전용')
        states = [r['단계'] for r in rows]
        self.assertIn('LAP', states)
        self.assertIn('LOSTSTOP', states)                               # 정지 조건이 와도 계속 기록
        self.assertGreaterEqual(states.count('WAITLINES'), 1)

    def test_drive_mode_corners_then_finish_at_start_marks(self):
        both = frame()
        turning = frame(yellow=[(300, 306)])                              # 왼쪽 선이 가운데 쪽 → 강한 우회전
        chevrons = frame(white=[(120, 540, 320, 380)])
        camera = FakeCamera([(0.5, both), (3.2, turning), (99, chevrons)])
        bot = FakeBot()
        stream = io.StringIO()
        with unittest.mock.patch.object(m, 'CORNER_TURN_SUM', 4.0), unittest.mock.patch.object(m, 'CORNER_COOLDOWN', 0.5):
            reason, report = m.run_loop(camera, None, m.Display(), lambda: False, stream, check=False, bot=bot)
        self.assertEqual(reason, 'FINISH', report)
        self.assertGreaterEqual(report['corners'], 4)
        states = [r['단계'] for r in csv.DictReader(io.StringIO(stream.getvalue()))]
        for state in ('LAP', 'ARMED', 'FINISH'):
            self.assertIn(state, states)
        motor = [c for c in bot.calls if c[0] == 'motor']
        self.assertEqual(motor[-12:], [('motor', n, 0) for n in range(4)] * 3)   # 종료 정지 3회
        self.assertTrue(all(0 <= c[2] <= 80 for c in motor))

    def test_key_press_stops_with_three_zero_bundles(self):
        camera = FakeCamera([(99, frame())])
        bot = FakeBot()
        presses = iter([False] * 10 + [True] * 100)
        stream = io.StringIO()
        reason, _ = m.run_loop(camera, None, m.Display(), lambda: next(presses), stream, check=False, bot=bot)
        self.assertEqual(reason, '키 입력')
        motor = [c for c in bot.calls if c[0] == 'motor']
        self.assertEqual(motor[-12:], [('motor', n, 0) for n in range(4)] * 3)

    def test_zone_lights_only_on_change(self):
        camera = FakeCamera([(0.3, frame()), (0.8, frame(fill=[(110, 550, 300, 480, BLUE)])), (99, frame())])
        bot = FakeBot()
        presses = iter([False] * 200 + [True] * 100)
        stream = io.StringIO()
        with unittest.mock.patch.object(m, 'ZONE_HOLD', 0.1):
            reason, _ = m.run_loop(camera, None, m.Display(), lambda: next(presses), stream, check=False, bot=bot)
        lights = [c for c in bot.calls if c[0] == 'light']
        self.assertEqual(lights, [('light', 1, 2), ('light', 0, 0)])   # 파랑 켜기 한 번, 끄기 한 번
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertTrue(any(r['구간'] == 'BLUE' and float(r['배율']) == m.FAST_FACTOR for r in rows))

    def test_beep_and_append_run(self):
        bot = FakeBot()
        m.beep(bot, 2)
        self.assertEqual([c for c in bot.calls if c[0] == 'beep'], [('beep', 1), ('beep', 0)] * 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runs.csv'
            m.append_run(path, 't1', 'FINISH', 98.7, 'results/x.csv', [], {'corners': 4, 'notes': []})
            rows = list(csv.DictReader(path.open(encoding='utf-8')))
            self.assertEqual(rows[0]['회차'], 't1')
            self.assertIn('12 바깥 고리', rows[0]['시도'])
            self.assertIn('모서리 4회', rows[0]['메모'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
