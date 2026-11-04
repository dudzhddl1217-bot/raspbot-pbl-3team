#!/usr/bin/env python3
# 지시서: specs/11-right-lane-turn.md
"""실제 카메라·I2C·모터 없이 흰 실선 검출·바닥 흰 비율·단계 전환·주행 반복·정지를 검증한다."""
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
spec = importlib.util.spec_from_file_location('right_lane_turn', ROOT / 'missions/11_right_lane_turn.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
lane = m.lane
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
ROAD = (20, 20, 20)


def limited(target, previous):
    """07의 속도 차이 변화 10 제한을 거친 기대값."""
    return lane.limit_difference(target, previous)


def frame(yellow=((100, 106), (554, 560)), white=(), slanted=None, yellow_line=None):
    img = np.full((480, 640, 3), ROAD, np.uint8)
    for a, b in yellow:
        img[:, a:b] = YELLOW
    if yellow_line is not None:
        cv2.line(img, yellow_line[0], yellow_line[1], YELLOW, 6)
    for x0, x1, y0, y1 in white:
        img[y0:y1, x0:x1] = WHITE
    if slanted is not None:
        cv2.line(img, slanted[0], slanted[1], WHITE, 5)
    return img


def L(left=None, right=None):
    return dict(left=left, right=right, left_count=0 if left is None else 500, right_count=0 if right is None else 500)


class VisionTests(unittest.TestCase):
    def test_divider_between_yellow_lines(self):
        f = frame(white=[(330, 336, 100, 320)])
        lines = lane.find_lines(f)
        self.assertAlmostEqual(lines['left'], 102.5, delta=1)
        self.assertAlmostEqual(lines['right'], 556.5, delta=1)
        d = m.find_divider(f, lines)
        self.assertAlmostEqual(d['divider'], 332.5, delta=1)
        self.assertEqual(d['divider_rows'], 1.0)

    def test_slanted_divider_detected(self):
        f = frame(slanted=((300, 299), (360, 120)))
        d = m.find_divider(f, lane.find_lines(f))
        self.assertIsNotNone(d['divider'])
        self.assertTrue(300 <= d['divider'] <= 360)
        self.assertGreaterEqual(d['divider_rows'], m.DIVIDER_MIN_ROWS)

    def test_arrow_blobs_rejected(self):
        wide = frame(white=[(380, 440, 200, 240)])          # 넓은 덩어리
        self.assertIsNone(m.find_divider(wide, lane.find_lines(wide))['divider'])
        short = frame(white=[(400, 410, 180, 260)])         # 가늘지만 짧음 (행 비율 0.44)
        d = m.find_divider(short, lane.find_lines(short))
        self.assertIsNone(d['divider'])
        self.assertAlmostEqual(d['divider_rows'], 80 / 180, places=2)

    def test_divider_preferred_over_arrow_on_its_right(self):
        f = frame(white=[(330, 336, 100, 320), (400, 410, 120, 300)])
        d = m.find_divider(f, lane.find_lines(f))
        self.assertAlmostEqual(d['divider'], 332.5, delta=1)

    def test_white_island_and_start_marks_rejected(self):
        island = frame(yellow=[(554, 560)], white=[(0, 90, 0, 480)])   # 왼쪽 노란 선 없음 + 흰 섬
        self.assertIsNone(m.find_divider(island, lane.find_lines(island))['divider'])
        island2 = frame(white=[(0, 90, 0, 480)])                        # 왼쪽 노란 선 있음 + 흰 섬
        self.assertIsNone(m.find_divider(island2, lane.find_lines(island2))['divider'])
        chevrons = frame(white=[(120, 540, 150, 280)])
        self.assertIsNone(m.find_divider(chevrons, lane.find_lines(chevrons))['divider'])

    def test_yellow_fringe_is_not_white(self):
        region = np.full((40, 640, 3), ROAD, np.uint8)
        region[:, 330:336] = YELLOW
        region[:, 336:340] = (230, 255, 255)                 # 노란 선 가장자리 번짐 (채도 낮고 밝음)
        region[:, 360:364] = WHITE
        mask = m._masks(region)
        self.assertFalse(mask[:, 336:340].any())
        self.assertTrue(mask[:, 360:364].all())
        plain = frame()
        self.assertIsNone(m.find_divider(plain, lane.find_lines(plain))['divider'])

    def test_floor_white_ratio(self):
        plain = frame()
        self.assertEqual(m.floor_white_ratio(plain, lane.find_lines(plain)), 0.0)
        chevrons = frame(white=[(120, 540, 320, 380)])
        ratio = m.floor_white_ratio(chevrons, lane.find_lines(chevrons))
        self.assertGreaterEqual(ratio, m.START_WHITE)
        self.assertAlmostEqual(ratio, 0.6, delta=0.02)
        no_lines = frame(yellow=[], white=[(160, 480, 300, 400)])
        self.assertAlmostEqual(m.floor_white_ratio(no_lines, lane.find_lines(no_lines)), 1.0, delta=0.01)
        with self.assertRaises(ValueError):
            m.find_divider(np.zeros((100, 100, 3), np.uint8), L())

    def test_analyze_and_annotated(self):
        f = frame(white=[(330, 336, 100, 320)])
        lines, divider, white = m.analyze(f)
        self.assertIsNotNone(divider['divider'])
        c = m.RightLaneController()
        c.update(lines, divider['divider'], white, 0)
        out = m.annotated(f, c.report())
        self.assertEqual(out.shape, f.shape)
        self.assertFalse(np.array_equal(out, f))


class ControllerTests(unittest.TestCase):
    """시각은 실제처럼 0.1초 간격으로 흘린다 (0.5초 넘게 화면이 없으면 07 규칙대로 FRAMESTOP).
    바퀴 기대값은 숫자를 박지 않고 07의 wheel_speeds·limit_difference로 계산한다."""

    @staticmethod
    def advance(c, lines, divider, white, t_from, t_to, step=0.1):
        t = t_from
        while t + step <= t_to + 1e-9:
            t = round(t + step, 6)
            c.update(lines, divider, white, t)
        return t

    def enter_rightlane(self, c, t=0.0):
        c.update(L(100, 560), None, 0.0, t)
        c.update(L(100, 560), 330, 0.0, t + 0.1)
        c.update(L(100, 560), 332, 0.0, t + 0.2)
        self.assertEqual(c.state, 'RIGHTLANE')
        return round(t + 0.2, 6)

    def enter_turnright(self, c):
        t = self.enter_rightlane(c)
        t = self.advance(c, L(None, 560), None, 0.0, t, t + 1.0)     # 흰 실선 끝 + 1초 경과
        self.assertEqual(c.state, 'TURNRIGHT')
        return t

    def enter_finishrun(self, c):
        t = self.enter_turnright(c)
        c.update(L(100, 560), None, 0.0, t + 0.1)
        c.update(L(100, 560), None, 0.0, t + 0.2)
        self.assertEqual(c.state, 'FINISHRUN')
        return round(t + 0.2, 6)

    def test_waitlane_follows_07_then_switches(self):
        c = m.RightLaneController()
        self.assertEqual(c.update(L(100, 560), None, 0.0, 0), limited(lane.wheel_speeds(330), m.STOP))
        self.assertEqual(c.state, 'WAITLANE')
        c.update(L(100, 560), 330, 0.0, 0.1)
        self.assertEqual(c.state, 'WAITLANE')                 # 연속 1회는 아직
        prev = c.speeds
        speeds = c.update(L(100, 560), 332, 0.0, 0.2)
        self.assertEqual(c.state, 'RIGHTLANE')
        self.assertEqual(c.right_width, 228)
        self.assertEqual(c.full_width, 460)
        self.assertEqual(c.center, 446)
        self.assertEqual(speeds, limited(lane.wheel_speeds(446), prev))
        self.assertLessEqual(abs((speeds[0] - speeds[2]) - (prev[0] - prev[2])), 10)
        self.assertEqual(c.previous, dict(left=100, right=560))

    def test_speed_difference_change_limited_every_frame(self):
        c = m.RightLaneController()
        t = self.enter_rightlane(c)
        t = self.advance(c, L(None, 560), None, 0.0, t, t + 1.0)
        self.assertEqual(c.state, 'TURNRIGHT')
        prev = c.speeds
        for lines in (L(None, None), L(None, None), L(None, 500), L(None, None)):
            t = round(t + 0.1, 6)
            speeds = c.update(lines, None, 0.0, t)
            self.assertLessEqual(abs((speeds[0] - speeds[2]) - (prev[0] - prev[2])), 10)
            prev = speeds
        self.assertEqual(c.single_name, 'right')

    def test_divider_needs_consecutive_new_results(self):
        c = m.RightLaneController()
        c.update(L(100, 560), 330, 0.0, 0)
        c.update(L(100, 560), None, 0.0, 0.1)
        c.update(L(100, 560), 331, 0.0, 0.2)
        self.assertEqual(c.state, 'WAITLANE')

    def test_waitlane_width_from_right_line_or_stop_without_yellow(self):
        c = m.RightLaneController()
        c.update(L(None, 560), 330, 0.0, 0)
        c.update(L(None, 560), 331, 0.0, 0.1)                # 오른쪽 선과 흰 실선으로 폭을 안다
        self.assertEqual(c.state, 'RIGHTLANE')
        self.assertEqual(c.right_width, 229)
        c2 = m.RightLaneController()
        c2.update(L(None, None), 330, 0.0, 0)                # 노란 선이 하나도 없으면(흰 실선만) 선 대기
        self.assertEqual(c2.state, 'WAITLINES')
        self.assertFalse(c2.stopped)

    def test_rightlane_center_from_one_line(self):
        c = m.RightLaneController()
        t = self.enter_rightlane(c)
        c.update(L(None, 560), None, 0.0, t + 0.1)
        self.assertEqual(c.center, 446)
        c.update(L(None, None), 332, 0.0, t + 0.2)
        self.assertEqual(c.center, 446)
        held = c.speeds
        c.update(L(None, None), None, 0.0, t + 0.3)         # 잠깐 놓침: 직전 명령 유지
        self.assertEqual(c.speeds, held)
        self.assertEqual(c.state, 'RIGHTLANE')

    def test_divider_gone_switches_after_min_time(self):
        c = m.RightLaneController()
        t = self.enter_rightlane(c)                          # t = 0.2, 마지막 흰 실선 0.2
        for dt in (0.2, 0.4, 0.6, 0.8, 0.95):
            prev = c.speeds
            c.update(L(None, 560), None, 0.0, t + dt)
            self.assertEqual(c.state, 'RIGHTLANE', dt)       # 1초 전에는 오른쪽 선으로 차선 유지
            self.assertEqual(c.speeds, limited(lane.wheel_speeds(446), prev))
        prev = c.speeds
        c.update(L(None, 560), None, 0.0, t + 1.0)
        self.assertEqual(c.state, 'TURNRIGHT')
        self.assertEqual(c.speeds, limited(lane.wheel_speeds(560 - 114), prev))
        c3 = m.RightLaneController()                         # 흰 실선이 계속 보이면 넘어가지 않는다
        t = self.enter_rightlane(c3)
        self.advance(c3, L(None, 560), 332, 0.0, t, t + 3.0)
        self.assertEqual(c3.state, 'RIGHTLANE')
        self.assertEqual(c3.speeds, lane.wheel_speeds(446))  # 3초면 제한이 풀려 목표값에 도달

    def test_turnright_arc_and_exit_on_wide_lines(self):
        c = m.RightLaneController()
        t = self.enter_turnright(c)
        prev = c.speeds
        self.assertEqual(c.update(L(None, None), None, 0.0, t + 0.1), limited(m.TURN_ARC, prev))
        prev = c.speeds
        self.assertEqual(c.update(L(None, 500), None, 0.0, t + 0.2), limited(lane.wheel_speeds(386), prev))
        c.update(L(100, 560), None, 0.0, t + 0.3)            # 큰 길 1회
        self.assertEqual(c.state, 'TURNRIGHT')
        before = c.speeds
        c.update(L(100, 560), None, 0.0, t + 0.4)            # 2회 → FINISHRUN
        self.assertEqual(c.state, 'FINISHRUN')
        self.assertEqual(c.lane.width, 460)
        self.assertLessEqual(abs((c.speeds[0] - c.speeds[2]) - (before[0] - before[2])), 10)   # 전환에도 제한 유지
        c.update(L(None, 560), None, 0.0, t + 0.5)           # 한쪽 선만 보여도 폭을 알아 바로 간다
        self.assertEqual(c.lane.state, 'TRACKRIGHT')
        self.assertEqual(c.notes, [])

    def test_turnright_timeout_and_lost(self):
        c = m.RightLaneController()
        t = self.enter_turnright(c)
        self.advance(c, L(None, 500), None, 0.0, t, t + 3.9)
        self.assertEqual(c.state, 'TURNRIGHT')
        c.update(L(None, 500), None, 0.0, t + 4.0)
        self.assertEqual(c.state, 'FINISHRUN')
        self.assertIn('TURN_TIMEOUT', c.notes)
        c2 = m.RightLaneController()
        t = self.enter_turnright(c2)                         # 마지막으로 선을 본 시각 = t
        self.advance(c2, L(None, None), None, 0.0, t, t + 2.0)
        self.assertEqual(c2.state, 'TURNRIGHT')
        self.assertEqual(c2.speeds, m.TURN_ARC)              # 2초면 제한이 풀려 호 그리기 값에 도달
        c2.update(L(None, None), None, 0.0, t + 2.1)
        self.assertEqual(c2.state, 'LOSTSTOP')
        self.assertTrue(c2.stopped)
        self.assertEqual(c2.update(L(100, 560), 330, 0.0, t + 2.2), m.STOP)

    def test_finishrun_stops_on_start_or_timeout(self):
        c = m.RightLaneController()
        t = self.enter_finishrun(c)
        prev = c.speeds
        self.assertEqual(c.update(L(100, 560), None, 0.3, t + 0.1), limited(lane.wheel_speeds(330), prev))
        prev = c.speeds
        self.assertEqual(c.update(L(100, 560), None, 0.1, t + 0.2), limited(lane.wheel_speeds(330), prev))   # 연속 끊김
        c.update(L(100, 560), None, 0.3, t + 0.3)
        self.assertEqual(c.update(L(100, 560), None, 0.3, t + 0.4), m.STOP)
        self.assertEqual(c.state, 'FINISH')
        self.assertTrue(c.stopped)
        self.assertEqual(c.update(L(100, 560), None, 0.0, t + 0.5), m.STOP)
        c2 = m.RightLaneController()
        t = self.enter_finishrun(c2)
        self.advance(c2, L(100, 560), None, 0.0, t, t + 7.9)
        self.assertEqual(c2.state, 'FINISHRUN')
        c2.update(L(100, 560), None, 0.0, t + 8.0)
        self.assertEqual(c2.state, 'TIMEOUT')
        self.assertIn('FINISH_TIMEOUT', c2.notes)

    def test_frame_stale_and_lane_lost_stop(self):
        c = m.RightLaneController()
        c.update(L(100, 560), None, 0.0, 0)
        self.assertEqual(c.tick(0.5), limited(lane.wheel_speeds(330), m.STOP))
        self.assertEqual(c.tick(0.51), m.STOP)
        self.assertEqual(c.state, 'FRAMESTOP')
        c2 = m.RightLaneController()
        t = 0.0
        for _ in range(29):                                  # 2.9초 동안 선 없음 → 정지 대기 유지
            c2.update(L(None, None), None, 0.0, t)
            self.assertEqual((c2.state, c2.speeds, c2.stopped), ('WAITLINES', m.STOP, False))
            t = round(t + 0.1, 6)
        c2.update(L(None, None), None, 0.0, 3.0)              # 3초 → LOSTSTOP
        self.assertEqual(c2.state, 'LOSTSTOP')
        self.assertTrue(c2.stopped)
        self.assertIn('NO_LINES_AT_START', c2.notes)
        c3 = m.RightLaneController()
        c3.update(L(None, None), None, 0.0, 0)
        c3.update(dict(L(None, None), unassigned=True), None, 0.0, 0.1)   # 이름 없는 노란 덩어리도 「선을 봄」
        self.assertEqual(c3.state, 'WAITLANE')
        self.assertEqual(c3.lane.state, 'WAITWIDTH')
        c4 = m.RightLaneController()
        c4.update(L(None, None), None, 0.0, 0)
        c4.update(L(100, 560), None, 0.0, 0.1)                # 선이 보이면 바로 07 주행
        self.assertEqual(c4.state, 'WAITLANE')
        self.assertEqual(c4.speeds, limited(lane.wheel_speeds(330), m.STOP))
        c5 = m.RightLaneController()                          # 선을 본 뒤(직진, 꺾은 적 없음) 놓치면 07 규칙대로 즉시 정지
        c5.update(L(100, 540), None, 0.0, 0)
        c5.update(L(None, None), None, 0.0, 0.1)
        self.assertEqual(c5.state, 'LOSTSTOP')

    def test_report_fields(self):
        c = m.RightLaneController()
        self.enter_rightlane(c)
        r = c.report()
        for key in ('state', 'speeds', 'divider', 'white', 'right_width', 'full_width', 'center', 'width', 'notes'):
            self.assertIn(key, r)
        self.assertEqual(r['width'], 228)


class FakeCamera:
    """경과 시간에 따라 장면을 바꿔 주는 가짜 카메라."""
    def __init__(self, scenes):
        self.scenes = scenes
        self.start = time.monotonic()
        self.sequence = 0

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
    def test_check_mode_sends_no_motor_commands(self):
        scenes = [(99, frame(white=[(330, 336, 100, 320)]))]
        camera = FakeCamera(scenes)
        presses = iter([False] * 12 + [True] * 100)
        stream = io.StringIO()
        reason, report = m.run_loop(camera, None, m.Display(), lambda: next(presses), stream, check=True, bot=None)
        self.assertEqual(reason, '키 입력')
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]['모드'], '읽기전용')
        self.assertEqual(rows[-1]['단계'], 'RIGHTLANE')
        self.assertAlmostEqual(float(rows[-1]['흰실선_x']), 332.5, delta=1)

    def test_check_mode_keeps_observing_after_stop_condition(self):
        dark = frame(yellow=[])                                # 선이 하나도 없는 화면
        camera = FakeCamera([(99, dark)])
        presses = iter([False] * 40 + [True] * 100)
        stream = io.StringIO()
        with unittest.mock.patch.object(m, 'START_GRACE', 0.05):
            reason, report = m.run_loop(camera, None, m.Display(), lambda: next(presses), stream, check=True, bot=None)
        self.assertEqual(reason, '키 입력')                    # LOSTSTOP으로 끝나지 않고 키까지 계속 돈다
        states = [r['단계'] for r in csv.DictReader(io.StringIO(stream.getvalue()))]
        self.assertIn('LOSTSTOP', states)
        self.assertGreaterEqual(states.count('WAITLINES'), 2)   # 정지 뒤 새 판단으로 다시 시작

    def test_drive_mode_stops_when_no_lines_at_start(self):
        camera = FakeCamera([(99, frame(yellow=[]))])
        bot = FakeBot()
        stream = io.StringIO()
        with unittest.mock.patch.object(m, 'START_GRACE', 0.2):
            reason, report = m.run_loop(camera, None, m.Display(), lambda: False, stream, check=False, bot=bot)
        self.assertEqual(reason, 'LOSTSTOP')
        self.assertIn('NO_LINES_AT_START', report['notes'])
        motor = [c for c in bot.calls if c[0] == 'motor']
        self.assertTrue(all(c[2] == 0 for c in motor))         # 선을 못 본 동안 바퀴 명령은 0뿐

    def test_drive_mode_reaches_finish_and_stops(self):
        two_lane = frame(white=[(330, 336, 100, 320)])
        right_only = frame(yellow=[], yellow_line=((540, 120), (575, 299)))   # 기울어진 오른쪽 선 (07 기울기 규칙)
        big_road = frame()
        chevrons = frame(white=[(120, 540, 320, 380)])
        camera = FakeCamera([(1.3, two_lane), (2.0, right_only), (2.4, big_road), (99, chevrons)])
        bot = FakeBot()
        stream = io.StringIO()
        reason, report = m.run_loop(camera, None, m.Display(), lambda: False, stream, check=False, bot=bot)
        self.assertEqual(reason, 'FINISH')
        self.assertTrue(report['stopped'])
        states = [r['단계'] for r in csv.DictReader(io.StringIO(stream.getvalue()))]
        for state in ('WAITLANE', 'RIGHTLANE', 'TURNRIGHT', 'FINISHRUN', 'FINISH'):
            self.assertIn(state, states)
        motor = [c for c in bot.calls if c[0] == 'motor']
        self.assertEqual(motor[-12:], [('motor', n, 0) for n in range(4)] * 3)   # 종료 정지 3회
        self.assertTrue(any(c[2] > 0 for c in motor))
        self.assertTrue(all(abs(c[2]) <= 80 for c in motor))
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertTrue(any(r['단계'] == 'RIGHTLANE' and r['오른쪽선_x'] for r in rows))   # 기울어진 한쪽 선도 오른쪽으로 이름 붙음
        m.celebrate(bot)
        self.assertEqual([c for c in bot.calls if c[0] == 'light'], [('light', 1, 1)])
        self.assertEqual([c for c in bot.calls if c[0] == 'beep'], [('beep', 1), ('beep', 0)] * 2)

    def test_append_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runs.csv'
            m.append_run(path, 't1', 'FINISH', 12.3, 'results/x.csv', [], ['TURN_TIMEOUT'])
            rows = list(csv.DictReader(path.open(encoding='utf-8')))
            self.assertEqual(rows[0]['회차'], 't1')
            self.assertIn('11 오른쪽 차선', rows[0]['시도'])
            self.assertIn('TURN_TIMEOUT', rows[0]['메모'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
