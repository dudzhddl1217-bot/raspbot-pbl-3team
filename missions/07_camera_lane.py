#!/usr/bin/env python3
# 지시서: specs/07-camera-lane.md
"""노란 차선 비례 조향. --check는 드라이버를 불러오거나 모터 명령을 보내지 않는다."""
import argparse
from collections import deque
from contextlib import contextmanager
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import signal
import sys
import threading
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


line = load('camera_lane_stop', '02_line.py')
video = load('camera_lane_blackbox', '03_video.py')
STOP = (0, 0, 0, 0)
KST = ZoneInfo('Asia/Seoul')
ROI_TOP, ROI_BOTTOM = 120, 300
BASE_SPEED = 35
STEERING_K = 0.3
MAX_SPEED = 80
SEARCH_SECONDS = 1.0


class Component(tuple):
    """기존 (평균x, 점수) 표현에 실제 연결 점들을 보관한다."""
    def __new__(cls, xs, ys):
        obj = super().__new__(cls, (float(xs.mean()), len(xs)))
        obj.xs, obj.ys = xs, ys
        return obj


def find_components(frame):
    if frame is None or frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise ValueError('차선 입력 화면은 640x480 uint8 BGR이어야 합니다.')
    hsv = cv2.cvtColor(frame[ROI_TOP:ROI_BOTTOM], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 80, 100), (40, 255, 255))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    selected = sorted((i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 50),
                      key=lambda i: (-stats[i, cv2.CC_STAT_AREA], centers[i, 0]))[:2]
    parts = []
    for i in selected:
        ys, xs = np.nonzero(labels == i)
        parts.append(Component(xs, ys))
    return parts


def assign_lines(components, previous=None, width=None, last_name=None):
    result = dict(left=None, right=None, left_count=0, right_count=0)
    parts = sorted(components, key=lambda c: c[0])
    if len(parts) == 2 and parts[1][0] - parts[0][0] < 300:
        if all(isinstance(c, Component) for c in parts):
            parts = [Component(np.concatenate([c.xs for c in parts]),
                               np.concatenate([c.ys for c in parts]))]
        else:
            total = sum(c[1] for c in parts)
            parts = [(sum(x*n for x,n in parts)/total, total)]
    if not parts:
        return result
    if len(parts) == 2:
        assigned = zip(('left', 'right'), parts)
    elif isinstance(parts[0], Component):
        part = parts[0]
        xs, ys = part.xs, part.ys
        name = last_name
        if np.ptp(xs)+1 < 3*(np.ptp(ys)+1):
            dx, dy = xs-xs.mean(), ys-ys.mean()
            slope = float(np.dot(dx, dy) / np.dot(dy, dy)) if np.dot(dy, dy) else 0.
            if slope < 0:
                name = 'left'
            elif slope > 0:
                name = 'right'
        if name is None:
            return dict(result, unassigned=True)
        assigned = [(name, (float(xs[ys == ys.max()].mean()), part[1]))]
    elif previous is None:
        # 원본 픽셀이 없는 저장 좌표를 쓰는 오프라인 검증 호환 경로.
        return dict(result, unassigned=True)
    else:
        x, _ = parts[0]
        name = min(('left', 'right'), key=lambda name: abs(x-previous[name]))
        assigned = [(name, parts[0])]
    for name, (x, area) in assigned:
        result[name], result[name+'_count'] = x, area
    return result


def find_lines(frame, previous=None, width=None, last_name=None):
    return assign_lines(find_components(frame), previous, width, last_name)


def limit_difference(target, previous):
    old = previous[0] - previous[2]
    difference = max(old-10, min(old+10, target[0]-target[2]))
    left = max(max(0, difference), min(min(MAX_SPEED, MAX_SPEED+difference),
               round((target[0]+target[2]+difference)/2)))
    right = left-difference
    return (left, left, right, right)


def wheel_speeds(center):
    correction = STEERING_K * (center - 320)
    left = max(0, min(MAX_SPEED, round(BASE_SPEED + correction)))
    right = max(0, min(MAX_SPEED, round(BASE_SPEED - correction)))
    return (left, left, right, right)


class LaneController:
    """장치 없이 선 위치·시간으로 바퀴 값을 계산한다."""
    def __init__(self):
        self.single_name = None
        self.width = None
        self.previous = None
        self.center = None
        self.raw_center = None
        self.centers = deque(maxlen=3)
        self.last_frame = None
        self.lost_since = None
        self.last_turn = None
        self.speeds = STOP
        self.stopped = False
        self.state = 'WAITFRAME'
        self.lines = dict(left=None, right=None, left_count=0, right_count=0)

    def tick(self, now):
        if self.stopped:
            return STOP
        if self.last_frame is not None and now - self.last_frame > 0.5:
            self.state = 'FRAMESTOP'
            self.stopped = True
        elif self.lost_since is not None and now - self.lost_since >= SEARCH_SECONDS:
            self.state = 'LOSTSTOP'
            self.stopped = True
        if self.stopped:
            self.speeds = STOP
        return self.speeds

    def update(self, lines, now):
        self.tick(now)
        if self.stopped:
            return STOP
        self.last_frame = now
        # 직접 입력된 두 후보도 같은 규칙으로 정리한다.
        if lines['left'] is not None and lines['right'] is not None:
            lines = assign_lines([(lines[name], lines[name+'_count']) for name in ('left','right')],
                                 self.previous, self.width)
        self.lines = dict(lines)
        self.raw_center = None
        left, right = lines['left'], lines['right']
        if (left is None) != (right is None):
            self.single_name = 'left' if left is not None else 'right'
        if lines.get('unassigned'):
            self.state = 'WAITWIDTH'
            self.center = None
            self.speeds = STOP
            return STOP
        if left is not None and right is not None and self.previous is None:
            self.previous = dict(left=left, right=right)
        elif self.previous is not None:
            for name in ('left', 'right'):
                if lines[name] is not None:
                    self.previous[name] = lines[name]
        if left is None and right is None:
            if self.lost_since is None:
                self.lost_since = now
            self.center = None
            if self.last_turn is None:
                self.state = 'LOSTSTOP'
                self.stopped = True
                self.speeds = STOP
            else:
                self.state = 'SEARCH' + self.last_turn
                target = (60, 60, 0, 0) if self.last_turn == 'RIGHT' else (0, 0, 60, 60)
                self.speeds = limit_difference(target, self.speeds)
            return self.speeds
        self.lost_since = None
        if left is not None and right is not None:
            if 300 <= right - left <= 600:
                self.width = right - left
            self.center = (left + right) / 2
            self.state = 'TRACKBOTH'
        elif self.width is None:
            self.center = None
            self.speeds = STOP
            self.state = 'WAITWIDTH'
            return STOP
        elif left is not None:
            self.center = left + self.width / 2
            self.state = 'TRACKLEFT'
        else:
            self.center = right - self.width / 2
            self.state = 'TRACKRIGHT'
        self.raw_center = max(0., min(640., self.center))
        self.centers.append(self.raw_center)
        self.center = sum(self.centers) / len(self.centers)
        self.speeds = limit_difference(wheel_speeds(self.center), self.speeds)
        if self.speeds[0] > self.speeds[2]:
            self.last_turn = 'RIGHT'
        elif self.speeds[0] < self.speeds[2]:
            self.last_turn = 'LEFT'
        return self.speeds

    def report(self):
        return dict(self.lines, width=self.width, center=self.center,
                    raw_center=self.raw_center, speeds=self.speeds, state=self.state, stopped=self.stopped, last_turn=self.last_turn)


def annotated(frame, report):
    out = frame.copy()
    cv2.rectangle(out, (0, ROI_TOP), (639, ROI_BOTTOM - 1), (255, 255, 255), 1)
    cv2.line(out, (320, ROI_TOP), (320, ROI_BOTTOM - 1), (255, 255, 255), 1)
    for name in ('left', 'right'):
        position = report[name]
        if position is not None:
            x = round(position)
            cv2.line(out, (x, ROI_TOP), (x, ROI_BOTTOM - 1), (0, 255, 255), 2)
            cv2.putText(out, f'{name.upper()} {position:.1f}', (max(2,min(480,x)),112),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,255), 1, cv2.LINE_AA)
    if report['center'] is not None:
        x = round(report['center'])
        cv2.line(out,(x,ROI_TOP),(x,ROI_BOTTOM - 1),(0,255,0),2)
    text = f"CENTER {report['center']} WIDTH {report['width']} WHEELS {report['speeds']}"
    cv2.putText(out,text,(8,330),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,report['state'],(8,355),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    return out


class LatestDisplay:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0

    def update(self, frame, report):
        marked = annotated(frame, report)
        with self.lock:
            self.frame = marked
            self.sequence += 1

    def snapshot(self):
        with self.lock:
            if self.frame is None:
                raise RuntimeError('녹화할 차선 화면이 없습니다.')
            return self.frame.copy(), self.sequence


class LaneVideo(video.BlackboxRecorder):
    def __init__(self, source):
        super().__init__(source)
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, name='lane-video')

    def run(self):
        writer = None
        previous = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            while not self.stop.is_set():
                if self.stop.wait(max(0,start+self.saved_frames/video.FPS-time.monotonic())):
                    break
                frame, sequence = self.camera.snapshot()
                if self.saved_frames % self.segment_frames == 0:
                    if writer is not None:
                        writer.release()
                        writer = None
                    writer = self._open_writer(frame)
                writer.write(self._overlay(frame))
                self.repeated_frames += int(sequence == previous)
                previous = sequence
                self.saved_frames += 1
        except Exception as exc:
            self.error = exc
        finally:
            if writer is not None:
                writer.release()

    def start(self):
        self.thread.start()
        return self

    def check(self):
        if self.error is not None:
            raise RuntimeError('차선 영상 녹화 오류') from self.error

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()
        self.check()


FIELDS = ('경과초','화면순번','상태','왼쪽선_x','오른쪽선_x','왼쪽노란점개수','오른쪽노란점개수',
          '차선폭_px','차선중심_x','바퀴값','모드','평균전중심_x')


def row_values(report, elapsed, sequence, check):
    return dict(zip(FIELDS,(round(elapsed,4),sequence,report['state'],report['left'],report['right'],
                            report['left_count'],report['right_count'],report['width'],report['center'],
                            json.dumps(report['speeds']), '읽기전용' if check else '주행', report['raw_center'])))


def run_loop(camera, recorder, display, key_pressed, stream, check=False, bot=None):
    if not check and bot is None:
        raise ValueError('주행 모드에는 바퀴 장치가 필요합니다.')
    controller = LaneController()
    started = time.monotonic()
    next_print = started
    sequence = None
    frame = None
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    last_record = None
    reason = '키 입력'
    try:
        while True:
            if key_pressed():
                break
            now = time.monotonic()
            controller.tick(now)
            candidate, new_sequence = camera.snapshot()
            received_at = time.monotonic()
            if new_sequence != sequence:
                lines = find_lines(candidate, controller.previous, controller.width, controller.single_name)
                # 선 찾기가 늦어진 경우에도 경과 시간 한계를 먼저 확인한다.
                now = time.monotonic()
                controller.update(lines, received_at)
                frame, sequence = candidate, new_sequence
            if key_pressed():
                break
            controller.tick(time.monotonic())
            report = controller.report()
            if not check:
                line.drive(bot, report['speeds'])
            display.update(frame, report)
            if recorder is not None:
                recorder.set_status(report['state'])
                recorder.check()
            current = (sequence, report['state'])
            if current != last_record:
                writer.writerow(row_values(report,now-started,sequence,check))
                stream.flush()
                last_record = current
            if now >= next_print:
                print('차선 계산: ' + json.dumps(report,ensure_ascii=False),flush=True)
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


def _main(stop_requested):
    parser = argparse.ArgumentParser(description='노란 차선 주행 / --check는 모터 명령 없는 카메라 점검')
    parser.add_argument('--check',action='store_true',help='모터 없이 선과 바퀴 계산 값을 1초마다 출력')
    args = parser.parse_args()
    camera = recorder = bot = None
    entered = False
    stamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S_%f')
    path = ROOT/'results'/f'camera_lane_{stamp}.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    display = LatestDisplay()
    try:
        with line.keyboard_stop() as terminal_key:
            key_pressed = lambda: stop_requested.is_set() or terminal_key()
            if not args.check:
                from Raspbot_Lib import Raspbot
                bot = Raspbot()
                line.stop_all(bot)
            camera = video.LatestCamera()
            camera.start()
            if stop_requested.is_set():
                return
            first, _ = camera.snapshot()
            display.update(first,LaneController().report())
            recorder = LaneVideo(display)
            recorder.start()
            with path.open('x',newline='',encoding='utf-8') as stream:
                entered = True
                reason, report = run_loop(camera,recorder,display,key_pressed,stream,args.check,bot)
            print(f'종료: {reason}, 기록: {path}',flush=True)
    finally:
        errors = []
        if bot is not None and not entered:
            try:
                line.stop_all(bot)
            except Exception as exc:
                errors.append(exc)
        for resource in (recorder,camera):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError('차선 종료 정리 오류: '+'; '.join(map(str,errors)))


@contextmanager
def stop_signals():
    requested = threading.Event()
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: requested.set())
    try:
        yield requested
    finally:
        signal.signal(signal.SIGINT, previous)


def main():
    with stop_signals() as requested:
        try:
            _main(requested)
        except KeyboardInterrupt:
            # 시작 준비 중 중단도 _main의 정리 이후 조용히 끝낸다.
            pass


if __name__ == '__main__':
    main()
