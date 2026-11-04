#!/usr/bin/env python3
# 지시서: specs/07-camera-lane.md
"""노란 차선 비례 조향. --check는 드라이버를 불러오거나 모터 명령을 보내지 않는다."""
import argparse
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
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


def find_lines(frame):
    if frame is None or frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise ValueError('차선 입력 화면은 640x480 uint8 BGR이어야 합니다.')
    hsv = cv2.cvtColor(frame[120:220], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 80, 100), (40, 255, 255))
    result = {}
    for name, start, end in (('left', 0, 320), ('right', 320, 640)):
        _, xs = np.nonzero(mask[:, start:end])
        result[name + '_count'] = len(xs)
        result[name] = float(xs.mean() + start) if len(xs) >= 50 else None
    return result


class LaneController:
    """장치 없이 선 위치·시간으로 바퀴 값을 계산한다."""
    def __init__(self):
        self.width = None
        self.center = None
        self.last_frame = None
        self.lost_since = None
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
        elif self.lost_since is not None and now - self.lost_since > 0.5:
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
        self.lines = dict(lines)
        left, right = lines['left'], lines['right']
        if left is None and right is None:
            if self.lost_since is None:
                self.lost_since = now
            self.center = None
            self.state = 'LOSTGRACE'
            return self.speeds
        self.lost_since = None
        if left is not None and right is not None:
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
        correction = .2 * (self.center - 320)
        lvalue = max(0, min(80, round(40 + correction)))
        rvalue = max(0, min(80, round(40 - correction)))
        self.speeds = (lvalue, lvalue, rvalue, rvalue)
        return self.speeds

    def report(self):
        return dict(self.lines, width=self.width, center=self.center,
                    speeds=self.speeds, state=self.state, stopped=self.stopped)


def annotated(frame, report):
    out = frame.copy()
    cv2.rectangle(out, (0, 120), (639, 219), (255, 255, 255), 1)
    cv2.line(out, (320, 120), (320, 219), (255, 255, 255), 1)
    for name in ('left', 'right'):
        position = report[name]
        if position is not None:
            x = round(position)
            cv2.line(out, (x, 120), (x, 219), (0, 255, 255), 2)
            cv2.putText(out, f'{name.upper()} {position:.1f}', (max(2,min(480,x)),112),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,255), 1, cv2.LINE_AA)
    if report['center'] is not None:
        x = round(report['center'])
        cv2.line(out,(x,120),(x,219),(0,255,0),2)
    text = f"CENTER {report['center']} WIDTH {report['width']} WHEELS {report['speeds']}"
    cv2.putText(out,text,(8,250),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,report['state'],(8,275),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
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
        self.thread = threading.Thread(target=self.run, name='lane-video', daemon=True)

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
            self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('차선 녹화 종료 응답 없음')
        self.check()


FIELDS = ('경과초','화면순번','상태','왼쪽선_x','오른쪽선_x','왼쪽노란점개수','오른쪽노란점개수',
          '차선폭_px','차선중심_x','바퀴값','모드')


def row_values(report, elapsed, sequence, check):
    return dict(zip(FIELDS,(round(elapsed,4),sequence,report['state'],report['left'],report['right'],
                            report['left_count'],report['right_count'],report['width'],report['center'],
                            json.dumps(report['speeds']), '읽기전용' if check else '주행')))


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
                lines = find_lines(candidate)
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


def main():
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
        with line.keyboard_stop() as key_pressed:
            if not args.check:
                from Raspbot_Lib import Raspbot
                bot = Raspbot()
                line.stop_all(bot)
            camera = video.LatestCamera()
            camera.start()
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


if __name__ == '__main__':
    main()
