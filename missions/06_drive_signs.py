#!/usr/bin/env python3
# 지시서: specs/06-drive-signs.md
"""줄 따라가기·사람/STOP 반응. 실제 실행 전 사용자 안전 확인이 필요하다."""
import argparse
import csv
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'missions' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


line = load('drive_signs_line', '02_line.py')
video = load('drive_signs_video', '03_video.py')
signs = load('drive_signs_detector', '05_signs.py')
STOP = line.STOP
KST = ZoneInfo('Asia/Seoul')


def targets(found):
    return [(name, float(score), tuple(box)) for name, score, box in found
            if name in ('person', 'stop sign') and math.isfinite(score) and score >= 0.4]


class SignController:
    """새 인식 결과와 센서·시각만 받아 판단한다. 장치 호출은 없다."""
    def __init__(self):
        self.line = line.LineFollower()
        self.sequence = None
        self.counts = {'person': 0, 'stop sign': 0}
        self.person = False
        self.person_absent = None
        self.stop_absent = None
        self.stop_armed = True
        self.stop_pending = False
        self.stop_until = None
        self.resume_pending = False
        self.ignore_until = None
        self.paused = True
        self.state = 'WAITVISION'
        self.found = []

    def invalidate(self):
        self.counts = {'person': 0, 'stop sign': 0}
        self.person_absent = self.stop_absent = None

    def observe(self, result, now):
        self.sequence = result['sequence']
        self.found = targets(result['found'])
        names = {name for name, _, _ in self.found}
        for name in self.counts:
            self.counts[name] = min(2, self.counts[name] + 1) if name in names else 0
        if self.counts['person'] >= 2:
            self.person = True
        if 'person' in names:
            self.person_absent = None
        elif self.person:
            if self.person_absent is None:
                self.person_absent = now
            if now - self.person_absent >= 1:
                self.person = False
                self.person_absent = None
        if 'stop sign' not in names:
            if self.stop_absent is None:
                self.stop_absent = now
            if now - self.stop_absent >= 1:
                self.stop_armed = True
        else:
            self.stop_absent = None
        allowed = self.ignore_until is None or now >= self.ignore_until
        if (self.stop_armed and not self.resume_pending and allowed
                and self.counts['stop sign'] >= 2):
            self.stop_armed = False
            self.stop_pending = True
            self.resume_pending = True

    def step(self, bits, result, now):
        if self.line.stopped:
            self.state = 'LOSTSTOP'
            return STOP
        fresh = result is not None and 0 <= now - result['captured_at'] <= 1
        if fresh and result['sequence'] != self.sequence:
            self.observe(result, now)
        if not fresh:
            self.invalidate()
            self.state = 'WAITVISION' if result is None else 'STALEVISION'
        elif self.person:
            self.state = 'PERSONWAIT'
        elif self.stop_pending or (self.stop_until is not None and now < self.stop_until):
            self.state = 'STOPWAIT'
        else:
            if self.paused:
                # 정지 이전 탐색 시간은 제외하고 새로 1초를 준다. 방향은 보존한다.
                self.line.lost_since = None
            speeds = self.line.step(bits, now)
            self.state = 'LOSTSTOP' if self.line.stopped else self.line.status.replace('_', '')
            self.paused = False
            return speeds
        self.paused = True
        return STOP

    def motor_sent(self, speeds, now):
        """첫 정지/재출발 명령 전달 시각을 기준으로 STOP 시계를 시작한다."""
        if self.stop_pending and speeds == STOP:
            self.stop_pending = False
            self.stop_until = now + 3
        if self.resume_pending and any(speeds):
            self.resume_pending = False
            self.ignore_until = now + 5
            self.stop_until = None


class VisionWorker:
    """새 카메라 화면만 한 번씩 인식하고 최신 결과 하나만 공유한다."""
    def __init__(self, camera, detector=None):
        self.camera = camera
        self.detector = detector
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.thread = threading.Thread(target=self.run, name='drive-signs-yolo', daemon=True)

    def run(self):
        previous = None
        try:
            detector = self.detector if self.detector is not None else signs.make_detector(640)
            while not self.stop.is_set():
                frame, sequence = self.camera.snapshot()
                if sequence == previous:
                    self.stop.wait(0.005)
                    continue
                captured_at = time.monotonic()
                found = targets(detector(frame))
                with self.lock:
                    self.latest = {'sequence': sequence, 'captured_at': captured_at,
                                   'completed_at': time.monotonic(), 'found': found}
                previous = sequence
        except Exception as exc:
            self.error = exc

    def start(self):
        self.thread.start()
        return self

    def snapshot(self):
        if self.error is not None:
            raise RuntimeError('YOLO 스레드 오류') from self.error
        with self.lock:
            return None if self.latest is None else dict(self.latest, found=list(self.latest['found']))

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('YOLO 스레드가 종료 요청에 응답하지 않습니다.')


class DriveVideo(video.BlackboxRecorder):
    """03의 파일 분할·보관을 재사용하고 별도 스레드에서 즉시 녹화한다."""
    def __init__(self, camera):
        super().__init__(camera)
        self.stop = threading.Event()
        self.error = None
        self.info = 'WAITVISION'
        self.thread = threading.Thread(target=self.run, name='drive-signs-video', daemon=True)

    def update(self, state, bits, found):
        seen = ' '.join(f'{name} {score:.2f} H{box[3]}' for name, score, box in found) or 'NONE'
        with self._status_lock:
            self._status = state
            self.info = 'LINE ' + ''.join(map(str, bits)) + ' SEEN ' + seen

    def _overlay(self, frame):
        frame = super()._overlay(frame)
        with self._status_lock:
            info = self.info
        video.cv2.putText(frame, info, (12, 55), video.cv2.FONT_HERSHEY_SIMPLEX,
                          0.45, (255, 255, 255), 1, video.cv2.LINE_AA)
        return frame

    def run(self):
        writer = None
        previous = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            while not self.stop.is_set():
                if self.stop.wait(max(0, started + self.saved_frames / video.FPS - time.monotonic())):
                    break
                frame, sequence = self.camera.snapshot()
                if self.saved_frames % self.segment_frames == 0:
                    if writer is not None:
                        writer.release()
                        writer = None
                    writer = self._open_writer(frame)
                if sequence == previous:
                    self.repeated_frames += 1
                writer.write(self._overlay(frame.copy()))
                self.saved_frames += 1
                previous = sequence
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
            raise RuntimeError('녹화 스레드 오류') from self.error

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('녹화 스레드가 종료 요청에 응답하지 않습니다.')
        self.check()


DETAIL_FIELDS = ('경과초', '상태', 'x1', 'x2', 'x3', 'x4', '바퀴속도', '화면순번',
                 '결과나이_초', '인식이름', '점수', '상자높이_px', 'STOP정지종료', 'STOP무시종료')


def run_loop(bot, worker, recorder, key_pressed, output):
    controller = SignController()
    writer = csv.DictWriter(output, fieldnames=DETAIL_FIELDS, lineterminator='\n')
    writer.writeheader()
    started = time.monotonic()
    reason = '오류'
    previous = None
    try:
        while True:
            if key_pressed():
                reason = '키 입력'
                break
            recorder.check()
            result = worker.snapshot()
            bits = line.read_sensors(bot)
            if key_pressed():
                reason = '키 입력'
                break
            now = time.monotonic()
            speeds = controller.step(bits, result, now)
            first_sent = time.monotonic()
            line.drive(bot, speeds)
            controller.motor_sent(speeds, first_sent)
            recorder.update(controller.state, bits, controller.found)
            current = (controller.state, bits, speeds, controller.sequence)
            if current != previous:
                found = controller.found
                writer.writerow(dict(zip(DETAIL_FIELDS, (
                    round(now-started, 4), controller.state, *bits, json.dumps(speeds),
                    '' if result is None else result['sequence'],
                    '' if result is None else round(now-result['captured_at'], 4),
                    ';'.join(n for n,_,_ in found), ';'.join(f'{s:.6f}' for _,s,_ in found),
                    ';'.join(str(b[3]) for _,_,b in found), controller.stop_until, controller.ignore_until))))
                output.flush()
                if previous is None or current[0] != previous[0]:
                    print(f'상태 {controller.state}, 센서 {bits}, 바퀴 {speeds}', flush=True)
                previous = current
            if controller.line.stopped:
                reason = '줄 놓침 정지'
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        reason = 'Ctrl+C'
    finally:
        line.stop_all(bot)
    return reason, round(time.monotonic()-started, 4)


def append_run(path, trial, reason, elapsed, raw, paths):
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
        if subprocess.check_output(['git', 'status', '--porcelain', '--', 'missions/06_drive_signs.py'], cwd=ROOT, text=True).strip():
            commit += '+미커밋'
    except subprocess.CalledProcessError:
        commit = '미확인'
    row = dict.fromkeys(fields, '')
    values = {'회차': trial, '코드 버전(커밋 번호)': commit, '시도':'06 사람·STOP 주행',
              '완주':'사용자 확인 대기', '이탈':'사용자 확인 대기', '시간(초)': elapsed,
              '손댄 횟수':'사용자 확인 대기', '메모': f'{reason}; 실제 동작 사용자 확인 대기; 원시 기록 {raw}; 영상 {paths}',
              '날짜': datetime.now(KST).strftime('%Y-%m-%d'), '프로그램':'missions/06_drive_signs.py',
              '속도':50, '원시기록':str(raw), '종료후전체정지확인':'사용자 확인 대기'}
    row.update({k:v for k,v in values.items() if k in fields})
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='줄 따라가면서 사람·STOP에 반응 (안전 확인 후 실행)')
    parser.add_argument('--trial', required=True, help='시험 회차 이름')
    args = parser.parse_args()
    from Raspbot_Lib import Raspbot
    bot = None
    camera = worker = recorder = None
    started = time.monotonic()
    reason, elapsed = '시작 준비 오류', 0
    loop_entered = False
    session = datetime.now(KST).strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid4().hex[:6]
    raw = ROOT / 'results' / f'drive_signs_{session}.csv'
    raw.parent.mkdir(parents=True, exist_ok=True)
    try:
        with line.keyboard_stop() as key_pressed:
            bot = Raspbot()
            line.stop_all(bot)
            camera = video.LatestCamera()
            camera.start()
            worker = VisionWorker(camera)
            worker.start()
            recorder = DriveVideo(camera)
            recorder.start()
            bot.Ctrl_BEEP_Switch(1)
            try:
                time.sleep(0.2)
            finally:
                bot.Ctrl_BEEP_Switch(0)
            with raw.open('x', newline='', encoding='utf-8') as output:
                loop_entered = True
                reason, elapsed = run_loop(bot, worker, recorder, key_pressed, output)
    except KeyboardInterrupt:
        reason = 'Ctrl+C'
    except Exception as exc:
        reason = f'오류: {exc}'
        raise
    finally:
        errors = []
        if bot is not None:
            cleanups = [lambda: bot.Ctrl_BEEP_Switch(0)]
            if not loop_entered:
                cleanups.insert(0, lambda: line.stop_all(bot))
            for cleanup in cleanups:
                try:
                    cleanup()
                except Exception as exc:
                    errors.append(exc)
        for resource in (recorder, worker, camera):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(exc)
        if elapsed == 0:
            elapsed = round(time.monotonic()-started, 4)
        append_run(ROOT / 'results/runs.csv', args.trial, reason, elapsed, raw,
                   [] if recorder is None else [str(p) for p in recorder.paths])
        print(f'종료: {reason}, 기록: {raw}. 실제 완주·정지는 사용자 확인이 필요합니다.', flush=True)
        if errors:
            raise RuntimeError('종료 정리 오류: ' + '; '.join(map(str, errors)))


if __name__ == '__main__':
    main()
