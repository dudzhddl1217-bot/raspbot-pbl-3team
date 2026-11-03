#!/usr/bin/env python3
# 지시서: specs/03-video.md
"""최신 카메라 화면 공유와 15fps 블랙박스 녹화. 바퀴는 제어하지 않는다."""
import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from yolo import open_camera

VIDEO_ROOT = ROOT / "results" / "videos"
KST = ZoneInfo("Asia/Seoul")
FPS = 15


class LatestCamera:
    """카메라는 이 스레드만 읽는다. 소비자는 snapshot()으로 복사본을 받는다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._frame = None
        self._sequence = 0
        self._error = None
        self._thread = threading.Thread(target=self._capture, name="blackbox-camera")

    def _capture(self):
        cap = None
        try:
            cap = open_camera()
            if not cap.isOpened():
                raise RuntimeError("카메라를 열지 못했습니다. USB 연결을 확인하세요.")
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    # 일시적인 프레임 부족에는 마지막 화면을 유지한다.
                    self._stop.wait(0.01)
                    continue
                with self._lock:
                    self._frame = frame
                    self._sequence += 1
                self._ready.set()
        except Exception as exc:
            with self._lock:
                self._error = exc
            self._ready.set()
        finally:
            if cap is not None:
                cap.release()

    def start(self):
        self._thread.start()
        if not self._ready.wait(10):
            self.close()
            raise RuntimeError("10초 동안 카메라 화면을 받지 못했습니다.")
        self.snapshot()
        return self

    def snapshot(self):
        """(화면 복사본, 촬영 순번). YOLO도 같은 인스턴스를 공유할 수 있다."""
        with self._lock:
            if self._error is not None:
                raise RuntimeError("카메라 읽기 스레드 오류") from self._error
            if self._frame is None:
                raise RuntimeError("아직 카메라 화면이 없습니다.")
            return self._frame.copy(), self._sequence

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()


class BlackboxRecorder:
    """공유 카메라를 받아 녹화한다. 외부 제어는 set_status('STATE')를 쓴다."""

    def __init__(self, camera, output_dir=VIDEO_ROOT, segment_seconds=60, keep=30):
        self.camera = camera
        self.output_dir = Path(output_dir).resolve()
        if not self.output_dir.is_relative_to(VIDEO_ROOT.resolve()):
            raise ValueError("저장 폴더는 results/videos/ 안이어야 합니다.")
        if not math.isfinite(segment_seconds) or segment_seconds <= 0 or keep < 1:
            raise ValueError("분할 시간과 보관 개수는 양수여야 합니다.")
        self.segment_frames = round(segment_seconds * FPS)
        if self.segment_frames < 1:
            raise ValueError("분할 시간은 최소 한 프레임 길이여야 합니다.")
        self.keep = keep
        self._status_lock = threading.Lock()
        self._status = "REC"
        self.paths = []
        self.saved_frames = 0
        self.repeated_frames = 0

    def set_status(self, text):
        if not text or not text.isascii() or not text.isalnum():
            raise ValueError("상태 글자는 영어와 숫자만 사용하세요.")
        with self._status_lock:
            self._status = text

    def _overlay(self, frame):
        with self._status_lock:
            status = self._status
        timestamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        for text, position in ((timestamp, (12, 28)), (status, (12, frame.shape[0] - 16))):
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _prune(self, active):
        files = sorted((p for p in self.output_dir.glob("blackbox_*.mp4")
                        if p.is_file() and not p.is_symlink()),
                       key=lambda p: (p.stat().st_mtime_ns, p.name))
        for path in files[:max(0, len(files) - self.keep)]:
            if path != active:
                path.unlink()
                print(f"삭제: {path}", flush=True)

    def _open_writer(self, frame):
        stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"blackbox_{stamp}_{uuid4().hex[:8]}.mp4"
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"),
                                 FPS, (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("H.264(avc1) mp4 파일을 열지 못했습니다.")
        self.paths.append(path)
        try:
            self._prune(path)
        except BaseException:
            writer.release()
            raise
        print(f"파일: {path}", flush=True)
        return writer

    def record(self, seconds=None):
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("시험 시간은 양수여야 합니다.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        writer = None
        previous_sequence = None
        start = time.monotonic()
        limit = math.ceil(seconds * FPS) if seconds is not None else None
        try:
            while limit is None or self.saved_frames < limit:
                # 프레임별 절대 시각으로 예약해 처리 시간이 누적되지 않게 한다.
                due = start + self.saved_frames / FPS
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                frame, sequence = self.camera.snapshot()
                if self.saved_frames % self.segment_frames == 0:
                    if writer is not None:
                        writer.release()
                        writer = None
                    writer = self._open_writer(frame)
                if sequence == previous_sequence:
                    self.repeated_frames += 1
                writer.write(self._overlay(frame))
                previous_sequence = sequence
                self.saved_frames += 1
            # 마지막 프레임을 포함한 실제 녹화 시간이 지정 시간까지 가도록 기다린다.
            if seconds is not None:
                time.sleep(max(0, start + seconds - time.monotonic()))
        except KeyboardInterrupt:
            print("Ctrl+C: 녹화를 정리합니다.", flush=True)
        finally:
            if writer is not None:
                writer.release()
            elapsed = time.monotonic() - start
            print(f"녹화 종료: 실제 {elapsed:.3f}초, 저장 {self.saved_frames}장, "
                  f"재생 {self.saved_frames / FPS:.3f}초, "
                  f"반복 {self.repeated_frames}장", flush=True)


def main():
    parser = argparse.ArgumentParser(description="15fps 블랙박스 영상 녹화")
    parser.add_argument("--seconds", type=float, help="시험 녹화 시간(생략하면 Ctrl+C까지)")
    parser.add_argument("--segment-seconds", type=float, default=60)
    parser.add_argument("--keep", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=VIDEO_ROOT)
    args = parser.parse_args()
    camera = LatestCamera()
    try:
        camera.start()
        recorder = BlackboxRecorder(camera, args.output_dir, args.segment_seconds, args.keep)
        recorder.record(args.seconds)
    except KeyboardInterrupt:
        print("Ctrl+C: 카메라를 종료합니다.", flush=True)
    finally:
        camera.close()


if __name__ == "__main__":
    main()
