#!/usr/bin/env python3
# 지시서: specs/05-signs.md
"""정지된 라즈봇에서 인쇄 표지 30장 촬영·인식·채점. 모터 접근 없음."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import importlib.util
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from yolo import Detector, draw

video_module_spec = importlib.util.spec_from_file_location("blackbox_camera", ROOT / "missions/03_video.py")
video_module = importlib.util.module_from_spec(video_module_spec)
video_module_spec.loader.exec_module(video_module)
LatestCamera = video_module.LatestCamera

TARGETS = ("stop sign", "person", "car")
CONFIDENCE = 0.4
QUOTAS = {30: 4, 50: 3, 70: 3}
ANGLES = ("정면", "왼쪽", "오른쪽")
PHOTO_ROOT = ROOT / "results/signs_images"
VIDEO_ROOT = ROOT / "results/videos"
FPS = 15
KST = ZoneInfo("Asia/Seoul")
FIELDS = ("사진번호", "정답", "거리_cm", "각도", "인식이름", "점수", "맞힘", "원본사진", "결과사진")


def make_detector(size=320):
    # 기존 Detector와 NMS는 '초과' 비교다. float32 한 칸 아래로 열고 아래에서 >=0.4를 적용한다.
    threshold = float(np.nextafter(np.float32(CONFIDENCE), np.float32(0)))
    return Detector(size=size, conf=threshold)


def accepted(found):
    return [(name, float(score), tuple(box)) for name, score, box in found
            if name in TARGETS and np.isfinite(score) and float(score) >= CONFIDENCE]


def annotated(frame, found):
    return draw(frame.copy(), found)


class InferenceWorker:
    """계속 최신 화면을 인식한다. 엔터 촬영 때만 그 순간 사진 한 장을 별도로 처리한다."""

    def __init__(self, camera, detector):
        self.camera, self.detector = camera, detector
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._error = None
        self._photo = Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, name="signs-yolo")

    def _run(self):
        previous = None
        request = None
        try:
            while not self._stop.is_set():
                try:
                    request = self._photo.get_nowait()
                except Empty:
                    request = None
                if request is None:
                    frame, sequence = self.camera.snapshot()
                    if sequence == previous:
                        self._stop.wait(0.005)
                        continue
                else:
                    frame, sequence = request["frame"], request["sequence"]
                found = accepted(self.detector(frame))
                with self._lock:
                    self._latest = (frame, found, sequence)
                previous = sequence
                self._ready.set()
                if request is not None:
                    request["found"] = found
                    request["done"].set()
        except Exception as exc:
            self._error = exc
            if request is not None:
                request["done"].set()
            self._ready.set()

    def check(self):
        if self._error is not None:
            raise RuntimeError("YOLO 인식 스레드 오류") from self._error

    def start(self):
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("10초 동안 YOLO 결과가 없습니다.")
        self.check()
        return self

    def snapshot(self):
        self.check()
        with self._lock:
            if self._latest is None:
                raise RuntimeError("아직 인식 결과가 없습니다.")
            frame, found, sequence = self._latest
            return frame.copy(), list(found), sequence

    def analyze_photo(self, frame, sequence):
        self.check()
        request = {"frame": frame.copy(), "sequence": sequence, "done": threading.Event()}
        self._photo.put_nowait(request)
        if not request["done"].wait(10):
            self.check()
            raise RuntimeError("촬영한 사진의 인식이 끝나지 않았습니다.")
        self.check()
        return request["found"]

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()


class ResultVideo:
    """완료된 인식 화면을 15fps로 즉시 저장한다. 느리면 마지막 화면을 반복한다."""

    def __init__(self, source, path):
        self.source, self.path = source, Path(path)
        self.frames = 0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="signs-video")

    def _run(self):
        writer = None
        try:
            start = time.monotonic()
            while not self._stop.is_set():
                delay = max(0, start + self.frames / FPS - time.monotonic())
                if self._stop.wait(delay):
                    break
                frame, found, _ = self.source.snapshot()
                if writer is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("H.264 결과 영상을 열지 못했습니다.")
                writer.write(annotated(frame, found))
                self.frames += 1
                self._ready.set()
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            if writer is not None:
                writer.release()

    def check(self):
        if self._error is not None:
            raise RuntimeError("결과 영상 저장 오류") from self._error

    def start(self):
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("결과 영상 저장을 시작하지 못했습니다.")
        self.check()
        return self

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()
        self.check()


def write_table(path, rows, fields=FIELDS):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def assess(rows):
    complete = [r for r in rows if r["맞힘"] in (0, 1)]
    counts = Counter((r["정답"], r["거리_cm"]) for r in complete)
    plan_ok = len(complete) == 30
    for name in TARGETS:
        for distance, quota in QUOTAS.items():
            plan_ok &= counts[(name, distance)] == quota
            angles = {r["각도"] for r in complete if r["정답"] == name and r["거리_cm"] == distance}
            plan_ok &= set(ANGLES).issubset(angles)
    correct = sum(r["맞힘"] for r in complete)
    return {"complete": len(complete), "correct": correct, "plan_ok": bool(plan_ok),
            "passed": bool(plan_ok and correct >= 24),
            "by_class": {name: sum(r["맞힘"] for r in complete if r["정답"] == name) for name in TARGETS}}


def choose(prompt, options, input_fn=input):
    while True:
        value = input_fn(prompt).strip()
        if value in options:
            return options[value]
        print("안내된 선택지로 입력해 주세요.", flush=True)


def ask_metadata(input_fn=input):
    truth = choose("정답: 1 stop sign / 2 person / 3 car: ",
                   {"1": "stop sign", "2": "person", "3": "car", **{s: s for s in TARGETS}}, input_fn)
    distance = choose("카메라와 표지 거리(cm): 30 / 50 / 70: ",
                      {str(d): d for d in QUOTAS}, input_fn)
    angle = choose("각도: 1 정면 / 2 왼쪽 비스듬히 / 3 오른쪽 비스듬히: ",
                   {"1": "정면", "2": "왼쪽", "3": "오른쪽"}, input_fn)
    return truth, distance, angle


def collect_photos(camera, worker, video, photo_dir, table_path, input_fn=input):
    rows = []
    photo_dir = Path(photo_dir)
    photo_dir.mkdir(parents=True, exist_ok=True)
    write_table(table_path, rows)
    for number in range(1, 31):
        counts = Counter((r["정답"], r["거리_cm"]) for r in rows if r["맞힘"] in (0, 1))
        for name in TARGETS:
            remaining = " / ".join(f"{d}cm 남은 {max(0,q-counts[(name,d)])}장" for d,q in QUOTAS.items())
            print(f"{name}: {remaining}", flush=True)
        input_fn(f"[{number}/30] 표지판을 놓고 엔터를 누르면 촬영합니다. 종료는 Ctrl+C: ")
        video.check()
        worker.check()
        frame, sequence = camera.snapshot()
        raw_path = photo_dir / f"photo_{number:02d}_raw.jpg"
        marked_path = photo_dir / f"photo_{number:02d}_result.jpg"
        if not cv2.imwrite(str(raw_path), frame):
            raise RuntimeError("원본 사진 저장 실패")
        row = {field: "" for field in FIELDS}
        row.update(사진번호=number, 원본사진=str(raw_path.relative_to(ROOT)), 결과사진=str(marked_path.relative_to(ROOT)))
        rows.append(row)
        write_table(table_path, rows)
        found = worker.analyze_photo(frame, sequence)
        if not cv2.imwrite(str(marked_path), annotated(frame, found)):
            raise RuntimeError("인식 결과 사진 저장 실패")
        # 정답 입력 전에 분석한다. 사용자 정답으로 검출 결과를 바꾸지 않는다.
        truth, distance, angle = ask_metadata(input_fn)
        row.update(정답=truth, 거리_cm=distance, 각도=angle,
                   인식이름=";".join(name for name,_,_ in found),
                   점수=";".join(f"{score:.6f}" for _,score,_ in found),
                   맞힘=int(any(name == truth for name,_,_ in found)))
        write_table(table_path, rows)
        summary = assess(rows)
        print(f"이번 사진: {'맞힘' if row['맞힘'] else '못 맞힘'}, 현재 {summary['correct']}/{summary['complete']}", flush=True)
    return assess(rows)


def rescore(source, size):
    """기존 원본 사진만 재채점한다. 카메라·모터에 접근하지 않는다."""
    source = Path(source).resolve()
    if source.is_dir():
        source = ROOT / "results" / f"signs_{source.name}.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30:
        raise ValueError("재채점에는 메타정보 입력을 마친 30장 CSV가 필요합니다.")
    if any(row.get("모델", "yolo26n_320") != "yolo26n_320" for row in rows):
        raise ValueError("비교 기준에는 원래 320 모델로 채점한 CSV를 지정해 주세요.")
    photos = []
    for row in rows:
        row["사진번호"] = int(row["사진번호"])
        row["거리_cm"] = int(row["거리_cm"])
        row["맞힘"] = int(row["맞힘"])
        if (row["정답"] not in TARGETS or row["거리_cm"] not in QUOTAS
                or row["각도"] not in ANGLES or row["맞힘"] not in (0, 1)):
            raise ValueError("원본 CSV의 정답·거리·각도·맞힘 값을 확인해 주세요.")
        path = (ROOT / row["원본사진"]).resolve()
        if not path.is_relative_to(PHOTO_ROOT.resolve()) or not path.name.endswith("_raw.jpg"):
            raise ValueError("signs_images 안의 원본 사진(_raw.jpg)만 재채점할 수 있습니다.")
        if not path.is_file():
            raise FileNotFoundError(path)
        photos.append(path)
    if len({r["사진번호"] for r in rows}) != 30 or len(set(photos)) != 30:
        raise ValueError("사진번호와 원본 사진은 30장 모두 달라야 합니다.")
    baseline = assess(rows)
    detector = make_detector(size)
    session = (source.stem + f"_rescore_{size}_"
               + datetime.now(KST).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:6])
    photo_dir = PHOTO_ROOT / session
    table_path = ROOT / "results" / f"{session}.csv"
    comparison_path = ROOT / "results" / f"{session}_comparison.csv"
    summary_path = ROOT / "results" / f"{session}_summary.csv"
    photo_dir.mkdir(parents=True, exist_ok=False)
    rescored, comparison = [], []
    note = "거리는 자로 잰 대략값"
    new_fields = (*FIELDS, "모델", "거리측정", "원본CSV")
    comparison_fields = ("사진번호", "정답", "거리_cm", "각도", "원본사진", "거리측정",
                         "320_인식이름", "320_점수", "320_맞힘", "재채점모델",
                         "재채점_인식이름", "재채점_점수", "재채점_맞힘", "재채점_결과사진")
    for row, raw_path in zip(rows, photos):
        frame = cv2.imread(str(raw_path))
        if frame is None:
            raise RuntimeError(f"원본 사진을 읽지 못했습니다: {raw_path}")
        found = accepted(detector(frame))
        marked_path = photo_dir / f"photo_{row['사진번호']:02d}_result.jpg"
        if not cv2.imwrite(str(marked_path), annotated(frame, found)):
            raise RuntimeError(f"결과 사진 저장 실패: {marked_path}")
        new = {field: row[field] for field in FIELDS}
        new.update(인식이름=";".join(name for name, _, _ in found),
                   점수=";".join(f"{score:.6f}" for _, score, _ in found),
                   맞힘=int(any(name == row["정답"] for name, _, _ in found)),
                   결과사진=str(marked_path.relative_to(ROOT)),
                   모델=f"yolo26n_{size}", 거리측정=note, 원본CSV=str(source))
        rescored.append(new)
        paired = {key: row[key] for key in comparison_fields[:5]}
        paired.update(거리측정=note, 재채점모델=f"yolo26n_{size}")
        for field in ("인식이름", "점수", "맞힘"):
            paired[f"320_{field}"] = row[field]
            paired[f"재채점_{field}"] = new[field]
        paired["재채점_결과사진"] = new["결과사진"]
        comparison.append(paired)
        write_table(table_path, rescored, new_fields)
        write_table(comparison_path, comparison, comparison_fields)
        print(f"재채점 {len(rescored)}/30: {new['정답']} {'맞힘' if new['맞힘'] else '못 맞힘'}", flush=True)
    result = assess(rescored)
    summary_fields = ("모델", *TARGETS, "전체맞힘", "전체장수", "촬영조건충족", "통과", "거리측정", "원본CSV")
    summary_rows = []
    for model, assessment in (("yolo26n_320 (원래 결과)", baseline), (f"yolo26n_{size} (저장 사진 재채점)", result)):
        summary_rows.append({"모델": model, **assessment["by_class"],
                             "전체맞힘": assessment["correct"], "전체장수": assessment["complete"],
                             "촬영조건충족": assessment["plan_ok"], "통과": assessment["passed"],
                             "거리측정": note, "원본CSV": str(source)})
    write_table(summary_path, summary_rows, summary_fields)
    print(f"320: {baseline['correct']}/30 → {size}: {result['correct']}/30\n"
          f"종류별: {result['by_class']}\n촬영 조건: {result['plan_ok']}, 통과: {result['passed']}\n"
          f"{note}\n새 기록표: {table_path}\n사진별 비교: {comparison_path}\n합계 비교: {summary_path}", flush=True)
    return {"baseline": baseline, "result": result, "table": table_path,
            "comparison": comparison_path, "summary": summary_path, "photo_dir": photo_dir}


def main():
    parser = argparse.ArgumentParser(description="인쇄 표지 30장 촬영·YOLO 인식 시험, 바퀴는 움직이지 않음")
    parser.add_argument("--rescore", type=Path, help="기존 사진 폴더 또는 CSV로 재촬영 없이 채점")
    parser.add_argument("--size", type=int, choices=(320, 640), default=320, help="이미 있는 모델 크기 (기본 320)")
    args = parser.parse_args()
    if args.rescore is not None:
        rescore(args.rescore, args.size)
        return
    if args.size != 320:
        parser.error("640 모델 비교는 --rescore로 기존 30장을 지정해 주세요.")
    session = datetime.now(KST).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:6]
    photo_dir = PHOTO_ROOT / session
    table_path = ROOT / "results" / f"signs_{session}.csv"
    video_path = VIDEO_ROOT / f"signs_{session}.mp4"
    camera = LatestCamera()
    worker = video = None
    try:
        camera.start()
        worker = InferenceWorker(camera, make_detector())
        worker.start()
        video = ResultVideo(worker, video_path)
        video.start()
        print(f"사진 폴더: {photo_dir}\n기록표: {table_path}\n결과 영상: {video_path}", flush=True)
        print("인쇄 표지 준비: stop sign 9cm, person 전신 18cm, car 12cm. 각도는 거리마다 섞어 주세요.", flush=True)
        result = collect_photos(camera, worker, video, photo_dir, table_path)
        print(f"종류별 맞힘: {result['by_class']}", flush=True)
        print(f"촬영 조건 {'충족' if result['plan_ok'] else '미충족'}, {result['correct']}/30, "
              f"{'통과' if result['passed'] else '미통과'}", flush=True)
    except KeyboardInterrupt:
        print("촬영을 중단합니다. 입력을 마친 기록은 보존합니다.", flush=True)
    finally:
        errors = []
        for resource in (video, worker, camera):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(exc)
                    print(f"정리 오류: {exc}", flush=True)
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError("종료 정리 실패") from errors[0]
        if video is not None:
            print(f"영상 저장: {video.frames}장, {video.frames / FPS:.2f}초", flush=True)


if __name__ == "__main__":
    main()
