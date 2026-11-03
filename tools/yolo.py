#!/usr/bin/env python3
# 카메라로 물체 알아보기 — YOLO26n 모델(models/)을 OpenCV 로 돌린다. 새로 설치할 것 없음.
#   점검:   python3 tools/yolo.py          → 카메라 앞에 보이는 것을 알려 주고 사진을 results/yolo_test.jpg 로 저장
#   부품:   missions/ 프로그램에서 Detector 를 불러 쓴다 (AGENTS.md 「카메라로 알아보기」 참고)
# 모델: Ultralytics YOLO26n (AGPL-3.0) · 학습 데이터 COCO 80가지 물체 · 자세한 출처는 models/README.md
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")


class Detector:
    """사진 한 장을 넣으면 [(이름, 점수, (x, y, 너비, 높이)), ...] 를 돌려준다.

    size 320 = 빠름(라즈봇에서 1초에 약 16번) · 640 = 멀리 있는 것을 더 잘 봄(1초에 약 4번)
    threads 2 = CPU 4개 중 2개만 써서 바퀴 · 센서 · 녹화 몫을 남긴다
    """

    def __init__(self, size=320, conf=0.4, iou=0.45, threads=2):
        cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNetFromONNX(os.path.join(MODELS, f"yolo26n_{size}.onnx"))
        with open(os.path.join(MODELS, "coco.txt"), encoding="utf-8") as f:
            self.names = [line.strip() for line in f]
        self.size, self.conf, self.iou = size, conf, iou

    def __call__(self, img):
        h, w = img.shape[:2]
        r = self.size / max(h, w)  # 비율을 지킨 채 줄여서 정사각형 왼쪽 위에 얹는다
        nh, nw = round(h * r), round(w * r)
        square = np.full((self.size, self.size, 3), 114, np.uint8)
        square[:nh, :nw] = cv2.resize(img, (nw, nh))
        self.net.setInput(cv2.dnn.blobFromImage(square, 1 / 255.0, swapRB=True))
        out = self.net.forward()[0].T  # 후보마다: 가운데 x · y · 너비 · 높이 + 물체 80가지 점수
        scores = out[:, 4:]
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        keep = best > self.conf
        out, cls, best = out[keep], cls[keep], best[keep]
        boxes = [[(cx - bw / 2) / r, (cy - bh / 2) / r, bw / r, bh / r] for cx, cy, bw, bh in out[:, :4]]
        picked = cv2.dnn.NMSBoxes(boxes, best.tolist(), self.conf, self.iou)  # 겹친 상자는 하나만 남김
        found = [(self.names[cls[i]], float(best[i]), tuple(int(v) for v in boxes[i])) for i in np.array(picked).flatten()]
        return sorted(found, key=lambda d: -d[1])


def draw(img, found):
    """상자와 이름을 그린다 (글자는 영어·숫자만 — OpenCV 는 한글을 못 씀)."""
    for name, score, (x, y, bw, bh) in found:
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(img, f"{name} {score:.2f}", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def open_camera():
    """USB 카메라를 640x480 · 초당 30장(MJPG)으로 연다."""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


if __name__ == "__main__":
    print("[1/3] 모델 불러오기: models/yolo26n_320.onnx")
    det = Detector()
    print("[2/3] 카메라 앞에 보이는 것 알아보기 (3초) — 사람이 앞에 서거나 STOP 표지 사진을 보여 주세요")
    cap = open_camera()
    frame = None
    for _ in range(10):  # 켜자마자 몇 장은 어둡게 나와서 버린다
        ok, frame = cap.read()
    if frame is None:
        print("   카메라를 못 열었습니다 → 카메라 USB 선을 확인하세요 (ls /dev/video*)")
        raise SystemExit(1)
    count, seen, last, end = 0, {}, [], time.monotonic() + 3
    start = time.monotonic()
    while time.monotonic() < end:
        ok, frame = cap.read()
        if not ok:
            continue
        last = det(frame)
        count += 1
        for name, _, _ in last:
            seen[name] = seen.get(name, 0) + 1
    cap.release()
    print(f"   1초에 {count / (time.monotonic() - start):.1f}번 알아봤습니다")
    print("   본 것:", ", ".join(f"{k} ({v}번)" for k, v in sorted(seen.items(), key=lambda x: -x[1])) or "없음")
    path = os.path.join(ROOT, "results", "yolo_test.jpg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, draw(frame, last))
    print(f"[3/3] 마지막 화면을 상자와 함께 저장: {path}")
    print("[OK] YOLO 점검 끝")
