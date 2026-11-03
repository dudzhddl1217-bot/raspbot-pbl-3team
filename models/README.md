# models — 카메라로 물체를 알아보는 YOLO 모델

강사가 넣어 둔 파일입니다. 바꾸거나 지우지 마세요. 새로 설치할 것은 없습니다(라즈봇의 OpenCV 로 돎).

| 파일 | 무엇 | 라즈봇에서 (CPU 2개) |
|---|---|---|
| `yolo26n_320.onnx` | **기본** · 빠름 | 1초에 약 16번 |
| `yolo26n_640.onnx` | 멀리 있는 작은 것을 더 잘 봄 · 느림 | 1초에 약 4번 |
| `coco.txt` | 알아보는 물체 80가지 이름 (0번 `person`, 11번 `stop sign` …) | — |

- 쓰는 법: `python3 tools/yolo.py` (점검) · 프로그램에서는 `tools/yolo.py` 의 `Detector` (AGENTS.md 「카메라로 알아보기」)
- 자율주행에 쓸 만한 것: `stop sign` · `person` · `traffic light` · `car` · `bicycle` · `motorcycle` · `bus` · `truck`
- 알아보는 것은 「무엇이 있다」까지입니다. 신호등의 불 색, 사람이 누구인지는 모릅니다. 80가지에 없는 모양(직접 그린 새 표지)은 못 찾습니다.

## 출처 · 라이선스
- 모델: Ultralytics **YOLO26n** — https://huggingface.co/Ultralytics/YOLO26 의 `yolo26n.pt` (sha256 `9b09cc8bf347f0fc…`)
- 학습 데이터: COCO (일상 사진 약 12만 장 · 물체 80가지)
- 라이선스: **AGPL-3.0** — https://www.ultralytics.com/license · 수업과 비공개 레포에서 자유롭게 사용. 이 모델로 만든 서비스를 밖에 공개하려면 라이선스 조건을 먼저 확인.
- 변환: `ultralytics` 8.4.199 로 `export(format="onnx", imgsz=320 또는 640, opset=12, simplify=True)` — 2026-11-03 · 강사 라즈봇에서 확인
