#!/usr/bin/env python3
# 지시서: specs/07-camera-lane.md
"""저장 파일만 읽는 검증. 카메라·드라이버를 만들지 않고 사진도 저장하지 않는다."""
import csv
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lane_saved', ROOT/'missions/07_camera_lane.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def digest(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_photos(folder):
    paths = sorted(folder.glob('*.jpg'))
    if not paths:
        raise RuntimeError(f'사진 없음: {folder}')
    before = digest(paths)
    c = m.LaneController()
    rows = []
    previous = None
    width = None
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        parts = m.find_components(frame)
        lines = m.assign_lines(parts, previous, width)
        # 사진 촬영기는 최소0.3초 간격이다. 실제 타임스탬프가 없으므로 시간 정지 성능 판정은 하지 않는다.
        c.update(lines, index*.3)
        center = None
        if lines['left'] is not None and lines['right'] is not None:
            center = (lines['left']+lines['right'])/2
        elif width is not None:
            if lines['left'] is not None: center = lines['left']+width/2
            elif lines['right'] is not None: center = lines['right']-width/2
        if lines['left'] is not None and lines['right'] is not None:
            width = abs(lines['right']-lines['left'])
            if previous is None:
                previous = {}
        if previous is not None:
            for name in ('left','right'):
                if lines[name] is not None: previous[name] = lines[name]
        rows.append(dict(session=folder.name, file=path.name, components=json.dumps(parts),
                         left=lines['left'], right=lines['right'], left_count=lines['left_count'],
                         right_count=lines['right_count'], unassigned=lines.get('unassigned',False),
                         width=width, center=center,
                         calculated_wheels=json.dumps(m.wheel_speeds(center)) if center is not None else '',
                         controller_state=c.state, controller_wheels=json.dumps(c.speeds),
                         latched_stop=c.stopped))
    summary = dict(folder=str(folder), photos=len(paths), original_sha256=before,
                   originals_unchanged=before==digest(paths),
                   detected_components=dict(Counter(len(json.loads(r['components'])) for r in rows)),
                   assigned_left_over_320=sum(r['left'] is not None and r['left']>320 for r in rows),
                   assigned_right_under_320=sum(r['right'] is not None and r['right']<320 for r in rows),
                   merged_frames=sum(len(json.loads(r['components']))==2 and (r['left'] is None or r['right'] is None) for r in rows),
                   states=dict(Counter(r['controller_state'] for r in rows)),
                   note='정지 후에도 별도 진단 추적은 계속함. controller 값은 정지 유지. 실제 주행/정답 정확도/정지 시간 판정 아님. 최소0.3초 간격, 실제 시각 없음.')
    return rows, summary


def csv_proxy():
    source = ROOT/'results/camera_lane_20261104_173951_358798.csv'
    before = digest([source])
    c = m.LaneController()
    output = []
    for row in csv.DictReader(source.open()):
        # 반쪽 평균은 연결 덩어리가 아니다. 탈락 점 위치는 복원하지 않는다.
        parts = [(float(row[name+'선_x']), int(row[name+'노란점개수']))
                 for name in ('왼쪽','오른쪽') if row[name+'선_x']]
        lines = m.assign_lines(parts, c.previous, c.width)
        c.update(lines, float(row['경과초']))
        if 12.2 <= float(row['경과초']) <= 12.5:
            output.append(dict(seconds=row['경과초'], old_left=row['왼쪽선_x'],old_right=row['오른쪽선_x'],
                               old_counts=row['왼쪽노란점개수']+'/'+row['오른쪽노란점개수'],
                               old_center=row['차선중심_x'],old_wheels=row['바퀴값'],
                               proxy_left=lines['left'],proxy_right=lines['right'],proxy_width=c.width,
                               proxy_center=c.center,proxy_wheels=json.dumps(c.speeds),proxy_state=c.state,
                               limitation='반쪽 평균을 후보로 사용한 참고 계산. 실제 연결 덩어리 재검출 아님.'))
    return output, dict(source=str(source), original_unchanged=before==digest([source]),
                        exact_component_replay=False,
                        reason='CSV에 원본 픽셀 없음. 별도174248/174610 사진은173951 시험의 동일 화면이 아님.')


def main():
    summaries=[]; rows=[]
    for name in ('slow_174248','slow2_174610'):
        part, summary = raw_photos(Path('/home/pi/train_raw')/name)
        rows.extend(part); summaries.append(summary)
    output=ROOT/'results'
    save_csv(output/'2026-11-04-camera-lane-components-photos.csv', rows)
    proxy, proxy_summary = csv_proxy()
    save_csv(output/'2026-11-04-camera-lane-components-csv-proxy.csv', proxy)
    report = dict(raw_sessions=summaries, csv_proxy=proxy_summary, camera_used=False,
                  motors_used=False, images_copied_or_saved=False,
                  accuracy='선 정답 라벨 없음. 검출·계산 확인이며 실제 코너 통과 검증 아님.')
    (output/'2026-11-04-camera-lane-components-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
