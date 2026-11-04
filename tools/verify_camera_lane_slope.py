#!/usr/bin/env python3
# 지시서: specs/07-camera-lane.md
"""수정 전 소스 경로를 받아 CSV 참고 계산과 원본 사진 재검출을 비교한다."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / 'results/2026-11-04-camera-lane-slope'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    source = path.read_text().replace("ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(ROOT)!r})")
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def digest(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def single(lines):
    if (lines['left'] is None) == (lines['right'] is None):
        return None
    return 'left' if lines['left'] is not None else 'right'


def metrics(rows, prefix):
    names = [r[prefix+'single'] for r in rows]
    wheels = [json.loads(r[prefix+'wheels']) for r in rows]
    return dict(rows=len(rows), single_rows=sum(n is not None for n in names),
                single_name_changes=sum(a is not None and b is not None and a != b
                                        for a, b in zip(names, names[1:])),
                turn_ge70_rows=sum(abs(w[0]-w[2]) >= 70 for w in wheels),
                difference_jump_gt10_rows=sum(abs((a[0]-a[2])-(b[0]-b[2])) > 10
                                             for a,b in zip(wheels,wheels[1:]) if any(b)))


def save(name, rows):
    with Path(str(PREFIX)+name+'.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def photo_replay(before, after, folder):
    paths = sorted(folder.glob('*.jpg'))
    original = digest(paths)
    controllers = [before.LaneController(), after.LaneController()]
    rows = []
    for index,path in enumerate(paths):
        frame = cv2.imread(str(path))
        row = dict(session=folder.name, file=path.name)
        for prefix,mod,c in zip(('before_','after_'),(before,after),controllers):
            # 전체 사진의 검출·조향 진단. 정지 시간이나 실제 주행 재현이 아니다.
            c.stopped = False
            c.last_frame = None
            c.lost_since = None
            if mod is after:
                lines = mod.find_lines(frame,c.previous,c.width,c.single_name)
            else:
                lines = mod.find_lines(frame,c.previous,c.width)
            c.update(lines,index/30)
            row.update({prefix+'single':single(lines),prefix+'left':lines['left'],
                        prefix+'right':lines['right'],prefix+'width':c.width,
                        prefix+'center':c.center,prefix+'state':c.state,
                        prefix+'wheels':json.dumps(c.speeds)})
        rows.append(row)
    save('-'+folder.name,rows)
    return dict(session=folder.name, before=metrics(rows,'before_'),
                after=metrics(rows,'after_'), original_sha256=original,
                originals_unchanged=original==digest(paths))


def csv_replay(after, path):
    original = digest([path])
    rows = []
    previous = after.STOP
    for source in csv.DictReader(path.open()):
        old = tuple(json.loads(source['바퀴값']))
        # 원본 선 판정/중심/정지 순서는 유지. 속도차 제한 효과만 계산.
        new = after.STOP if not any(old) else after.limit_difference(old,previous)
        previous = new
        name = single(dict(left=source['왼쪽선_x'] or None,
                           right=source['오른쪽선_x'] or None))
        rows.append(dict(seconds=source['경과초'],sequence=source['화면순번'],
                         state=source['상태'],before_single=name,after_single=name,
                         before_wheels=json.dumps(old),after_wheels=json.dumps(new)))
    save('-'+path.stem,rows)
    old_stats = metrics(rows,'before_')
    new_stats = metrics(rows,'after_')
    new_stats['single_name_changes'] = None
    return dict(source=path.name,before=old_stats,speed_limit_only=new_stats,
                originals_unchanged=original==digest([path]),original_sha256=original,
                limitation='기울기·최하단 픽셀 없음. 수정 후 이름 변경 횟수는 계산 불가. after_single은 원본 이름 복사이며 새 검출이 아님.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before',type=Path,required=True)
    args = parser.parse_args()
    before = load('lane_slope_before',args.before)
    after = load('lane_slope_after',ROOT/'missions/07_camera_lane.py')
    report = dict(before_code_sha256=hashlib.sha256(args.before.read_bytes()).hexdigest(),
                  after_code_sha256=hashlib.sha256((ROOT/'missions/07_camera_lane.py').read_bytes()).hexdigest(),
                  definitions='single_name_changes: 인접 단일선 행 이름 변경, 양선/미검출/미배정에서 연결 끊음. turn_ge70_rows: abs(L-R)>=70. 속도차 변화>10은 즉시정지 행 제외.',
                  csv=[csv_replay(after,ROOT/'results'/name) for name in
                       ('camera_lane_20261104_184432_530537.csv','camera_lane_20261104_183948_623953.csv')],
                  photos=[photo_replay(before,after,Path('/home/pi/train_raw')/name)
                          for name in ('slow_174248','slow2_174610')],
                  limitation='사진은 별도 세션·정답 없음. 정지 래치와 시각 조건을 해제한 전체 프레임 진단이며 실제 주행/정지시간 검증 아님. 새 움직임에 따른 영상 변화 재현 불가.',
                  hardware_used=False,images_copied_or_saved=False)
    Path(str(PREFIX)+'-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
