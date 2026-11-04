#!/usr/bin/env python3
# 지시서: specs/05-signs.md
"""실제 카메라·모델 추론·모터 없이 촬영 절차와 채점·스레드를 확인한다."""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("signs", Path(__file__).resolve().parents[1] / "missions/05_signs.py")
signs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signs)


def plan_rows(correct=30):
    rows=[]
    for name in signs.TARGETS:
        for distance, quota in signs.QUOTAS.items():
            for i in range(quota):
                rows.append({"정답":name,"거리_cm":distance,"각도":signs.ANGLES[i%3],
                             "맞힘":int(len(rows)<correct)})
    return rows


class FixedCamera:
    def __init__(self):
        self.frame=np.full((64,64,3),7,dtype=np.uint8)

    def snapshot(self):
        return self.frame.copy(),1


class SignsTests(unittest.TestCase):
    def test_confidence_inclusive_and_target_filter(self):
        found=[('stop sign',.4,(1,2,3,4)),('person',.39999,(1,2,3,4)),
               ('car',.9,(1,2,3,4)),('bus',.99,(1,2,3,4)),('person',float('nan'),(1,2,3,4))]
        self.assertEqual([r[0] for r in signs.accepted(found)],['stop sign','car'])

    def test_detector_uses_existing_320_and_includes_boundary(self):
        with patch.object(signs,'Detector') as factory:
            signs.make_detector()
        args=factory.call_args.kwargs
        self.assertEqual(args['size'],320)
        self.assertLess(args['conf'],.4)
        self.assertTrue((np.array([.4],dtype=np.float32)>args['conf'])[0])

    def test_24_pass_and_23_fail(self):
        self.assertTrue(signs.assess(plan_rows(24))['passed'])
        self.assertFalse(signs.assess(plan_rows(23))['passed'])
        self.assertEqual(signs.assess(plan_rows())['by_class'],{name:10 for name in signs.TARGETS})

    def test_wrong_quota_or_angle_is_reported(self):
        rows=plan_rows()
        rows[0]['거리_cm']=50
        self.assertFalse(signs.assess(rows)['plan_ok'])
        rows=plan_rows()
        for r in rows: r['각도']='정면'
        self.assertFalse(signs.assess(rows)['plan_ok'])

    def test_partial_table_not_passed(self):
        rows=plan_rows()[:24]
        self.assertFalse(signs.assess(rows)['passed'])
        rows.append({'맞힘':'','정답':'','거리_cm':'','각도':''})
        self.assertEqual(signs.assess(rows)['complete'],24)

    def test_metadata_validation(self):
        answers=iter(['틀린 값','2','20','50','9','3'])
        with redirect_stdout(io.StringIO()):
            result=signs.ask_metadata(lambda _:next(answers))
        self.assertEqual(result,('person',50,'오른쪽'))

    def test_annotation_does_not_modify_shared_frame(self):
        frame=np.zeros((64,64,3),dtype=np.uint8)
        result=signs.annotated(frame,[('car',.9,(2,2,40,40))])
        self.assertFalse(frame.any())
        self.assertTrue(result.any())

    def test_worker_analyzes_exact_photo_on_own_thread(self):
        camera=FixedCamera()
        calls=[]
        def detector(frame):
            calls.append((int(frame[0,0,0]),threading.get_ident()))
            return [('car',.4,(1,1,20,20))]
        worker=signs.InferenceWorker(camera,detector)
        try:
            worker.start()
            photo=np.full((64,64,3),42,dtype=np.uint8)
            found=worker.analyze_photo(photo,42)
            self.assertEqual(found[0][0],'car')
            self.assertTrue(any(value==42 for value,_ in calls))
            self.assertTrue(all(tid!=threading.get_ident() for _,tid in calls))
            frame,_,_=worker.snapshot()
            frame[:]=255
            frame2,_,_=worker.snapshot()
            self.assertFalse(np.all(frame2==255))
        finally:
            worker.close()
        self.assertFalse(worker._thread.is_alive())

    def test_worker_reports_inference_failure(self):
        def fail(_): raise ValueError('가짜 인식 실패')
        worker=signs.InferenceWorker(FixedCamera(),fail)
        try:
            with self.assertRaises(RuntimeError): worker.start()
        finally:
            worker.close()

    def test_video_15fps_repeats_and_releases(self):
        class Source:
            def snapshot(self): return np.zeros((64,64,3),dtype=np.uint8),[],1
        class Writer:
            def __init__(self): self.count=0; self.released=False
            def isOpened(self): return True
            def write(self,frame):
                self.count+=1
                if self.count==3: video._stop.set()
            def release(self): self.released=True
        signs.VIDEO_ROOT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=signs.VIDEO_ROOT) as tmp:
            video=signs.ResultVideo(Source(),Path(tmp)/'test.mp4')
            writer=Writer()
            with patch.object(signs.cv2,'VideoWriter',return_value=writer) as factory:
                try:
                    video.start()
                    video._thread.join(2)
                finally:
                    video.close()
            self.assertEqual(factory.call_args.args[2],15)
            self.assertEqual(video.frames,3)
            self.assertTrue(writer.released)

    def test_enter_capture_then_metadata_and_30_rows(self):
        plan=plan_rows()
        answers=[]
        for row in plan:
            answers.extend(['',str(signs.TARGETS.index(row['정답'])+1),str(row['거리_cm']),
                            str(signs.ANGLES.index(row['각도'])+1)])
        consumed=[]
        iterator=iter(answers)
        def input_fn(prompt):
            result=next(iterator)
            consumed.append(result)
            return result
        class Camera:
            count=0
            def snapshot(self):
                self.count+=1
                # 각 사진은 엔터 뒤, 정답 입력 전에 찍혀야 한다.
                assert len(consumed)==(self.count-1)*4+1
                return np.full((64,64,3),self.count,dtype=np.uint8),self.count
        class Worker:
            def check(self): pass
            def analyze_photo(self,frame,sequence):
                assert int(frame[0,0,0])==sequence
                return [(plan[sequence-1]['정답'],.4,(2,2,40,40))]
        class Video:
            def check(self): pass
        signs.PHOTO_ROOT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=signs.PHOTO_ROOT) as tmp:
            photo_dir=Path(tmp)/'photos'
            table=Path(tmp)/'table.csv'
            camera=Camera()
            with redirect_stdout(io.StringIO()):
                result=signs.collect_photos(camera,Worker(),Video(),photo_dir,table,input_fn)
            self.assertTrue(result['passed'])
            self.assertEqual(camera.count,30)
            self.assertEqual(len(list(photo_dir.glob('*_raw.jpg'))),30)
            self.assertEqual(len(list(photo_dir.glob('*_result.jpg'))),30)
            with table.open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),30)
            self.assertTrue(all(r['점수']=='0.400000' for r in rows))
            self.assertTrue(all(r['맞힘']=='1' for r in rows))
            first=signs.cv2.imread(str(photo_dir/'photo_01_raw.jpg'))
            self.assertTrue(np.all(first==1))

    def test_interrupt_keeps_pending_photo_in_table(self):
        class Worker:
            def check(self): pass
            def analyze_photo(self,*_): return []
        class Video:
            def check(self): pass
        answers=iter([''])
        def input_fn(prompt):
            try: return next(answers)
            except StopIteration: raise KeyboardInterrupt()
        signs.PHOTO_ROOT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=signs.PHOTO_ROOT) as tmp:
            table=Path(tmp)/'partial.csv'
            with redirect_stdout(io.StringIO()),self.assertRaises(KeyboardInterrupt):
                signs.collect_photos(FixedCamera(),Worker(),Video(),Path(tmp)/'photos',table,input_fn)
            with table.open() as stream: rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['맞힘'],'')

    def test_main_closes_all_resources_on_interrupt(self):
        closed=[]
        class Resource:
            frames=1
            def __init__(self,name): self.name=name
            def start(self): return self
            def close(self): closed.append(self.name)
        with patch.object(signs,'LatestCamera',return_value=Resource('카메라')), \
             patch.object(signs,'make_detector'), \
             patch.object(signs,'InferenceWorker',return_value=Resource('인식')), \
             patch.object(signs,'ResultVideo',return_value=Resource('영상')), \
             patch.object(signs,'collect_photos',side_effect=KeyboardInterrupt()), \
             patch.object(signs.sys,'argv',['05_signs.py']), redirect_stdout(io.StringIO()):
            signs.main()
        self.assertEqual(closed,['영상','인식','카메라'])

    def test_rescore_saved_photos_preserves_inputs_and_pairs_results(self):
        signs.PHOTO_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=signs.PHOTO_ROOT) as tmp:
            photo_dir = Path(tmp)
            source = photo_dir / 'original.csv'
            rows = plan_rows(18)
            for i, row in enumerate(rows, 1):
                raw = photo_dir / f'photo_{i:02d}_raw.jpg'
                old_result = photo_dir / f'photo_{i:02d}_result.jpg'
                signs.cv2.imwrite(str(raw), np.full((64, 64, 3), i, np.uint8))
                old_result.write_bytes(b'original result')
                row.update(사진번호=i, 인식이름='', 점수='',
                           원본사진=str(raw.relative_to(signs.ROOT)),
                           결과사진=str(old_result.relative_to(signs.ROOT)))
            signs.write_table(source, rows)
            original_bytes = {p: p.read_bytes() for p in photo_dir.iterdir()}

            def detector(frame):
                i = int(frame[0, 0, 0])
                return [(rows[i - 1]['정답'], .4, (2, 2, 40, 40))]

            with patch.object(signs, 'make_detector', return_value=detector) as factory, \
                 patch.object(signs, 'LatestCamera', side_effect=AssertionError('camera opened')), \
                 patch.object(signs, 'ResultVideo', side_effect=AssertionError('video opened')), \
                 redirect_stdout(io.StringIO()):
                output = signs.rescore(source, 640)
            self.addCleanup(output['table'].unlink)
            self.addCleanup(output['comparison'].unlink)
            self.addCleanup(output['summary'].unlink)
            import shutil
            self.addCleanup(shutil.rmtree, output['photo_dir'])
            factory.assert_called_once_with(640)
            self.assertEqual(output['baseline']['correct'], 18)
            self.assertEqual(output['result']['correct'], 30)
            self.assertTrue(output['result']['passed'])
            for path, content in original_bytes.items():
                self.assertEqual(path.read_bytes(), content)
            with output['comparison'].open() as stream:
                paired = list(csv.DictReader(stream))
            self.assertEqual(len(paired), 30)
            self.assertEqual(sum(int(r['320_맞힘']) for r in paired), 18)
            self.assertEqual(sum(int(r['재채점_맞힘']) for r in paired), 30)
            self.assertTrue(all(r['거리측정'] == '거리는 자로 잰 대략값' for r in paired))
            self.assertEqual(len(list(output['photo_dir'].glob('*_result.jpg'))), 30)

    def test_rescore_incomplete_source_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory(dir=signs.PHOTO_ROOT) as tmp:
            source = Path(tmp) / 'incomplete.csv'
            signs.write_table(source, [])
            with patch.object(signs, 'make_detector') as factory:
                with self.assertRaises(ValueError):
                    signs.rescore(source, 640)
            factory.assert_not_called()

    def test_rescore_cli_returns_before_camera_construction(self):
        with patch.object(signs, 'rescore') as run, \
             patch.object(signs, 'LatestCamera', side_effect=AssertionError('camera opened')), \
             patch.object(signs.sys, 'argv', ['05_signs.py', '--rescore', 'results/signs_images/example', '--size', '640']):
            signs.main()
        run.assert_called_once_with(Path('results/signs_images/example'), 640)


if __name__=='__main__':
    unittest.main()
