#!/usr/bin/env python3
# 지시서: specs/06-drive-signs.md
"""실제 카메라·I2C·모터 없이 상태와 정지·녹화·기록을 검증한다."""
from contextlib import contextmanager
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('drive_signs', ROOT / 'missions/06_drive_signs.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
MIDDLE = (0, 1, 0, 1)
WHITE = (1, 1, 1, 1)
LEFT = (0, 1, 1, 1)
FORWARD = (50, 50, 50, 50)


def found(*names):
    return [(name, 0.4, (10, 20, 30, 80)) for name in names]


def packet(sequence, now, *names):
    return dict(sequence=sequence, captured_at=now, completed_at=now, found=found(*names))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.c = m.SignController()
        self.seq = 0

    def step(self, now, *names, bits=MIDDLE):
        self.seq += 1
        command = self.c.step(bits, packet(self.seq, now, *names), now)
        self.c.motor_sent(command, now)
        return command

    def test_initial_result_and_stale_boundary(self):
        self.assertEqual(self.c.step(MIDDLE, None, 0), m.STOP)
        self.assertEqual(self.c.state, 'WAITVISION')
        self.assertEqual(self.step(0), FORWARD)
        p = packet(self.seq, 0)
        self.assertEqual(self.c.step(MIDDLE, p, 1), FORWARD)
        self.assertEqual(self.c.step(MIDDLE, p, 1.001), m.STOP)
        self.assertEqual(self.c.state, 'STALEVISION')
        self.assertEqual(self.step(1.01), FORWARD)

    def test_same_result_does_not_confirm(self):
        p = packet(1, 0, 'stop sign')
        for now in (0, .1, .2, .5):
            self.assertEqual(self.c.step(MIDDLE, p, now), FORWARD)
        self.assertEqual(self.step(.6, 'stop sign'), FORWARD)  # 같은 순번 1
        self.assertEqual(self.step(.7, 'stop sign'), m.STOP)

    def test_stop_three_seconds_and_send_timestamp(self):
        self.assertEqual(self.step(0, 'stop sign'), FORWARD)
        self.seq += 1
        self.assertEqual(self.c.step(MIDDLE, packet(self.seq, .2, 'stop sign'), .2), m.STOP)
        self.assertIsNone(self.c.stop_until)
        self.c.motor_sent(m.STOP, .3)
        self.assertEqual(self.c.stop_until, 3.3)
        self.assertEqual(self.step(3.299, 'stop sign'), m.STOP)
        self.assertEqual(self.step(3.3, 'stop sign'), FORWARD)
        self.assertEqual(self.c.ignore_until, 8.3)

    def test_persistent_stop_not_retriggered(self):
        self.step(0, 'stop sign'); self.step(.25, 'stop sign')
        self.step(3.25, 'stop sign')
        for now in (8.25, 8.5, 9, 12):
            self.assertEqual(self.step(now, 'stop sign'), FORWARD)

    def test_stop_absence_one_second_and_five_second_cooldown(self):
        self.step(0, 'stop sign'); self.step(.25, 'stop sign'); self.step(3.25, 'stop sign')
        self.step(3.5); self.step(4.5)
        self.assertTrue(self.c.stop_armed)
        self.assertEqual(self.step(4.75, 'stop sign'), FORWARD)
        self.assertEqual(self.step(5, 'stop sign'), FORWARD)
        self.assertEqual(self.step(8.249, 'stop sign'), FORWARD)
        self.assertEqual(self.step(8.25, 'stop sign'), m.STOP)

    def test_short_stop_absence_not_rearm(self):
        self.step(0, 'stop sign'); self.step(.25, 'stop sign'); self.step(3.25, 'stop sign')
        self.step(4); self.step(4.9)
        self.step(5, 'stop sign')
        self.assertEqual(self.step(9, 'stop sign'), FORWARD)
        self.assertFalse(self.c.stop_armed)

    def test_nonconsecutive_detection_resets(self):
        self.step(0, 'person'); self.step(.25)
        self.assertEqual(self.step(.5, 'person'), FORWARD)
        self.assertEqual(self.step(.75, 'person'), m.STOP)

    def test_person_release_needs_new_absent_results(self):
        self.step(0, 'person'); self.step(.25, 'person')
        self.assertEqual(self.step(.5), m.STOP)
        self.assertEqual(self.step(1.499), m.STOP)
        self.assertEqual(self.step(1.5), FORWARD)

    def test_person_reappears_resets_absence(self):
        self.step(0, 'person'); self.step(.25, 'person')
        self.step(.5); self.step(1, 'person'); self.step(1.25)
        self.assertEqual(self.step(2), m.STOP)
        self.assertEqual(self.step(2.25), FORWARD)

    def test_person_priority_stop_clock_continues(self):
        self.step(0, 'person', 'stop sign'); self.step(.25, 'person', 'stop sign')
        self.assertEqual(self.c.state, 'PERSONWAIT')
        self.assertEqual(self.c.stop_until, 3.25)
        self.assertEqual(self.step(4, 'person', 'stop sign'), m.STOP)
        self.assertIsNone(self.c.ignore_until)
        self.step(4.25, 'stop sign')
        self.assertEqual(self.step(5.25, 'stop sign'), FORWARD)
        self.assertEqual(self.c.ignore_until, 10.25)

    def test_person_does_not_pause_existing_cooldown(self):
        self.step(0, 'stop sign'); self.step(.25, 'stop sign'); self.step(3.25, 'stop sign')
        self.step(4, 'person'); self.step(4.25, 'person')
        self.step(9, 'person')
        self.assertEqual(self.c.ignore_until, 8.25)
        self.step(9.25)
        self.assertEqual(self.step(10.25), FORWARD)

    def test_stale_does_not_count_as_absence(self):
        self.step(0, 'person'); self.step(.25, 'person'); self.step(.5)
        self.assertEqual(self.c.step(MIDDLE, packet(self.seq, .5), 2), m.STOP)
        self.assertEqual(self.step(2.1), m.STOP)
        self.assertEqual(self.step(3.099), m.STOP)
        self.assertEqual(self.step(3.1), FORWARD)

    def test_stale_resets_two_result_confirmation(self):
        self.step(0, 'person')
        self.c.step(MIDDLE, packet(self.seq, 0, 'person'), 1.1)
        self.assertEqual(self.step(1.2, 'person'), FORWARD)
        self.assertEqual(self.step(1.4, 'person'), m.STOP)

    def test_line_search_paused_and_direction_retained(self):
        self.step(0, bits=LEFT)
        self.step(.25, 'person', bits=LEFT)
        self.assertEqual(self.step(.5, 'person', bits=WHITE), m.STOP)
        self.assertEqual(self.c.line.last_turn, 'LEFT')
        self.step(10, 'person', bits=WHITE)
        self.step(10.25, bits=WHITE)
        self.assertEqual(self.step(11.25, bits=WHITE), (0, 0, 40, 40))
        self.assertEqual(self.step(12.25, bits=WHITE), (0, 0, 40, 40))
        self.assertEqual(self.step(12.251, bits=WHITE), m.STOP)
        self.assertEqual(self.c.state, 'LOSTSTOP')
        self.assertEqual(self.step(13, bits=MIDDLE), m.STOP)

    def test_no_last_turn_stops_on_resume_without_line(self):
        self.c.step(WHITE, None, 0)
        self.assertEqual(self.step(.1, bits=WHITE), m.STOP)
        self.assertTrue(self.c.line.stopped)

    def test_existing_turn_speeds(self):
        for bits in (MIDDLE, LEFT, (0,0,1,1), (1,0,1,1), (1,1,0,1), (1,1,0,0), (1,1,1,0)):
            expected = m.line.LineFollower().step(bits, 0)
            self.assertEqual(m.SignController().step(bits, packet(1,0),0), expected)

    def test_filter_inclusive_score_and_height(self):
        self.assertEqual(m.targets([('person',.4,(0,0,5,18)), ('car',.9,(0,0,5,9)),
                                    ('stop sign',.399,(0,0,5,9)), ('person',float('nan'),(0,0,5,9))]),
                         [('person',.4,(0,0,5,18))])


class Clock:
    def __init__(self): self.now = 0
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class IntegrationTests(unittest.TestCase):
    def test_loop_stop_on_key_error_interrupt(self):
        for action in ('key', 'sensor', 'worker', 'video', 'interrupt'):
            with self.subTest(action=action):
                clock = Clock(); bot = Mock(); bot.read_data_array.return_value = [5]
                worker = Mock(); worker.snapshot.return_value = packet(1,0)
                recorder = Mock(); count = 0
                def key():
                    nonlocal count
                    count += 1
                    return count > 2 if action == 'key' else False
                if action == 'sensor': bot.read_data_array.side_effect = RuntimeError('가짜 센서 오류')
                if action == 'worker': worker.snapshot.side_effect = RuntimeError('가짜 YOLO 오류')
                if action == 'video': recorder.check.side_effect = RuntimeError('가짜 녹화 오류')
                if action == 'interrupt': worker.snapshot.side_effect = KeyboardInterrupt
                with patch.object(m.time,'monotonic',clock.monotonic), patch.object(m.time,'sleep',clock.sleep):
                    if action in ('key','interrupt'):
                        m.run_loop(bot,worker,recorder,key,io.StringIO())
                    else:
                        with self.assertRaises(RuntimeError):
                            m.run_loop(bot,worker,recorder,key,io.StringIO())
                self.assertEqual([tuple(c.args) for c in bot.Ctrl_Muto.call_args_list[-12:]], [(i,0) for _ in range(3) for i in range(4)])

    def test_loop_waits_for_vision_and_logs_height(self):
        clock=Clock(); bot=Mock(); bot.read_data_array.return_value=[5]
        worker=Mock(); worker.snapshot.side_effect=[None,packet(1,.02,'person'),packet(2,.04,'person')]
        recorder=Mock(); calls=0
        def key():
            nonlocal calls
            calls+=1
            return calls>6
        output=io.StringIO()
        with patch.object(m.time,'monotonic',clock.monotonic), patch.object(m.time,'sleep',clock.sleep):
            m.run_loop(bot,worker,recorder,key,output)
        rows=list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual([r['상태'] for r in rows], ['WAITVISION','FORWARD','PERSONWAIT'])
        self.assertEqual(rows[-1]['상자높이_px'],'80')
        self.assertEqual([c.args for c in bot.Ctrl_Muto.call_args_list[:4]],[(i,0) for i in range(4)])

    def test_worker_uses_thread_and_only_new_camera_sequence(self):
        camera=Mock(); frame=np.zeros((12,16,3),np.uint8)
        camera.snapshot.return_value=(frame,1)
        ids=[]
        def detector(frame):
            ids.append(threading.get_ident())
            return found('person')
        worker=m.VisionWorker(camera,detector)
        try:
            worker.start()
            end=time.monotonic()+1
            while worker.snapshot() is None and time.monotonic()<end:
                time.sleep(.005)
            result=worker.snapshot()
            self.assertIsNotNone(result)
            time.sleep(.03)
            self.assertEqual(len(ids),1)
            self.assertNotEqual(ids[0],threading.get_ident())
            self.assertLessEqual(result['captured_at'],result['completed_at'])
            camera.snapshot.return_value=(frame,2)
            end=time.monotonic()+1
            while worker.snapshot()['sequence']!=2 and time.monotonic()<end:
                time.sleep(.005)
            self.assertEqual(worker.snapshot()['sequence'],2)
        finally: worker.close()

    def test_worker_model640_and_error_propagation(self):
        camera=Mock(); camera.snapshot.side_effect=RuntimeError('가짜 카메라 오류')
        with patch.object(m.signs,'make_detector',return_value=Mock()) as factory:
            worker=m.VisionWorker(camera)
            try:
                worker.start(); worker.thread.join(timeout=1)
                with self.assertRaises(RuntimeError): worker.snapshot()
                factory.assert_called_once_with(640)
            finally: worker.close()

    def test_video_repeats_15fps_and_releases(self):
        clock=Clock(); camera=Mock(); camera.snapshot.return_value=(np.zeros((480,640,3),np.uint8),1)
        recorder=m.DriveVideo(camera); writer=Mock()
        class Event:
            def is_set(self): return recorder.saved_frames>=4
            def wait(self, delay): clock.sleep(delay); return self.is_set()
        recorder.stop=Event()
        with tempfile.TemporaryDirectory() as directory:
            recorder.output_dir=Path(directory)
            with patch.object(m.time,'monotonic',clock.monotonic), patch.object(recorder,'_open_writer',return_value=writer):
                recorder.run()
        self.assertEqual(recorder.saved_frames,4)
        self.assertEqual(recorder.repeated_frames,3)
        self.assertAlmostEqual(clock.now,3/15)
        self.assertEqual(writer.write.call_count,4)
        writer.release.assert_called_once()
        self.assertEqual(int(camera.snapshot()[0].sum()),0)
        self.assertEqual(recorder.segment_frames,900)
        self.assertEqual(recorder.keep,30)

    def test_runs_csv_preserves_existing_fields_and_unknown_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'runs.csv'
            original='회차,코드 버전(커밋 번호),시도,완주,이탈,시간(초),손댄 횟수,메모,기존칸\n이전,a,이전,예,없음,1,0,보존,유지\n'
            path.write_text(original)
            m.append_run(path,'상자1','키 입력',3.2,'가짜.csv',[])
            text=path.read_text()
            self.assertTrue(text.startswith(original))
            rows=list(csv.DictReader(io.StringIO(text)))
            self.assertEqual(rows[-1]['완주'],'사용자 확인 대기')
            self.assertEqual(rows[-1]['기존칸'],'')

    def test_main_start_failure_and_loop_error_close_all(self):
        @contextmanager
        def keyboard(): yield lambda: False
        for stage in ('camera','loop','success'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                bot=Mock(); camera=Mock(); worker=Mock(); recorder=Mock(); recorder.paths=[]
                if stage=='camera': camera.start.side_effect=RuntimeError('가짜 열기 오류')
                loop=Mock(return_value=('키 입력',1))
                if stage=='loop': loop.side_effect=RuntimeError('가짜 주행 오류')
                with patch.object(m,'ROOT',Path(directory)), patch.object(m.line,'keyboard_stop',keyboard), \
                     patch.object(m.line,'stop_all') as stop, patch.object(m.video,'LatestCamera',return_value=camera), \
                     patch.object(m,'VisionWorker',return_value=worker), patch.object(m,'DriveVideo',return_value=recorder), \
                     patch.object(m,'run_loop',loop), patch.object(m,'append_run') as append, \
                     patch.object(m.time,'sleep'), patch.dict(sys.modules,{'Raspbot_Lib':Mock(Raspbot=Mock(return_value=bot))}), \
                     patch.object(sys,'argv',['06_drive_signs.py','--trial','가짜1']):
                    if stage=='success': m.main()
                    else:
                        with self.assertRaises(RuntimeError): m.main()
                    camera.close.assert_called_once()
                    if stage!='camera':
                        worker.close.assert_called_once(); recorder.close.assert_called_once()
                    self.assertGreaterEqual(stop.call_count,1)
                    append.assert_called_once()
                    self.assertEqual(bot.Ctrl_BEEP_Switch.call_args.args,(0,))


if __name__=='__main__':
    unittest.main(verbosity=2)
