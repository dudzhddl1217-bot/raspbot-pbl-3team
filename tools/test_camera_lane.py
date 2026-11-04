#!/usr/bin/env python3
# 지시서: specs/07-camera-lane.md
"""실제 카메라·I2C·바퀴 없이 노란 선과 정지·녹화·체크 모드를 검증한다."""
from contextlib import contextmanager
import csv
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('camera_lane',ROOT/'missions/07_camera_lane.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def image(left=160,right=480):
    frame=np.zeros((480,640,3),np.uint8)
    for x in (left,right):
        if x is not None:
            frame[120:220,x]=(0,255,255)
    return frame


def lines(left=None,right=None):
    return dict(left=left,right=right,left_count=100 if left is not None else 0,
                right_count=100 if right is not None else 0)


class FindTests(unittest.TestCase):
    def test_center_width_and_means(self):
        frame=image(100,500)
        frame[120:220,102]=(0,255,255)
        self.assertEqual(m.find_lines(frame),dict(left=101.,right=500.,left_count=200,right_count=100))

    def test_roi_rows_and_half_boundary(self):
        frame=image(None,None)
        frame[119,:]=(0,255,255);frame[300,:]=(0,255,255)
        self.assertEqual(m.find_lines(frame),lines())
        frame[120:220,319:321]=(0,255,255)
        result=m.find_lines(frame)
        self.assertEqual(result['left'],319)
        self.assertEqual(result['right'],320)
        self.assertEqual(result['left_count'],100)

    def test_minimum_fifty_pixels(self):
        frame=image(None,None)
        frame[120:169,100]=(0,255,255)
        self.assertIsNone(m.find_lines(frame)['left'])
        frame[169,100]=(0,255,255)
        self.assertEqual(m.find_lines(frame)['left'],100.)
        self.assertEqual(m.find_lines(frame)['left_count'],50)

    def test_extended_roi_detects_rows_below_old_band(self):
        frame=image(None,None)
        frame[220:300,150]=(0,255,255)
        frame[220:300,490]=(0,255,255)
        result=m.find_lines(frame)
        self.assertEqual(result,dict(left=150.,right=490.,left_count=80,right_count=80))

    def test_hsv_thresholds_inclusive(self):
        cases=[((15,255,255),True),((40,255,255),True),((14,255,255),False),
               ((41,255,255),False),((30,80,255),True),((30,79,255),False),
               ((30,255,100),True),((30,255,99),False)]
        for color,accepted in cases:
            with self.subTest(color=color):
                # HSV 변환의 양자화와 분리해 정확한 문턱값을 검증한다.
                hsv=np.zeros((100,640,3),np.uint8)
                hsv[:,100]=color
                with patch.object(m.cv2,'cvtColor',return_value=hsv):
                    self.assertEqual(m.find_lines(image())['left'] is not None,accepted)

    def test_invalid_image_rejected(self):
        for frame in (None,np.zeros((720,1280,3),np.uint8),np.zeros((480,640),np.uint8),np.zeros((480,640,3),float)):
            with self.assertRaises(ValueError):m.find_lines(frame)

    def test_saved_fixture_detected(self):
        folder=ROOT/'results/camera_lane_test_images'
        folder.mkdir(parents=True,exist_ok=True)
        for name,frame,expected in (('both.png',image(160,480),lines(160.,480.)),
                                    ('left.png',image(160,None),lines(160.,None)),
                                    ('right.png',image(None,480),lines(None,480.)),
                                    ('none.png',image(None,None),lines())):
            p=folder/name
            self.assertTrue(cv2.imwrite(str(p),frame))
            self.assertEqual(m.find_lines(cv2.imread(str(p))),expected)


class ControllerTests(unittest.TestCase):
    def setUp(self):self.c=m.LaneController()

    def test_straight_and_direction(self):
        self.assertEqual(self.c.update(lines(160,480),0),(35,35,35,35))
        self.assertEqual(self.c.update(lines(200,520),.1),(51,51,19,19))
        self.assertEqual(self.c.update(lines(100,420),.2),(11,11,59,59))

    def test_corner_recalculation(self):
        self.assertEqual(m.wheel_speeds(378),(58,58,12,12))
        self.assertEqual(m.wheel_speeds(379.18628875968994),(59,59,11,11))

    def test_speed_limits_no_reverse(self):
        self.c.update(lines(0,639),0)
        for i,left in enumerate((0,300)):
            speeds=self.c.update(lines(left,None),.1+i*.1)
            self.assertTrue(all(0<=v<=80 for v in speeds))
        self.c.width=640
        self.assertEqual(self.c.update(lines(319,None),.4),(80,80,0,0))
        self.assertEqual(self.c.update(lines(None,320),.5),(0,0,80,80))

    def test_initial_one_line_waits_without_guess(self):
        self.assertEqual(self.c.update(lines(160,None),0),m.STOP)
        self.assertEqual(self.c.state,'WAITWIDTH')
        self.assertIsNone(self.c.width)
        self.assertEqual(self.c.update(lines(None,480),.4),m.STOP)
        self.assertEqual(self.c.update(lines(160,480),.5),(35,35,35,35))

    def test_width_updated_only_when_both_seen(self):
        self.c.update(lines(100,500),0)
        self.c.update(lines(120,None),.1)
        self.assertEqual(self.c.center,320)
        self.assertEqual(self.c.width,400)
        self.c.update(lines(None,490),.2)
        self.assertEqual(self.c.center,290)
        self.assertEqual(self.c.width,400)
        self.c.update(lines(200,400),.3)
        self.assertEqual(self.c.width,200)

    def test_last_turn_search_until_one_second_then_latched_stop(self):
        self.c.update(lines(200,520),0)
        for now in (.1,.4,.7,1.0):
            self.assertEqual(self.c.update(lines(),now),(60,60,0,0))
            self.assertEqual(self.c.state,'SEARCHRIGHT')
        self.assertEqual(self.c.tick(1.0999),(60,60,0,0))
        self.assertEqual(self.c.tick(1.1),m.STOP)
        self.assertEqual(self.c.state,'LOSTSTOP')
        self.assertEqual(self.c.update(lines(160,480),1.11),m.STOP)

    def test_left_search_and_straight_preserves_last_turn(self):
        self.c.update(lines(100,420),0)
        self.c.update(lines(160,480),.1)
        self.assertEqual(self.c.last_turn,'LEFT')
        self.assertEqual(self.c.update(lines(),.2),(0,0,60,60))
        self.assertEqual(self.c.state,'SEARCHLEFT')

    def test_recovery_before_loss_deadline(self):
        self.c.update(lines(200,520),0)
        self.c.update(lines(),.1)
        self.c.update(lines(),.4)
        self.c.update(lines(180,None),.7)
        self.assertFalse(self.c.stopped)
        self.assertIsNone(self.c.lost_since)
        self.assertEqual(self.c.state,'TRACKLEFT')

    def test_late_visible_frame_cannot_restart(self):
        self.c.update(lines(200,520),0)
        for now in (.1,.4,.7,1.0):self.c.update(lines(),now)
        self.assertEqual(self.c.update(lines(160,480),1.1001),m.STOP)
        self.assertTrue(self.c.stopped)

    def test_camera_freeze_during_search_still_stops_at_half_second(self):
        self.c.update(lines(200,520),0)
        self.c.update(lines(),.1)
        self.assertEqual(self.c.tick(.6001),m.STOP)
        self.assertEqual(self.c.state,'FRAMESTOP')

    def test_frame_stale_boundary_and_no_restart(self):
        self.c.update(lines(160,480),0)
        self.assertEqual(self.c.tick(.5),(35,35,35,35))
        self.assertEqual(self.c.tick(.5001),m.STOP)
        self.assertEqual(self.c.state,'FRAMESTOP')
        self.assertEqual(self.c.update(lines(160,480),.6),m.STOP)

    def test_no_previous_turn_stops_immediately(self):
        self.assertEqual(self.c.update(lines(),0),m.STOP)
        self.assertTrue(self.c.stopped)
        self.assertEqual(self.c.state,'LOSTSTOP')
        self.assertEqual(self.c.update(lines(160,480),.1),m.STOP)
        straight=m.LaneController()
        straight.update(lines(160,480),0)
        self.assertIsNone(straight.last_turn)
        self.assertEqual(straight.update(lines(),.1),m.STOP)
        self.assertTrue(straight.stopped)

    def test_overlay_copy_and_fields(self):
        frame=image();original=frame.copy()
        self.c.update(lines(160,480),0)
        report=self.c.report()
        marked=m.annotated(frame,report)
        np.testing.assert_array_equal(frame,original)
        self.assertGreater(np.count_nonzero(marked!=frame),0)
        source=m.LatestDisplay();source.update(frame,report)
        a,_=source.snapshot();a[:]=0
        self.assertGreater(source.snapshot()[0].sum(),0)


class Clock:
    def __init__(self):self.now=0
    def monotonic(self):return self.now
    def sleep(self,delay):self.now+=delay


class LoopTests(unittest.TestCase):
    def run_fake(self,check,mode):
        clock=Clock();camera=Mock();recorder=Mock();display=Mock();bot=Mock();output=io.StringIO()
        count=0
        def snapshot():
            nonlocal count
            count+=1
            if mode=='error':raise RuntimeError('가짜 카메라 오류')
            if mode=='interrupt':raise KeyboardInterrupt
            return image(),1 if mode=='frozen' else count
        camera.snapshot.side_effect=snapshot
        def key():return clock.now>=2.1 if mode=='long' else count>=2 if mode=='key' else False
        prints=[]
        with patch.object(m.time,'monotonic',clock.monotonic),patch.object(m.time,'sleep',clock.sleep), \
             patch('builtins.print',side_effect=lambda *a,**k:prints.append(clock.now)):
            if mode=='error':
                with self.assertRaises(RuntimeError):m.run_loop(camera,recorder,display,key,output,check,bot)
                result=None
            else:result=m.run_loop(camera,recorder,display,key,output,check,bot)
        return bot,output,result,prints

    def test_check_never_calls_motors_on_all_exits(self):
        for mode in ('key','frozen','error','interrupt'):
            with self.subTest(mode=mode):
                bot,_,_,_=self.run_fake(True,mode)
                bot.Ctrl_Muto.assert_not_called()

    def test_driving_all_exits_repeat_stop(self):
        for mode in ('key','frozen','error','interrupt'):
            with self.subTest(mode=mode):
                bot,_,_,_=self.run_fake(False,mode)
                self.assertEqual([c.args for c in bot.Ctrl_Muto.call_args_list[-12:]],[(i,0) for _ in range(3) for i in range(4)])

    def test_check_prints_once_a_second(self):
        bot,output,_,prints=self.run_fake(True,'long')
        self.assertEqual(len(prints),3)
        self.assertTrue(all(1<=b-a<1.02 for a,b in zip(prints,prints[1:])))
        rows=list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertTrue(all(r['모드']=='읽기전용' for r in rows))
        self.assertEqual(rows[0]['차선중심_x'],'320.0')

    def test_frozen_frames_do_not_refresh_timeout(self):
        _,_,(reason,report),_=self.run_fake(True,'frozen')
        self.assertEqual(reason,'FRAMESTOP');self.assertTrue(report['stopped'])

    def test_slow_processing_does_not_make_old_frame_fresh(self):
        clock=Clock(); camera=Mock(); camera.snapshot.return_value=(image(),1)
        bot=Mock(); recorder=Mock(); display=Mock()
        def slow_find(frame):
            clock.sleep(.6)
            return lines(160,480)
        with patch.object(m.time,'monotonic',clock.monotonic),patch.object(m.time,'sleep',clock.sleep), \
             patch.object(m,'find_lines',side_effect=slow_find):
            reason,report=m.run_loop(camera,recorder,display,lambda:False,io.StringIO(),False,bot)
        self.assertEqual(reason,'FRAMESTOP')
        self.assertTrue(all(c.args[1]==0 for c in bot.Ctrl_Muto.call_args_list))

    def test_check_main_has_no_driver_import_even_error(self):
        @contextmanager
        def keyboard():yield lambda:False
        camera=Mock();camera.start.side_effect=RuntimeError('가짜 열기 실패')
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'ROOT',Path(directory)), \
             patch.object(m.line,'keyboard_stop',keyboard),patch.object(m.video,'LatestCamera',return_value=camera), \
             patch.dict(sys.modules,{'Raspbot_Lib':None}),patch.object(sys,'argv',['07_camera_lane.py','--check']):
            with self.assertRaises(RuntimeError):m.main()
            camera.close.assert_called_once()

    def test_main_check_success_closes_without_driver(self):
        @contextmanager
        def keyboard():yield lambda:False
        camera=Mock();camera.snapshot.return_value=(image(),1);recorder=Mock()
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'ROOT',Path(directory)), \
             patch.object(m.line,'keyboard_stop',keyboard),patch.object(m.video,'LatestCamera',return_value=camera), \
             patch.object(m,'LaneVideo',return_value=recorder),patch.object(m,'run_loop',return_value=('키 입력',{})), \
             patch.dict(sys.modules,{'Raspbot_Lib':None}),patch.object(sys,'argv',['07_camera_lane.py','--check']):
            m.main();camera.close.assert_called_once();recorder.close.assert_called_once()

    def test_video_fps_repeats_release(self):
        clock=Clock();source=m.LatestDisplay();c=m.LaneController();c.update(lines(160,480),0)
        source.update(image(),c.report());recorder=m.LaneVideo(source);writer=Mock()
        class Event:
            def is_set(self):return recorder.saved_frames>=4
            def wait(self,delay):clock.sleep(delay);return self.is_set()
        recorder.stop=Event()
        with tempfile.TemporaryDirectory() as directory:
            recorder.output_dir=Path(directory)
            with patch.object(m.time,'monotonic',clock.monotonic),patch.object(recorder,'_open_writer',return_value=writer):recorder.run()
        self.assertEqual(recorder.saved_frames,4);self.assertEqual(recorder.repeated_frames,3)
        self.assertAlmostEqual(clock.now,3/15);writer.release.assert_called_once()
        self.assertEqual(recorder.segment_frames,900);self.assertEqual(recorder.keep,30)


if __name__=='__main__':unittest.main(verbosity=2)
