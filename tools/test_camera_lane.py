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
    def test_largest_two_components_and_means(self):
        frame=image(100,500)
        frame[120:220,101]=(0,255,255)
        frame[120:180,250]=(0,255,255)
        self.assertEqual(m.find_components(frame),[(100.5,200),(500.,100)])

    def test_roi_and_connected_crossing(self):
        frame=image(None,None)
        frame[119,:]=(0,255,255);frame[300,:]=(0,255,255)
        self.assertEqual(m.find_components(frame),[])
        frame[120:220,319:321]=(0,255,255)
        self.assertEqual(m.find_components(frame),[(319.5,200)])
        self.assertTrue(m.find_lines(frame)['unassigned'])

    def test_minimum_and_extended_roi(self):
        frame=image(None,None);frame[220:269,100]=(0,255,255)
        self.assertEqual(m.find_components(frame),[])
        frame[269,100]=(0,255,255)
        self.assertEqual(m.find_components(frame),[(100.,50)])

    def test_hsv_thresholds_inclusive(self):
        for color,accepted in [((15,255,255),True),((40,255,255),True),((14,255,255),False),
                ((41,255,255),False),((30,80,255),True),((30,79,255),False),
                ((30,255,100),True),((30,255,99),False)]:
            hsv=np.zeros((180,640,3),np.uint8);hsv[:,100]=color
            with patch.object(m.cv2,'cvtColor',return_value=hsv):
                self.assertEqual(bool(m.find_components(image())),accepted)

    def test_invalid_image_rejected(self):
        for frame in (None,np.zeros((720,1280,3),np.uint8),np.zeros((480,640),np.uint8),np.zeros((480,640,3),float)):
            with self.assertRaises(ValueError):m.find_components(frame)

    def test_saved_fixture_read_only(self):
        folder=ROOT/'results/camera_lane_test_images'
        self.assertEqual(m.find_lines(cv2.imread(str(folder/'both.png'))),lines(160.,480.))
        self.assertTrue(m.find_lines(cv2.imread(str(folder/'left.png')))['unassigned'])
        self.assertEqual(m.find_lines(cv2.imread(str(folder/'none.png'))),lines())

    def test_single_keeps_left_identity_after_320(self):
        previous=dict(left=300.,right=600.)
        result=m.assign_lines([(350.,100)],previous,300.)
        self.assertEqual(result,lines(350.,None))
        c=m.LaneController();c.update(lines(300,600),0)
        c.update(result,.1)
        self.assertEqual(c.raw_center,500.)
        self.assertEqual(c.center,475.)
        self.assertEqual(c.speeds,(50,50,30,30))

    def test_two_components_always_sorted_even_if_previous_reversed(self):
        self.assertEqual(m.assign_lines([(100.,100),(500.,100)],dict(left=500,right=100),400),lines(100.,500.))
        self.assertEqual(m.assign_lines([(553.,100),(173.,100)],dict(left=553,right=173),380),lines(173.,553.))

    def test_merge_weighted_and_strict_300_boundary(self):
        prev=dict(left=100,right=500)
        r=m.assign_lines([(110.,100),(150.,300)],prev,400)
        self.assertEqual(r,dict(left=140.,right=None,left_count=400,right_count=0))
        r=m.assign_lines([(110.,100),(410.,100)],prev,400)
        self.assertEqual(r,lines(110.,410.))
        r=m.assign_lines([(110.,100),(409.,100)],prev,None)
        self.assertEqual(r,dict(left=259.5,right=None,left_count=200,right_count=0))
        c=m.LaneController();c.update(lines(100,500),0);c.update(m.assign_lines([(110.,100),(150.,300)],c.previous,c.width),.1)
        self.assertEqual(c.width,400)
        self.assertEqual(c.previous,dict(left=140.,right=500))

    def test_initial_single_waits_then_initializes(self):
        c=m.LaneController();c.update(m.find_lines(image(350,None)),0)
        self.assertEqual(c.state,'WAITWIDTH');self.assertFalse(c.stopped)
        c.update(m.find_lines(image()),.1)
        self.assertEqual(c.width,320)


class ControllerTests(unittest.TestCase):
    def setUp(self):self.c=m.LaneController()

    def test_straight_and_direction(self):
        self.assertEqual(self.c.update(lines(160,480),0),(35,35,35,35))
        self.assertEqual(self.c.update(lines(200,520),.1),(40,40,30,30))
        self.assertEqual(self.c.update(lines(100,420),.2),(35,35,35,35))

    def test_corner_recalculation(self):
        self.assertEqual(m.wheel_speeds(378),(52,52,18,18))
        self.assertEqual(m.wheel_speeds(379.18628875968994),(53,53,17,17))

    def test_speed_limits_no_reverse(self):
        self.c.update(lines(0,639),0)
        for i,left in enumerate((0,300)):
            speeds=self.c.update(lines(left,None),.1+i*.1)
            self.assertTrue(all(0<=v<=80 for v in speeds))
        self.c.width=600
        self.c.centers.clear()
        self.assertEqual(self.c.update(lines(639,None),.4),(45,45,35,35))
        self.assertEqual(self.c.raw_center,640)
        self.c.centers.clear()
        self.assertEqual(self.c.update(lines(None,0),.5),(40,40,40,40))
        self.assertEqual(self.c.raw_center,0)

    def test_initial_one_line_waits_without_guess(self):
        self.assertEqual(self.c.update(lines(160,None),0),m.STOP)
        self.assertEqual(self.c.state,'WAITWIDTH')
        self.assertIsNone(self.c.width)
        self.assertEqual(self.c.update(lines(None,480),.4),m.STOP)
        self.assertEqual(self.c.update(lines(160,480),.5),(35,35,35,35))

    def test_width_updated_only_when_both_seen(self):
        self.c.update(lines(100,500),0)
        self.c.update(lines(120,None),.1)
        self.assertEqual(self.c.raw_center,320)
        self.assertEqual(self.c.width,400)
        self.c.update(lines(None,490),.2)
        self.assertEqual(self.c.raw_center,290)
        self.assertEqual(self.c.width,400)
        self.c.update(lines(200,400),.3)
        self.assertEqual(self.c.width,400)

    def test_width_limits_include_boundaries_and_keep_previous(self):
        self.c.update(lines(100,400),0)
        self.assertEqual(self.c.width,300)
        self.c.update(lines(20,620),.1)
        self.assertEqual(self.c.width,600)
        self.c.update(lines(0,639),.2)
        self.assertEqual(self.c.width,600)
        self.c.update(lines(100,200),.3)
        self.assertEqual(self.c.width,600)
        c=m.LaneController();c.update(lines(0,639),0)
        self.assertIsNone(c.width)

    def test_three_new_centers_smoothed_and_clipped(self):
        for now,left,right in ((0,100,500),(.1,120,520),(.2,140,540),(.3,160,560)):
            self.c.update(lines(left,right),now)
        self.assertEqual(self.c.raw_center,360)
        self.assertEqual(self.c.center,340)
        self.assertEqual(self.c.speeds,(40,40,30,30))
        self.c.update(lines(),.4)
        self.assertEqual(len(self.c.centers),3)

    def test_last_turn_search_until_one_second_then_latched_stop(self):
        self.c.update(lines(200,520),0)
        for now,expected in ((.1,(40,40,20,20)),(.4,(45,45,15,15)),(.7,(50,50,10,10)),(1.0,(55,55,5,5))):
            self.assertEqual(self.c.update(lines(),now),expected)
            self.assertEqual(self.c.state,'SEARCHRIGHT')
        self.assertEqual(self.c.tick(1.0999),(55,55,5,5))
        self.assertEqual(self.c.tick(1.1),m.STOP)
        self.assertEqual(self.c.state,'LOSTSTOP')
        self.assertEqual(self.c.update(lines(160,480),1.11),m.STOP)

    def test_left_search_and_straight_preserves_last_turn(self):
        self.c.update(lines(100,420),0)
        self.c.update(lines(160,480),.1)
        self.assertEqual(self.c.last_turn,'LEFT')
        self.assertEqual(self.c.update(lines(),.2),(16,16,44,44))
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


class SlopeAndSlewTests(unittest.TestCase):
    def slanted(self, start, end):
        frame=image(None,None)
        cv2.line(frame,(start,130),(end,290),(0,255,255),3)
        return frame

    def test_slope_overrules_previous_position_and_uses_bottom(self):
        for start,end,name in ((550,350,'left'),(100,300,'right')):
            frame=self.slanted(start,end)
            parts=m.find_components(frame)
            result=m.find_lines(frame,dict(left=100,right=500),400)
            other='right' if name=='left' else 'left'
            self.assertIsNone(result[other])
            part=parts[0]
            self.assertEqual(result[name],float(part.xs[part.ys==part.ys.max()].mean()))
            self.assertNotAlmostEqual(result[name],part[0])

    def test_horizontal_ratio_boundary_keeps_name_and_no_history_waits(self):
        for width in (60,61):
            frame=image(None,None);frame[150:170,200:200+width]=(0,255,255)
            self.assertTrue(m.find_lines(frame)['unassigned'])
            for name in ('left','right'):
                found=m.find_lines(frame,dict(left=0,right=640),400,name)
                self.assertIsNotNone(found[name])
        ys,xs=np.indices((20,59))
        xs=xs.ravel()+200;ys=ys.ravel()+150
        xs=np.append(xs,258);ys=np.append(ys,169)
        part=m.Component(xs,ys)
        self.assertIsNotNone(m.assign_lines([part],last_name='left')['right'])

    def test_merged_fragments_fit_combined_points(self):
        frame=image(None,None)
        cv2.line(frame,(490,130),(420,180),(0,255,255),2)
        cv2.line(frame,(390,200),(300,290),(0,255,255),2)
        self.assertEqual(len(m.find_components(frame)),2)
        found=m.find_lines(frame,last_name='right')
        self.assertIsNotNone(found['left']);self.assertIsNone(found['right'])
        self.assertLess(found['left'],310)

    def test_single_history_survives_both_lines(self):
        c=m.LaneController()
        c.update(m.find_lines(self.slanted(500,300)),0)
        self.assertEqual(c.single_name,'left')
        c.update(lines(100,500),.1)
        frame=image(None,None);frame[150:160,200:300]=(0,255,255)
        c.update(m.find_lines(frame,c.previous,c.width,c.single_name),.2)
        self.assertEqual(c.state,'TRACKLEFT')

    def test_difference_bound_all_targets_and_transition_to_search(self):
        c=m.LaneController();c.update(lines(100,500),0)
        previous=c.speeds
        for i in range(1,100):
            if i<40:
                c.centers.clear()
                target=lines(639,None) if i%2 else lines(None,0)
            else:
                target=lines() if i<50 else lines(100,500)
            speeds=c.update(target,i*.01)
            self.assertLessEqual(abs((speeds[0]-speeds[2])-(previous[0]-previous[2])),10)
            self.assertTrue(all(0<=v<=80 for v in speeds))
            previous=speeds
        self.assertEqual(c.tick(2),m.STOP)

    def test_sustained_turn_can_reach_80_and_stop_is_immediate(self):
        c=m.LaneController();c.update(lines(100,500),0)
        for i in range(1,12):
            c.update(lines(639,None),i*.01)
        self.assertEqual(c.speeds,(80,80,0,0))
        self.assertEqual(c.tick(1),m.STOP)
        self.assertTrue(c.stopped)


class Clock:
    def __init__(self):self.now=0
    def monotonic(self):return self.now
    def sleep(self,delay):self.now+=delay


class LoopTests(unittest.TestCase):
    def test_repeated_sigint_stops_before_threads_and_closes_quietly(self):
        import os
        import signal
        import threading
        import time
        @contextmanager
        def keyboard():
            yield lambda:False
        events=[]; timers=[]; motors=[]; cameras=[]; recorders=[]
        cap=Mock(); cap.isOpened.return_value=True
        def read():
            time.sleep(.01)
            return True,image()
        cap.read.side_effect=read
        cap.release.side_effect=lambda:events.append('camera-release')
        writer=Mock();writer.release.side_effect=lambda:events.append('writer-release')
        bot=Mock(); started=False
        def motor(number,speed):
            nonlocal started
            motors.append((number,speed));events.append(('motor',number,speed))
            if speed and not started:
                started=True
                for delay in (.02,.05):
                    timer=threading.Timer(delay,lambda:os.kill(os.getpid(),signal.SIGINT))
                    timers.append(timer);timer.start()
        bot.Ctrl_Muto.side_effect=motor
        camera_class=m.video.LatestCamera;recorder_class=m.LaneVideo
        def camera_factory():
            camera=camera_class();cameras.append(camera);return camera
        def recorder_factory(source):
            recorder=recorder_class(source);recorders.append(recorder);return recorder
        old_handler=signal.getsignal(signal.SIGINT)
        try:
            with tempfile.TemporaryDirectory() as directory,patch.object(m,'ROOT',Path(directory)), \
                 patch.object(m.line,'keyboard_stop',keyboard),patch.object(m.video,'open_camera',return_value=cap), \
                 patch.object(m.video,'LatestCamera',side_effect=camera_factory), \
                 patch.object(m,'LaneVideo',side_effect=recorder_factory), \
                 patch.object(recorder_class,'_open_writer',return_value=writer), \
                 patch.dict(sys.modules,{'Raspbot_Lib':Mock(Raspbot=Mock(return_value=bot))}), \
                 patch.object(sys,'argv',['07_camera_lane.py']),patch('builtins.print'):
                m.main()
        finally:
            for timer in timers:timer.join()
        self.assertTrue(started)
        self.assertEqual(motors[-12:],[(i,0) for _ in range(3) for i in range(4)])
        last_motor=max(i for i,event in enumerate(events) if isinstance(event,tuple))
        self.assertLess(last_motor,events.index('writer-release'))
        self.assertLess(last_motor,events.index('camera-release'))
        self.assertFalse(recorders[0].thread.daemon)
        self.assertFalse(recorders[0].thread.is_alive())
        self.assertFalse(cameras[0]._thread.is_alive())
        self.assertEqual(signal.getsignal(signal.SIGINT),old_handler)

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
        def slow_find(frame, *args):
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
