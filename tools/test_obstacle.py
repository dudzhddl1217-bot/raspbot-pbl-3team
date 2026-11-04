#!/usr/bin/env python3
# 지시서: specs/04-obstacle.md
"""실제 장치 접근 없이 장애물 정지·복구·부저·측정 주기를 검증한다."""
import csv
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("obstacle", Path(__file__).resolve().parents[1] / "missions/04_obstacle.py")
obstacle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(obstacle)


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        # 아주 작은 반올림 차이로 시간이 멈추지 않게 한다.
        self.now = round(self.now + max(seconds, 0.000001), 9)


class FakeBot:
    def __init__(self, samples=(300,), clock=None):
        self.samples = iter(samples)
        self.mm = None
        self.clock = clock or Clock()
        self.events = []
        self.read_times = []

    def read_data_array(self, reg, length):
        assert length == 1
        if reg == 0x1b:
            self.mm = next(self.samples)
            self.read_times.append(self.clock.now)
            return [self.mm >> 8]
        assert reg == 0x1a
        return [self.mm & 255]

    def Ctrl_Muto(self, number, speed):
        self.events.append((self.clock.now, 'motor', number, speed))

    def Ctrl_BEEP_Switch(self, state):
        self.events.append((self.clock.now, 'beep', state))

    def Ctrl_Ulatist_Switch(self, state):
        self.events.append((self.clock.now, 'ultrasonic', state))


class ObstacleTests(unittest.TestCase):
    def moving_controller(self):
        c = obstacle.ObstacleController()
        for value in (40,40,40):
            c.update(value)
        self.assertTrue(c.moving)
        return c

    def test_wait_for_three_valid_samples(self):
        c = obstacle.ObstacleController()
        for value in (40, 0, 40, 401):
            self.assertEqual(c.update(value), (0,False))
        self.assertEqual(c.update(40), (50,False))

    def test_median_rejects_single_low_and_high_outlier(self):
        c = self.moving_controller()
        self.assertEqual(c.update(1), (50,False))
        self.assertEqual(c.filtered_cm, 40)
        c = obstacle.ObstacleController()
        for value in (24, 25, 200):
            c.update(value)
        self.assertEqual(c.filtered_cm, 25)
        self.assertFalse(c.moving)

    def test_stop_restart_boundaries_and_single_beep(self):
        c = self.moving_controller()
        self.assertEqual(c.update(25), (50,False))
        self.assertEqual(c.update(25), (0,True))
        self.assertEqual(c.update(25), (0,False))
        for _ in range(3):
            self.assertEqual(c.update(34.9), (0,False))
        self.assertEqual(c.update(35), (0,False))
        self.assertEqual(c.update(35), (50,False))
        for _ in range(3):
            self.assertEqual(c.update(25.1), (50,False))

    def test_initial_middle_distance_stays_stopped(self):
        c = obstacle.ObstacleController()
        for _ in range(3):
            self.assertEqual(c.update(30), (0,False))

    def test_three_invalid_values_stop_and_clear_history(self):
        c = self.moving_controller()
        self.assertEqual(c.update(0), (50,False))
        self.assertEqual(c.update(401), (50,False))
        self.assertEqual(c.update(0), (0,False))
        self.assertEqual(list(c.samples), [])
        self.assertEqual(c.update(40), (0,False))
        self.assertEqual(c.update(40), (0,False))
        self.assertEqual(c.update(40), (50,False))

    def test_invalid_streak_resets_on_valid_sample(self):
        c = self.moving_controller()
        for value in (0,0,40,401,401,40):
            self.assertEqual(c.update(value), (50,False))
        self.assertEqual(c.invalid_count, 0)

    def test_recovery_does_not_restart_below_35(self):
        c = self.moving_controller()
        for _ in range(3): c.update(0)
        for _ in range(3): self.assertEqual(c.update(30), (0,False))
        for _ in range(3): c.update(35)
        self.assertTrue(c.moving)

    def test_invalid_during_recovery_is_not_used(self):
        c = self.moving_controller()
        for _ in range(3): c.update(0)
        for value in (40, 401, 40, 0):
            self.assertEqual(c.update(value), (0,False))
        self.assertEqual(c.update(40), (50,False))

    def test_invalid_ranges_and_400_included(self):
        for value in (0, -1, 400.1, None, float('nan'), float('inf'), True, '30'):
            with self.subTest(value=value):
                c = self.moving_controller()
                c.update(value)
                self.assertEqual(c.invalid_count, 1)
                self.assertEqual(list(c.samples), [40,40,40])
        c = obstacle.ObstacleController()
        for _ in range(3): c.update(400)
        self.assertTrue(c.moving)

    def test_mm_conversion_and_read_failure(self):
        self.assertEqual(obstacle.read_distance_cm(FakeBot((153,))),15.3)
        bot = FakeBot()
        bot.read_data_array = lambda *_: None
        with self.assertRaises(RuntimeError): obstacle.read_distance_cm(bot)

    def test_schedule_beep_duration_log_and_cleanup(self):
        clock = Clock()
        bot = FakeBot([400,400,400,250,250,350,350] + [400]*40, clock)
        stream = io.StringIO()
        with patch.object(obstacle.time,'monotonic',clock.monotonic), \
             patch.object(obstacle.time,'sleep',clock.sleep), redirect_stdout(io.StringIO()):
            obstacle.run_obstacle(bot, lambda: clock.now >= 1.8, stream)
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertGreater(len(rows),10)
        self.assertTrue(all(r['ruler_cm']=='' for r in rows))
        self.assertEqual([r['speed'] for r in rows[:3]], ['0','0','50'])
        self.assertTrue(all(abs(b-a-.05)<1e-5 for a,b in zip(bot.read_times,bot.read_times[1:])))
        self.assertGreaterEqual(bot.read_times[0],1)
        on = [e for e in bot.events if e[1:] == ('beep',1)]
        self.assertEqual(len(on),1)
        off = next(e for e in bot.events if e[0]>on[0][0] and e[1:]==('beep',0))
        self.assertAlmostEqual(off[0]-on[0][0],.2,places=5)
        self.assertTrue(any(e[1]=='motor' and e[-1]==50 and on[0][0]<e[0]<off[0] for e in bot.events))
        self.assert_cleanup(bot)

    def assert_cleanup(self,bot):
        self.assertEqual([e[1:] for e in bot.events[-6:]],
                         [('motor',i,0) for i in range(4)] + [('beep',0),('ultrasonic',0)])

    def test_keyboard_and_ctrl_c_stop(self):
        bot = FakeBot()
        with redirect_stdout(io.StringIO()):
            obstacle.run_obstacle(bot,lambda:True,io.StringIO())
        self.assert_cleanup(bot)
        self.assertFalse(any(e[1]=='motor' and e[-1]!=0 for e in bot.events))
        bot = FakeBot()
        def interrupt(): raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            obstacle.run_obstacle(bot,interrupt,io.StringIO())
        self.assert_cleanup(bot)

    def test_io_exception_cleanup(self):
        clock=Clock()
        bot=FakeBot(clock=clock)
        bot.read_data_array=lambda *_: None
        with patch.object(obstacle.time,'monotonic',clock.monotonic), \
             patch.object(obstacle.time,'sleep',clock.sleep):
            with self.assertRaises(RuntimeError):
                obstacle.run_obstacle(bot,lambda:False,io.StringIO())
        self.assert_cleanup(bot)

    def test_cleanup_attempts_all_devices_after_exception(self):
        bot=FakeBot()
        original=bot.Ctrl_Muto
        def fail_first(number,speed):
            original(number,speed)
            if number==0: raise OSError('가짜 정지 오류')
        bot.Ctrl_Muto=fail_first
        with self.assertRaises(RuntimeError): obstacle.cleanup(bot)
        self.assert_cleanup(bot)

    def test_speed_restriction(self):
        bot=FakeBot()
        for value in (-50,60,151):
            with self.assertRaises(ValueError): obstacle.set_speed(bot,value)
        self.assertEqual(bot.events,[])

    def test_gap_between_wheel_commands(self):
        clock = Clock()
        bot = FakeBot(clock=clock)
        with patch.object(obstacle.time, 'sleep', clock.sleep):
            obstacle.set_speed(bot, 50)
        self.assertEqual([e[2] for e in bot.events], [0,1,2,3])
        self.assertEqual([e[0] for e in bot.events], [0,.01,.02,.03])

    def test_periodic_resend_for_unchanged_forward_and_stop(self):
        for distance in (400,100):
            clock = Clock()
            bot = FakeBot([distance]*100,clock)
            with patch.object(obstacle.time,'monotonic',clock.monotonic), \
                 patch.object(obstacle.time,'sleep',clock.sleep), redirect_stdout(io.StringIO()):
                obstacle.run_obstacle(bot,lambda:clock.now>=1.9,io.StringIO())
            target = 50 if distance==400 else 0
            starts = [e[0] for e in bot.events
                      if e[1:]==('motor',0,target) and 1.1<=e[0]<1.9]
            self.assertGreaterEqual(len(starts),3)
            for a,b in zip(starts,starts[1:]):
                self.assertAlmostEqual(b-a,.2,places=5)

    def test_speed_change_does_not_wait_for_refresh(self):
        clock=Clock()
        bot=FakeBot([400,400,400,250,250]+[250]*100,clock)
        stream=io.StringIO()
        with patch.object(obstacle.time,'monotonic',clock.monotonic), \
             patch.object(obstacle.time,'sleep',clock.sleep), redirect_stdout(io.StringIO()):
            obstacle.run_obstacle(bot,lambda:clock.now>=1.7,stream)
        rows=list(csv.DictReader(io.StringIO(stream.getvalue())))
        moving=next(r for r in rows if r['speed']=='50')
        stopped=next(r for r in rows if r['beep_started']=='1')
        self.assertAlmostEqual(float(stopped['elapsed_s'])-float(moving['elapsed_s']),.1,places=4)
        self.assertTrue(any(e[1:]==('motor',0,0) and abs(e[0]-float(stopped['elapsed_s']))<.0001
                            for e in bot.events))

    def test_cleanup_repeats_stop_with_gap_even_after_exception(self):
        clock=Clock()
        bot=FakeBot(clock=clock)
        original=bot.Ctrl_Muto
        def fail_first(number,speed):
            original(number,speed)
            if len(bot.events)==1: raise OSError('첫 정지 명령 실패')
        bot.Ctrl_Muto=fail_first
        with patch.object(obstacle.time,'monotonic',clock.monotonic), \
             patch.object(obstacle.time,'sleep',clock.sleep):
            with self.assertRaises(RuntimeError): obstacle.cleanup(bot)
        motors=[e for e in bot.events if e[1]=='motor']
        self.assertEqual(len(motors),12)
        for i in range(3):
            batch=motors[i*4:(i+1)*4]
            self.assertEqual([e[2:] for e in batch],[(n,0) for n in range(4)])
            self.assertAlmostEqual(batch[0][0],i*.2,places=6)
            for a,b in zip(batch,batch[1:]):
                self.assertAlmostEqual(b[0]-a[0],.01,places=6)

    def test_silent_command_loss_is_recovered_by_resend(self):
        clock=Clock()
        bot=FakeBot([400]*100,clock)
        original=bot.Ctrl_Muto
        applied=[0,0,0,0]
        states=[]
        lost={'start':False,'stop':False}
        def drop_once(number,speed):
            original(number,speed)
            if number==2 and speed==50 and not lost['start']:
                lost['start']=True
            elif number==3 and speed==0 and applied[3]==50 and not lost['stop']:
                lost['stop']=True
            else:
                applied[number]=speed
            states.append((clock.now,tuple(applied)))
        bot.Ctrl_Muto=drop_once
        with patch.object(obstacle.time,'monotonic',clock.monotonic), \
             patch.object(obstacle.time,'sleep',clock.sleep), redirect_stdout(io.StringIO()):
            obstacle.run_obstacle(bot,lambda:clock.now>=1.8,io.StringIO())
        self.assertTrue(all(lost.values()))
        self.assertTrue(any(t<1.8 and wheels==(50,50,0,50) for t,wheels in states))
        self.assertTrue(any(t<1.8 and wheels==(50,50,50,50) for t,wheels in states))
        self.assertEqual(applied,[0,0,0,0])


if __name__ == '__main__':
    unittest.main()
