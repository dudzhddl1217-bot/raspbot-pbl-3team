#!/usr/bin/env python3
# 지시서: specs/02-line.md
"""실제 라즈봇 없이 줄 주행 판단과 정지 처리를 검증한다."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("line", Path(__file__).resolve().parents[1] / "missions/02_line.py")
line = importlib.util.module_from_spec(spec)
spec.loader.exec_module(line)


def raw(physical):
    return tuple(physical[line.SENSOR_ORDER.index(i)] for i in range(4))


class FakeBot:
    def __init__(self, data=(9,)):
        self.data = data
        self.commands = []

    def read_data_array(self, register, count):
        assert (register, count) == (0x0a, 1)
        return self.data

    def Ctrl_Muto(self, number, speed):
        self.commands.append((number, speed))


class LineTests(unittest.TestCase):
    def test_all_patterns_and_hold(self):
        expected = {
            '0000': (50,50,50,50), '0001': (50,50,50,50),
            '0010': (30,30,60,60), '0011': (15,15,60,60),
            '0100': (30,30,60,60), '0101': (30,30,60,60),
            '0110': (30,30,60,60), '0111': (0,0,60,60),
            '1000': (50,50,50,50), '1001': (50,50,50,50),
            '1010': (30,30,60,60), '1011': (30,30,60,60),
            '1100': (60,60,15,15), '1101': (60,60,30,30),
            '1110': (60,60,0,0), '1111': (0,0,40,40),
        }
        for text, speeds in expected.items():
            with self.subTest(pattern=text):
                controller = line.LineFollower()
                controller.step(raw((1,0,1,1)), 0)
                self.assertEqual(controller.step(raw(tuple(map(int,text))), .1), speeds)

    def test_lost_boundary_and_latched_stop(self):
        for pattern, search in [((0,1,1,1),(0,0,40,40)), ((1,1,1,0),(40,40,0,0))]:
            c = line.LineFollower()
            c.step(raw(pattern), 0)
            self.assertEqual(c.step((1,1,1,1), 1), search)
            self.assertEqual(c.step((1,1,1,1), 2), search)
            self.assertEqual(c.step((1,1,1,1), 2.001), line.STOP)
            self.assertTrue(c.stopped)
            self.assertEqual(c.step((0,0,0,0), 3), line.STOP)

    def test_no_turn_lost_stops_immediately(self):
        c = line.LineFollower()
        c.step(raw((1,0,0,1)), 0)
        self.assertEqual(c.step((1,1,1,1), .1), line.STOP)
        self.assertTrue(c.stopped)

    def test_reacquisition_resets_lost_timer(self):
        c = line.LineFollower()
        c.step(raw((1,1,0,1)), 0)
        c.step((1,1,1,1), 1)
        c.step(raw((1,0,0,1)), 1.9)
        self.assertEqual(c.step((1,1,1,1), 2.2), (40,40,0,0))
        self.assertFalse(c.stopped)

    def test_keyboard_and_error_cleanup(self):
        bot = FakeBot()
        keys = iter((False,False,True))
        with patch.object(line.time, 'sleep'):
            line.follow(bot, lambda: next(keys))
        self.assertEqual(bot.commands[-4:], list(enumerate(line.STOP)))
        bad = FakeBot(data=None)
        with self.assertRaises(RuntimeError):
            line.follow(bad, lambda: False)
        self.assertEqual(bad.commands, list(enumerate(line.STOP)))
        interrupted = FakeBot()
        with self.assertRaises(KeyboardInterrupt):
            line.follow(interrupted, lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertEqual(interrupted.commands, list(enumerate(line.STOP)))

    def test_stop_attempts_all_four_after_exception(self):
        bot = FakeBot()
        def motor(number, speed):
            bot.commands.append((number,speed))
            if number == 0:
                raise OSError('fake motor error')
        bot.Ctrl_Muto = motor
        with self.assertRaises(RuntimeError):
            line.stop_all(bot)
        self.assertEqual(bot.commands, list(enumerate(line.STOP)))

    def test_read_only_mode_sends_no_motor_commands(self):
        bot = FakeBot()
        with patch.object(line.time,'monotonic',side_effect=[0,0,11]), patch.object(line.time,'sleep'):
            line.check_sensors(bot, 10)
        self.assertEqual(bot.commands, [])

    def test_speed_limit(self):
        bot = FakeBot()
        with self.assertRaises(ValueError):
            line.drive(bot, (151,0,0,0))
        self.assertEqual(bot.commands, [])


if __name__ == '__main__':
    unittest.main()
