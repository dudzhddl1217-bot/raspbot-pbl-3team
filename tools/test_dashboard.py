#!/usr/bin/env python3
# 지시서: specs/09-dashboard.md
"""실제 라즈봇·카메라·I2C 없이 조종판을 검증한다. 가짜 주행 프로그램을 pty로 띄워 시작·로그·CSV·정지를 본다."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dashboard', ROOT / 'missions/09_dashboard.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

FAKE_DRIVE = r'''#!/usr/bin/env python3
"""가짜 주행 프로그램: 상태 출력·CSV 기록·아무 키/Ctrl+C 정지. 장치 없음."""
import csv, select, signal, sys, termios, time, tty
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
stubborn = 'stubborn' in sys.argv
trial = sys.argv[sys.argv.index('--trial') + 1] if '--trial' in sys.argv else 'x'
if stubborn:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
if not sys.stdin.isatty():
    print('TTY 아님', flush=True)
    sys.exit(3)
fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setcbreak(fd)
path = ROOT / 'results' / f'fake_{trial}.csv'
path.parent.mkdir(parents=True, exist_ok=True)
print(f'가짜 시작 trial={trial} stubborn={stubborn} 한글 확인 가나다', flush=True)
reason = '시간 종료'
try:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['경과초', '상태', 'x1', 'x2', 'x3', 'x4', '바퀴속도', '인식이름', '점수', '상자높이_px'])
        f.flush()
        start = time.monotonic()
        i = 0
        while time.monotonic() - start < 8:
            if select.select([sys.stdin], [], [], 0)[0]:
                reason = '키 입력'
                break
            state = 'FORWARD' if i % 2 == 0 else 'LEFTGENTLE'
            w.writerow([round(time.monotonic() - start, 3), state, 0, 1, 0, 1, '[50, 50, 50, 50]', 'person', '0.8123', 120])
            f.flush()
            print(f'상태 {state}, 센서 (0, 1, 0, 1), 바퀴 (50, 50, 50, 50)', flush=True)
            i += 1
            time.sleep(0.1)
except KeyboardInterrupt:
    reason = 'Ctrl+C'
finally:
    termios.tcsetattr(fd, termios.TCSANOW, old)
    print(f'종료: {reason}', flush=True)
'''


class FakeBot:
    def __init__(self):
        self.calls = []

    def Ctrl_Muto(self, number, speed):
        self.calls.append(('motor', number, speed))

    def Ctrl_BEEP_Switch(self, state):
        self.calls.append(('beep', state))

    def Ctrl_Ulatist_Switch(self, state):
        self.calls.append(('ultra', state))


class FakePreview:
    def __init__(self):
        self.on = False
        self.closed = 0
        self.error = None

    @property
    def running(self):
        return self.on

    def start(self):
        self.on = True

    def close(self):
        self.on = False
        self.closed += 1

    def jpeg(self):
        return b'jpeg' if self.on else None


EXPECTED_ZEROS = ([('motor', n, 0) for n in range(4)] + [('beep', 0), ('ultra', 0)]) * m.STOP_REPEATS


def wait_for(condition, timeout=10, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(step)
    return condition()


class Harness:
    def __init__(self, bot_factory=None, grace=0.5):
        self.tmp = Path(tempfile.mkdtemp(prefix='dashboard-test-'))
        (self.tmp / 'missions').mkdir()
        (self.tmp / 'results').mkdir()
        (self.tmp / 'missions' / 'fake_drive.py').write_text(FAKE_DRIVE, encoding='utf-8')
        self.bots = []

        def factory():
            bot = FakeBot()
            self.bots.append(bot)
            return bot

        self.preview = FakePreview()
        self.app = m.App(root=self.tmp, bot_factory=bot_factory or factory, preview=self.preview, grace=grace)
        self.runner = self.app.runner
        self.events = self.app.bus.subscribe()

    def texts(self):
        return [e['text'] for e in self.runner.log]

    def has_text(self, part):
        return any(part in t for t in self.texts())

    def start(self, **kw):
        payload = dict(program='missions/fake_drive.py', trial='t1', args='--trial t1', confirmed=True)
        payload.update(kw)
        return self.app.start(payload)

    def wait_done(self, timeout=12):
        return wait_for(lambda: not self.runner.running, timeout) and self.runner.done.wait(timeout=5)

    def close(self):
        if self.runner.running:
            self.runner.estop('시험 정리')
            self.wait_done()
        shutil.rmtree(self.tmp, ignore_errors=True)


class LineSplitterTests(unittest.TestCase):
    def test_multibyte_boundary_and_crlf(self):
        s = m.LineSplitter()
        data = '상태 FORWARD\r\n한글'.encode('utf-8')
        first = s.feed(data[:7])          # '상태'의 둘째 글자 중간에서 자른다
        rest = s.feed(data[7:])
        self.assertEqual(first, [])
        self.assertEqual(rest, ['상태 FORWARD'])
        self.assertEqual(s.flush(), ['한글'])
        self.assertEqual(s.flush(), [])


class ZerosTests(unittest.TestCase):
    def test_zero_bundle_three_times(self):
        bot = FakeBot()
        said = []
        self.assertTrue(m.send_zeros(lambda: bot, said.append))
        self.assertEqual(bot.calls, EXPECTED_ZEROS)
        self.assertTrue(any('3회' in s for s in said))

    def test_without_driver_is_skipped_safely(self):
        said = []

        def missing():
            raise ImportError('No module named Raspbot_Lib')

        self.assertFalse(m.send_zeros(missing, said.append))
        self.assertTrue(any('드라이버' in s for s in said))

    def test_i2c_error_output_is_reported(self):
        class NoisyBot(FakeBot):
            def Ctrl_Muto(self, number, speed):
                print('write_array I2C error')
                super().Ctrl_Muto(number, speed)

        said = []
        self.assertFalse(m.send_zeros(NoisyBot, said.append))
        self.assertTrue(any('I2C 오류' in s for s in said))

    def test_exit_text(self):
        self.assertIn('정상', m.Runner._exit_text(0))
        self.assertIn('오류', m.Runner._exit_text(1))
        self.assertIn('SIGINT', m.Runner._exit_text(-2))
        self.assertIn('SIGKILL', m.Runner._exit_text(-9))


class CsvTailTests(unittest.TestCase):
    def test_picks_newest_after_start_and_ignores_runs(self):
        tmp = Path(tempfile.mkdtemp(prefix='csvtail-'))
        try:
            old = tmp / 'old.csv'
            old.write_text('a,b\n1,2\n', encoding='utf-8')
            os.utime(old, (time.time() - 100, time.time() - 100))
            (tmp / 'runs.csv').write_text('회차,메모\n1,x\n', encoding='utf-8')
            since = time.time() - 1
            new = tmp / 'drive_signs_new.csv'
            new.write_text('경과초,상태\n0.1,FORWARD\n', encoding='utf-8')
            files, rows = [], []
            tail = m.CsvTail(tmp, since, files.append, lambda f, r: rows.append((f, r)))
            tail.start()
            self.assertTrue(wait_for(lambda: len(rows) >= 1))
            self.assertEqual(files, [new])
            self.assertEqual(rows[0], (['경과초', '상태'], {'경과초': '0.1', '상태': 'FORWARD'}))
            with new.open('a', encoding='utf-8') as f:
                f.write('0.2,"LEFT, GENTLE"\n0.3')   # 줄 끝이 아직 안 온 조각
                f.flush()
            self.assertTrue(wait_for(lambda: len(rows) >= 2))
            self.assertEqual(rows[1][1]['상태'], 'LEFT, GENTLE')
            with new.open('a', encoding='utf-8') as f:
                f.write(',STOPWAIT\n')
            self.assertTrue(wait_for(lambda: len(rows) >= 3))
            self.assertEqual(rows[2][1], {'경과초': '0.3', '상태': 'STOPWAIT'})
            tail.close()
            self.assertFalse(tail.is_alive())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def tearDown(self):
        self.h.close()

    def test_start_requires_confirmation_and_known_program(self):
        with self.assertRaises(ValueError):
            self.h.start(confirmed=False)
        with self.assertRaises(ValueError):
            self.h.start(program='missions/none.py')
        with self.assertRaises(ValueError):
            self.h.start(max_seconds=-1)
        self.assertFalse(self.h.runner.running)
        self.assertEqual(self.h.runner.programs(), ['missions/fake_drive.py'])

    def test_run_logs_rows_key_stop_and_zeros(self):
        status = self.h.start()
        self.assertTrue(status['running'])
        self.assertGreater(status['pid'], 0)
        self.assertTrue(wait_for(lambda: self.h.has_text('가짜 시작 trial=t1')))
        self.assertTrue(self.h.has_text('한글 확인 가나다'))
        self.assertTrue(wait_for(lambda: self.h.has_text('상태 FORWARD, 센서 (0, 1, 0, 1), 바퀴 (50, 50, 50, 50)')))
        self.assertTrue(wait_for(lambda: len(self.h.runner.rows) >= 2))
        row = self.h.runner.rows[-1]
        self.assertEqual(self.h.runner.fields[:2], ['경과초', '상태'])
        self.assertIn(row['상태'], ('FORWARD', 'LEFTGENTLE'))
        self.assertEqual(row['바퀴속도'], '[50, 50, 50, 50]')
        self.assertEqual(self.h.runner.info['csv_file'], 'results/fake_t1.csv')
        self.assertEqual(self.h.bots, [])          # 주행 중에는 조종판이 바퀴 명령을 보내지 않는다
        self.h.runner.send_key()
        self.assertTrue(self.h.wait_done())
        self.assertTrue(self.h.has_text('종료: 키 입력'))
        self.assertEqual(self.h.runner.info['exit_code'], 0)
        self.assertEqual(len(self.h.bots), 1)
        self.assertEqual(self.h.bots[0].calls, EXPECTED_ZEROS)
        self.assertTrue(self.h.runner.info['zeros_sent'])
        self.assertIsNotNone(self.h.runner.info['ended_at'])
        log = (self.h.tmp / self.h.runner.info['log_file']).read_text(encoding='utf-8')
        self.assertIn('[child] 상태 FORWARD', log)
        self.assertIn('[dash] 네 바퀴 0', log)
        paths = [f['path'] for f in self.h.runner.files()]
        self.assertIn('fake_t1.csv', paths)
        self.assertTrue(any(p.startswith('dashboard_') for p in paths))
        events = []
        while not self.h.events.empty():
            events.append(self.h.events.get_nowait())
        kinds = {e['type'] for e in events}
        self.assertTrue({'log', 'row', 'file', 'status'}.issubset(kinds))

    def test_estop_sigint_path(self):
        self.h.start()
        self.assertTrue(wait_for(lambda: self.h.has_text('상태 FORWARD')))
        began = time.monotonic()
        status = self.h.runner.estop()
        self.assertLess(time.monotonic() - began, 5)
        self.assertFalse(status['running'])
        self.assertEqual(status['exit_code'], 0)
        self.assertTrue(self.h.has_text('종료: Ctrl+C'))
        self.assertTrue(self.h.has_text('SIGINT'))
        self.assertFalse(self.h.has_text('강제 종료'))
        self.assertEqual(len(self.h.bots), 1)
        self.assertEqual(self.h.bots[0].calls, EXPECTED_ZEROS)

    def test_estop_sigkill_when_sigint_ignored(self):
        self.h.start(args='stubborn --trial t1')
        self.assertTrue(wait_for(lambda: self.h.has_text('stubborn=True')))
        status = self.h.runner.estop()
        self.assertFalse(status['running'])
        self.assertEqual(status['exit_code'], -9)
        self.assertIn('SIGKILL', status['exit_text'])
        self.assertTrue(self.h.has_text('강제 종료'))
        self.assertEqual(len(self.h.bots), 1)
        self.assertEqual(self.h.bots[0].calls, EXPECTED_ZEROS)

    def test_second_start_rejected_and_zeros_rejected_while_running(self):
        self.h.start()
        with self.assertRaises(RuntimeError):
            self.h.start()
        with self.assertRaises(RuntimeError):
            self.h.runner.zeros()
        with self.assertRaises(RuntimeError):
            self.h.app.set_preview(True)
        self.assertEqual(self.h.bots, [])
        self.h.runner.estop()
        self.assertTrue(self.h.wait_done())

    def test_watchdog_stops_after_max_seconds(self):
        self.h.start(max_seconds=0.7)
        self.assertTrue(self.h.wait_done(timeout=6))
        self.assertTrue(self.h.has_text('최대 주행 시간 0.7초 초과'))
        self.assertEqual(self.h.runner.info['exit_code'], 0)
        self.assertEqual(self.h.bots[0].calls, EXPECTED_ZEROS)

    def test_preview_closed_before_start(self):
        self.h.preview.start()
        self.assertTrue(self.h.app.status()['preview'])
        self.h.start()
        self.assertEqual(self.h.preview.closed, 1)
        self.assertFalse(self.h.preview.running)
        self.assertTrue(self.h.has_text('미리보기를 끄고'))
        self.h.runner.estop()
        self.assertTrue(self.h.wait_done())

    def test_estop_and_zeros_when_idle(self):
        status = self.h.runner.estop()
        self.assertFalse(status['running'])
        self.assertEqual(self.h.bots[0].calls, EXPECTED_ZEROS)
        self.assertTrue(self.h.runner.zeros())
        self.assertEqual(len(self.h.bots), 2)

    def test_send_key_when_idle_is_error(self):
        with self.assertRaises(RuntimeError):
            self.h.runner.send_key()

    def test_shutdown_stops_running_program(self):
        self.h.start()
        self.assertTrue(wait_for(lambda: self.h.has_text('상태 FORWARD')))
        self.assertEqual(self.h.app.shutdown(), [])
        self.assertFalse(self.h.runner.running)
        self.assertTrue(self.h.has_text('조종판 종료'))
        self.assertEqual(self.h.bots[0].calls, EXPECTED_ZEROS)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), m.Handler)
        self.server.daemon_threads = True
        self.server.app = self.h.app
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.h.close()

    def get(self, path, headers=None):
        request = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=5) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read()

    def post(self, path, payload):
        request = urllib.request.Request(self.base + path, data=json.dumps(payload).encode('utf-8'),
                                         headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=20) as resp:
                return resp.status, json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode('utf-8'))

    def test_page_status_and_rejections(self):
        code, headers, body = self.get('/')
        self.assertEqual(code, 200)
        text = body.decode('utf-8')
        for needle in ('긴급 정지', '자동주행 ON', '정상 종료', '규칙표', 'EventSource', 'PERSONWAIT'):
            self.assertIn(needle, text)
        code, _, body = self.get('/api/status')
        status = json.loads(body)
        self.assertEqual(code, 200)
        self.assertFalse(status['running'])
        self.assertEqual(status['programs'], ['missions/fake_drive.py'])
        self.assertIn('others', status)
        code, j = self.post('/api/start', {'program': 'missions/fake_drive.py'})
        self.assertEqual(code, 400)
        self.assertIn('안전 확인', j['error'])
        code, j = self.post('/api/key', {})
        self.assertEqual(code, 409)
        code, _, _ = self.get('/nothing')
        self.assertEqual(code, 404)
        code, _, _ = self.get('/preview.mjpg')
        self.assertEqual(code, 409)

    def test_start_events_files_and_estop(self):
        code, j = self.post('/api/start', {'program': 'missions/fake_drive.py', 'trial': 'web1', 'args': '--trial web1', 'confirmed': True})
        self.assertEqual(code, 200)
        self.assertTrue(j['running'])
        code, j = self.post('/api/start', {'program': 'missions/fake_drive.py', 'confirmed': True})
        self.assertEqual(code, 409)
        self.assertTrue(wait_for(lambda: len(self.h.runner.rows) >= 1))
        # 실시간 이벤트: 첫 줄은 snapshot, 이어서 log/row 가 온다
        with urllib.request.urlopen(self.base + '/events', timeout=5) as resp:
            self.assertEqual(resp.headers['Content-Type'], 'text/event-stream; charset=utf-8')
            seen, data = set(), {}
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not {'snapshot', 'log', 'row'}.issubset(seen):
                line = resp.readline().decode('utf-8').rstrip('\n')
                if not line.startswith('event: '):
                    continue
                current = line[7:]
                payload = resp.readline().decode('utf-8').rstrip('\n')   # 바로 다음 줄이 data: 다
                self.assertTrue(payload.startswith('data: '), payload)
                seen.add(current)
                data.setdefault(current, json.loads(payload[6:]))
        self.assertTrue({'snapshot', 'log', 'row'}.issubset(seen), seen)
        self.assertTrue(data['snapshot']['running'])
        self.assertIn('log', data['snapshot'])
        self.assertEqual(data['row']['row']['x2'], '1')
        code, _, body = self.get('/files/fake_web1.csv')
        self.assertEqual(code, 200)
        self.assertIn('경과초,상태', body.decode('utf-8'))
        code, headers, body = self.get('/files/fake_web1.csv', {'Range': 'bytes=0-3'})
        self.assertEqual(code, 206)
        self.assertEqual(len(body), 4)
        self.assertTrue(headers['Content-Range'].startswith('bytes 0-3/'))
        code, _, _ = self.get('/files/../missions/fake_drive.py')
        self.assertEqual(code, 404)
        code, _, _ = self.get('/files/%2e%2e/missions/fake_drive.py')
        self.assertEqual(code, 404)
        code, j = self.post('/api/estop', {})
        self.assertEqual(code, 200)
        self.assertFalse(j['running'])
        self.assertEqual(j['exit_code'], 0)
        self.assertTrue(self.h.wait_done())
        code, _, body = self.get('/api/files')
        paths = [f['path'] for f in json.loads(body)['files']]
        self.assertIn('fake_web1.csv', paths)
        code, j = self.post('/api/zeros', {})
        self.assertEqual(code, 200)
        self.assertEqual(len(self.h.bots), 2)

    def test_preview_toggle_and_mjpeg(self):
        code, j = self.post('/api/preview', {'on': True})
        self.assertEqual(code, 200)
        self.assertTrue(j['preview'])
        request = urllib.request.Request(self.base + '/preview.mjpg')
        with urllib.request.urlopen(request, timeout=5) as resp:
            self.assertTrue(resp.headers['Content-Type'].startswith('multipart/x-mixed-replace'))
            chunk = resp.read(60)
            self.assertIn(b'--frame', chunk)
            self.assertIn(b'image/jpeg', chunk)
        code, j = self.post('/api/preview', {'on': False})
        self.assertFalse(j['preview'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
