#!/usr/bin/env python3
# 지시서: specs/09-dashboard.md
"""브라우저 조종판 — 자동주행 ON · 긴급 정지 · 실시간 로그.

주행 프로그램(missions/02~07)은 바꾸지 않고 가상 터미널(pty)에서 별도 프로세스로 띄운다.
조종판이 직접 보내는 바퀴 명령은 0(정지)뿐이다. 파이썬 표준 라이브러리만 쓴다(미리보기만 OpenCV).
실행: python3 missions/09_dashboard.py  →  브라우저에서 http://raspbot-03.local:8080
"""
import argparse
import codecs
from collections import deque
import contextlib
import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
import pty
import queue
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
KST = ZoneInfo('Asia/Seoul')
SELF_NAME = Path(__file__).name
DEFAULT_PROGRAM = 'missions/06_drive_signs.py'
# 프로그램마다 꼭 필요한 인자. {trial}은 회차 이름으로 바뀐다. 통합 프로그램이 바뀌면 여기만 고친다.
PROGRAM_ARGS = {'missions/06_drive_signs.py': ['--trial', '{trial}']}
ESTOP_GRACE = 2.0            # SIGINT 뒤 강제 종료까지 기다리는 시간
STOP_REPEATS = 3             # 02·04와 같은 정지 묶음
STOP_INTERVAL = 0.2
STOP_MOTOR_GAP = 0.01
LOG_KEEP = 2000
ROW_KEEP = 200
CSV_POLL = 0.25


class Bus:
    """구독자마다 대기열 하나. 느린 구독자는 넘치는 이벤트를 잃을 뿐 다른 구독자를 막지 않는다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queues = []

    def subscribe(self):
        q = queue.Queue(maxsize=2000)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    def publish(self, event):
        with self._lock:
            targets = list(self._queues)
        for q in targets:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


class LineSplitter:
    """바이트 조각을 UTF-8 줄로 바꾼다. 조각 경계에서 잘린 한글도 이어 붙인다."""

    def __init__(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.rest = ''

    def feed(self, data):
        text = self.rest + self.decoder.decode(data)
        lines = text.split('\n')
        self.rest = lines.pop()
        return [line.rstrip('\r') for line in lines]

    def flush(self):
        text = self.rest + self.decoder.decode(b'', final=True)
        self.rest = ''
        return [text.rstrip('\r')] if text else []


def real_bot():
    from Raspbot_Lib import Raspbot  # 라즈봇에만 있다
    return Raspbot()


def send_zeros(bot_factory, say):
    """조종판이 보내는 유일한 바퀴 명령: 네 바퀴 0 · 부저 끄기 · 초음파 끄기를 3회 (02·04와 같은 간격)."""
    try:
        bot = bot_factory()
    except ImportError:
        say('드라이버(Raspbot_Lib)가 없어 직접 정지 명령을 보내지 않았습니다. 라즈봇에서 실행하면 보냅니다.')
        return False
    except Exception as exc:
        say(f'직접 정지 실패 (드라이버 열기): {exc}')
        return False
    errors, captured = [], io.StringIO()
    with contextlib.redirect_stdout(captured):  # 드라이버는 예외 대신 'I2C error'를 출력한다
        for repeat in range(STOP_REPEATS):
            sent = time.monotonic()
            for number in range(4):
                try:
                    bot.Ctrl_Muto(number, 0)
                except Exception as exc:
                    errors.append(exc)
                if number < 3:
                    time.sleep(STOP_MOTOR_GAP)
            for switch in (bot.Ctrl_BEEP_Switch, bot.Ctrl_Ulatist_Switch):
                try:
                    switch(0)
                except Exception as exc:
                    errors.append(exc)
            if repeat < STOP_REPEATS - 1:
                time.sleep(max(0, sent + STOP_INTERVAL - time.monotonic()))
    printed = captured.getvalue().strip()
    if printed:
        say('드라이버 출력: ' + printed.replace('\n', ' / '))
    if errors:
        say(f'직접 정지 중 오류 {len(errors)}건: {errors[0]}')
        return False
    if 'I2C error' in printed:
        say('직접 정지 명령에 I2C 오류가 있었습니다. 확장보드 전원·I2C 선을 확인하고 바퀴를 눈으로 확인하세요.')
        return False
    say(f'네 바퀴 0 · 부저/초음파 끄기 {STOP_REPEATS}회 보냄 (조종판). 실제 정지는 눈으로 확인하세요.')
    return True


def other_drivers(exclude):
    """다른 터미널에서 이미 도는 missions/ 프로그램 목록 (경고용)."""
    try:
        out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.splitlines()[1:]:
        pid, _, args = line.strip().partition(' ')
        if not pid.isdigit() or int(pid) in exclude:
            continue
        if 'python' in args and 'missions/' in args and SELF_NAME not in args:
            found.append(f'{pid}: {args.strip()}')
    return found


class CsvTail(threading.Thread):
    """주행 프로그램이 results/에 새로 만든 원시 기록 CSV를 따라 읽어 줄마다 알린다."""

    def __init__(self, results, since, on_file, on_row):
        super().__init__(name='dashboard-csv', daemon=True)
        self.results, self.since = Path(results), since
        self.on_file, self.on_row = on_file, on_row
        self.stop = threading.Event()
        self.path = None

    def pick(self):
        candidates = []
        for path in self.results.glob('*.csv'):
            if path.name == 'runs.csv' or not path.is_file():
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= self.since:
                candidates.append((mtime, path.name, path))
        return max(candidates)[2] if candidates else None

    def run(self):
        stream = fields = None
        splitter = LineSplitter()
        try:
            while True:
                if stream is None:
                    self.path = self.pick()
                    if self.path is not None:
                        stream = self.path.open('rb')
                        self.on_file(self.path)
                if stream is not None:
                    data = stream.read()
                    for text in (splitter.feed(data) if data else []):
                        if not text.strip():
                            continue
                        values = next(csv.reader([text]))
                        if fields is None:
                            fields = values
                            continue
                        self.on_row(fields, dict(zip(fields, values)))
                if self.stop.is_set():
                    break
                self.stop.wait(CSV_POLL)
        finally:
            if stream is not None:
                stream.close()

    def close(self):
        self.stop.set()
        self.join(timeout=3)


class Runner:
    """주행 프로그램 하나를 pty로 띄우고 출력 · 종료 · 정지를 맡는다. 바퀴 명령은 0만 보낸다."""

    def __init__(self, root, bus, bot_factory=real_bot, python=None, grace=ESTOP_GRACE):
        self.root = Path(root).resolve()
        self.bus = bus
        self.bot_factory = bot_factory
        self.python = python or sys.executable
        self.grace = grace
        self.lock = threading.RLock()
        self.log = deque(maxlen=LOG_KEEP)
        self.rows = deque(maxlen=ROW_KEEP)
        self.fields = []
        self.seq = 0
        self.proc = None
        self.master = None
        self.logfile = None
        self.tail = None
        self.done = threading.Event()
        self.done.set()
        self.info = {'program': DEFAULT_PROGRAM, 'args': [], 'trial': '', 'started_at': None, 'ended_at': None,
                     'since': None, 'exit_code': None, 'exit_text': '', 'pid': None, 'log_file': None,
                     'csv_file': None, 'max_seconds': None, 'runs_last': None, 'i2c_errors': 0, 'zeros_sent': None}

    # ---- 로그 ----
    def say(self, text, src='dash'):
        with self.lock:
            self.seq += 1
            event = {'type': 'log', 'n': self.seq, 't': time.time(), 'src': src, 'text': text}
            self.log.append(event)
            if 'I2C error' in text:
                self.info['i2c_errors'] += 1
            if self.logfile is not None:
                try:
                    stamp = datetime.now(KST).strftime('%H:%M:%S.%f')[:-3]
                    self.logfile.write(f'{stamp} [{src}] {text}\n')
                    self.logfile.flush()
                except OSError:
                    pass
        self.bus.publish(event)

    # ---- 상태 ----
    def programs(self):
        folder = self.root / 'missions'
        return sorted(f'missions/{p.name}' for p in folder.glob('*.py') if p.name != SELF_NAME)

    @property
    def running(self):
        return self.proc is not None

    def status(self):
        with self.lock:
            info = dict(self.info)
        info.update(running=self.running, server_time=time.time(), programs=self.programs(),
                    default_program=DEFAULT_PROGRAM, program_args=PROGRAM_ARGS)
        return info

    def snapshot(self):
        with self.lock:
            return dict(self.status(), type='snapshot', log=list(self.log)[-400:],
                        fields=list(self.fields), rows=list(self.rows)[-50:])

    def files(self):
        since = self.info.get('since')
        if since is None:
            return []
        results = self.root / 'results'
        out = []
        for path in list(results.glob('*')) + list((results / 'videos').glob('*.mp4')):
            if not path.is_file() or path.name == 'runs.csv':
                continue
            if path.suffix.lower() not in ('.csv', '.log', '.mp4', '.json', '.txt'):
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_mtime >= since:
                out.append({'path': str(path.relative_to(results)), 'size': st.st_size, 'mtime': st.st_mtime})
        return sorted(out, key=lambda f: f['path'])

    # ---- 시작 ----
    def start(self, program, trial='', extra_args='', max_seconds=None, confirmed=False):
        if not confirmed:
            raise ValueError('바닥에 내려놓고 사방 1m를 비웠다는 안전 확인이 필요합니다.')
        if program not in self.programs():
            raise ValueError(f'알 수 없는 프로그램: {program}')
        trial = (trial or '').strip() or datetime.now(KST).strftime('dash_%H%M%S')
        try:
            extra = shlex.split(extra_args or '')
        except ValueError as exc:
            raise ValueError(f'추가 인자를 읽을 수 없습니다: {exc}')
        if max_seconds is not None:
            try:
                max_seconds = float(max_seconds)
            except (TypeError, ValueError):
                raise ValueError('최대 주행 시간은 숫자여야 합니다.')
            if not max_seconds > 0:
                raise ValueError('최대 주행 시간은 양수여야 합니다.')
        args = [a.format(trial=trial) for a in PROGRAM_ARGS.get(program, [])] + extra
        with self.lock:
            if self.running:
                raise RuntimeError('이미 주행 프로그램이 실행 중입니다. 먼저 정지하세요.')
            results = self.root / 'results'
            results.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
            log_path = results / f'dashboard_{stamp}.log'
            self.logfile = log_path.open('a', encoding='utf-8')
            self.rows.clear()
            self.fields = []
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', TERM='dumb')
            master, slave = pty.openpty()
            try:
                proc = subprocess.Popen([self.python, '-u', str(self.root / program), *args],
                                        cwd=str(self.root), stdin=slave, stdout=slave, stderr=slave,
                                        env=env, start_new_session=True, close_fds=True)
            except Exception:
                os.close(master)
                self.logfile.close()
                self.logfile = None
                raise
            finally:
                os.close(slave)
            self.proc, self.master = proc, master
            self.done.clear()
            now = time.time()
            self.info.update(program=program, args=args, trial=trial, started_at=now, ended_at=None,
                             since=now - 1, exit_code=None, exit_text='', pid=proc.pid,
                             log_file=str(log_path.relative_to(self.root)), csv_file=None,
                             max_seconds=max_seconds, runs_last=None, i2c_errors=0, zeros_sent=None)
            self.tail = CsvTail(results, now - 1, self._on_file, self._on_row)
            self.tail.start()
            reader = threading.Thread(target=self._read, args=(master,), name='dashboard-read', daemon=True)
            reader.start()
            threading.Thread(target=self._wait, args=(proc, reader), name='dashboard-wait', daemon=True).start()
            if max_seconds:
                threading.Thread(target=self._watchdog, args=(proc, max_seconds),
                                 name='dashboard-watchdog', daemon=True).start()
        self.say('──────── 새 주행 ────────')
        self.say(f'시작: python3 {program} {" ".join(args)} (PID {proc.pid}, 회차 {trial})')
        self.bus.publish(dict(self.status(), type='status'))
        return self.status()

    def _read(self, master):
        splitter = LineSplitter()
        while True:
            try:
                data = os.read(master, 65536)
            except OSError:
                break  # 리눅스: 자식이 닫으면 EIO
            if not data:
                break  # macOS: EOF
            for line in splitter.feed(data):
                self.say(line, 'child')
        for line in splitter.flush():
            self.say(line, 'child')

    def _wait(self, proc, reader):
        code = proc.wait()
        reader.join(timeout=3)
        tail = self.tail
        if tail is not None:
            tail.close()
        text = self._exit_text(code)
        self.say(f'주행 프로그램 종료: {text}')
        ok = send_zeros(self.bot_factory, self.say)
        runs_last = self._runs_last()
        with self.lock:
            self.proc = None
            if self.master is not None:
                try:
                    os.close(self.master)
                except OSError:
                    pass
                self.master = None
            self.info.update(exit_code=code, exit_text=text, pid=None, runs_last=runs_last,
                             zeros_sent=ok, ended_at=time.time())
            if self.logfile is not None:
                self.logfile.close()
                self.logfile = None
        self.done.set()
        self.bus.publish(dict(self.status(), type='status'))

    @staticmethod
    def _exit_text(code):
        if code is None:
            return '알 수 없음'
        if code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = str(-code)
            return f'신호 {name}로 종료'
        return f'종료 코드 {code}' + (' (정상)' if code == 0 else ' (오류)')

    def _runs_last(self):
        path = self.root / 'results' / 'runs.csv'
        try:
            if path.stat().st_mtime < self.info['since']:
                return None
            with path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, csv.Error, TypeError):
            return None
        return rows[-1] if rows else None

    def _on_file(self, path):
        rel = str(path.relative_to(self.root))
        with self.lock:
            self.info['csv_file'] = rel
        self.say(f'원시 기록 따라 읽기: {rel}')
        self.bus.publish({'type': 'file', 'csv_file': rel})

    def _on_row(self, fields, row):
        with self.lock:
            self.fields = list(fields)
            self.rows.append(row)
        self.bus.publish({'type': 'row', 'fields': list(fields), 'row': row})

    def _watchdog(self, proc, max_seconds):
        if self.done.wait(timeout=max_seconds):
            return
        with self.lock:
            if self.proc is not proc:
                return
        self.estop(f'최대 주행 시간 {max_seconds:g}초 초과')

    # ---- 정지 ----
    def send_key(self, key=b' '):
        with self.lock:
            if not self.running:
                raise RuntimeError('주행 중이 아닙니다.')
            os.write(self.master, key)
        self.say('키 입력을 보냈습니다 (아무 키나 정지). 프로그램이 스스로 정지·정리합니다.')

    @staticmethod
    def _signal(proc, sig):
        try:
            os.killpg(proc.pid, sig)  # start_new_session이라 프로세스 그룹 번호 = PID
        except ProcessLookupError:
            pass
        except PermissionError:
            proc.send_signal(sig)

    def estop(self, reason='긴급 정지 버튼'):
        with self.lock:
            proc = self.proc
        if proc is None:
            self.say(f'{reason}: 주행 중이 아님 → 바퀴 0만 다시 보냅니다.')
            ok = send_zeros(self.bot_factory, self.say)
            with self.lock:
                self.info['zeros_sent'] = ok
            self.bus.publish(dict(self.status(), type='status'))
            return self.status()
        self.say(f'{reason}: 주행 프로그램(PID {proc.pid})에 SIGINT(Ctrl+C) 보냄')
        self._signal(proc, signal.SIGINT)
        try:
            proc.wait(timeout=self.grace)
        except subprocess.TimeoutExpired:
            self.say(f'{self.grace:g}초 안에 끝나지 않아 강제 종료(SIGKILL)합니다.')
            self._signal(proc, signal.SIGKILL)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.say('강제 종료도 응답이 없습니다. 확장보드 전원 스위치로 멈추세요.')
        self.done.wait(timeout=10)  # _wait 스레드가 바퀴 0까지 보낸 뒤 set
        return self.status()

    def zeros(self):
        with self.lock:
            if self.running:
                raise RuntimeError('주행 중에는 보내지 않습니다. 긴급 정지를 사용하세요.')
        ok = send_zeros(self.bot_factory, self.say)
        with self.lock:
            self.info['zeros_sent'] = ok
        self.bus.publish(dict(self.status(), type='status'))
        return ok


class Preview:
    """주행 중이 아닐 때만 쓰는 카메라 미리보기. 주행 시작 전에 반드시 끈다."""

    def __init__(self, root, open_camera=None, encode=None):
        self.root = Path(root)
        self._open_camera, self._encode = open_camera, encode
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.latest = None
        self.error = None

    def _defaults(self):
        if self._open_camera is None or self._encode is None:
            sys.path.insert(0, str(self.root / 'tools'))
            import cv2
            from yolo import open_camera
            self._open_camera = self._open_camera or open_camera
            self._encode = self._encode or (lambda frame: cv2.imencode(
                '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1].tobytes())

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        with self.lock:
            if self.running:
                return
            self._defaults()
            self.stop.clear()
            self.latest = None
            self.error = None
            self.thread = threading.Thread(target=self._run, name='dashboard-preview', daemon=True)
            self.thread.start()

    def _run(self):
        cap = None
        try:
            cap = self._open_camera()
            if hasattr(cap, 'isOpened') and not cap.isOpened():
                raise RuntimeError('카메라를 열지 못했습니다. USB 연결을 확인하세요.')
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.stop.wait(0.05)
                    continue
                data = self._encode(frame)
                with self.lock:
                    self.latest = data
                self.stop.wait(0.1)
        except Exception as exc:
            self.error = exc
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def jpeg(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        thread = self.thread
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError('미리보기 카메라가 닫히지 않았습니다. 잠시 뒤 다시 시도하세요.')
        self.thread = None


class App:
    def __init__(self, root=ROOT, bot_factory=real_bot, python=None, preview=None, grace=ESTOP_GRACE):
        self.root = Path(root).resolve()
        self.bus = Bus()
        self.runner = Runner(self.root, self.bus, bot_factory, python, grace)
        self.preview = preview if preview is not None else Preview(self.root)

    def status(self):
        status = self.runner.status()
        status.update(preview=self.preview.running,
                      preview_error='' if self.preview.error is None else str(self.preview.error),
                      others=other_drivers({os.getpid(), status.get('pid') or -1}),
                      host=host_name())
        return status

    def start(self, payload):
        if self.preview.running:
            self.preview.close()
            self.runner.say('미리보기를 끄고 주행 프로그램을 띄웁니다.')
        return self.runner.start(program=payload.get('program') or DEFAULT_PROGRAM,
                                 trial=payload.get('trial') or '', extra_args=payload.get('args') or '',
                                 max_seconds=payload.get('max_seconds') or None,
                                 confirmed=bool(payload.get('confirmed')))

    def set_preview(self, on):
        if on:
            if self.runner.running:
                raise RuntimeError('주행 중에는 미리보기를 켤 수 없습니다 (주행 프로그램이 카메라를 씁니다).')
            self.preview.start()
        else:
            self.preview.close()
        self.bus.publish(dict(self.status(), type='status'))
        return self.preview.running

    def shutdown(self):
        errors = []
        if self.runner.running:
            try:
                self.runner.estop('조종판 종료')
            except Exception as exc:
                errors.append(exc)
        try:
            self.preview.close()
        except Exception as exc:
            errors.append(exc)
        return errors


CONTENT_TYPES = {'.csv': 'text/csv; charset=utf-8', '.log': 'text/plain; charset=utf-8',
                 '.txt': 'text/plain; charset=utf-8', '.json': 'application/json; charset=utf-8',
                 '.mp4': 'video/mp4', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}


class Handler(BaseHTTPRequestHandler):
    server_version = 'RaspbotDashboard/1.0'

    @property
    def app(self):
        return self.server.app

    def log_request(self, code='-', size='-'):
        pass  # 터미널에는 오류만 남긴다

    # ---- 공통 ----
    def _send(self, body, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(json.dumps(obj, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8', status)

    def _body(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise ValueError('JSON 본문을 읽을 수 없습니다.')
        if not isinstance(payload, dict):
            raise ValueError('JSON 객체가 필요합니다.')
        return payload

    # ---- 주소 ----
    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == '/':
                return self._send(PAGE.encode('utf-8'), 'text/html; charset=utf-8')
            if path == '/events':
                return self._events()
            if path == '/api/status':
                return self._json(self.app.status())
            if path == '/api/files':
                return self._json({'files': self.app.runner.files()})
            if path == '/preview.mjpg':
                return self._mjpeg()
            if path.startswith('/files/'):
                return self._file(unquote(path[len('/files/'):]))
        except (BrokenPipeError, ConnectionResetError):
            return
        self._json({'error': '없는 주소'}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            payload = self._body()
            if path == '/api/start':
                return self._json(self.app.start(payload))
            if path == '/api/estop':
                return self._json(self.app.runner.estop())
            if path == '/api/key':
                self.app.runner.send_key()
                return self._json(self.app.status())
            if path == '/api/zeros':
                self.app.runner.zeros()
                return self._json(self.app.status())
            if path == '/api/preview':
                return self._json({'preview': self.app.set_preview(bool(payload.get('on')))})
        except ValueError as exc:
            return self._json({'error': str(exc)}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            return self._json({'error': str(exc)}, HTTPStatus.CONFLICT)
        except (BrokenPipeError, ConnectionResetError):
            return
        self._json({'error': '없는 주소'}, HTTPStatus.NOT_FOUND)

    # ---- 실시간 이벤트 (SSE) ----
    def _event(self, event):
        self.wfile.write(f'event: {event["type"]}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n'.encode('utf-8'))
        self.wfile.flush()

    def _events(self):
        q = self.app.bus.subscribe()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Accel-Buffering', 'no')
            self.end_headers()
            snapshot = self.app.runner.snapshot()
            snapshot.update(preview=self.app.preview.running,
                            preview_error='' if self.app.preview.error is None else str(self.app.preview.error),
                            others=other_drivers({os.getpid(), snapshot.get('pid') or -1}),
                            host=host_name())
            self._event(snapshot)
            while True:
                try:
                    event = q.get(timeout=10)
                except queue.Empty:
                    self.wfile.write(b': ping\n\n')
                    self.wfile.flush()
                    continue
                self._event(event)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.app.bus.unsubscribe(q)

    # ---- 미리보기 (MJPEG) ----
    def _mjpeg(self):
        preview = self.app.preview
        if not preview.running:
            return self._json({'error': '미리보기가 꺼져 있습니다.'}, HTTPStatus.CONFLICT)
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        last = None
        try:
            while preview.running:
                data = preview.jpeg()
                if data is None or data is last:
                    time.sleep(0.05)
                    continue
                last = data
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                 + str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    # ---- results/ 파일 (CSV · 로그 · 영상) ----
    def _file(self, rel):
        results = (self.app.root / 'results').resolve()
        target = (results / rel).resolve()
        if not target.is_relative_to(results) or not target.is_file():
            return self._json({'error': '파일이 없습니다.'}, HTTPStatus.NOT_FOUND)
        size = target.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range') or '')
        if match and size:
            first, last = match.groups()
            if first:
                start, end = int(first), (int(last) if last else size - 1)
            elif last:
                start = max(0, size - int(last))
            end = min(end, size - 1)
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header('Content-Type', CONTENT_TYPES.get(target.suffix.lower(), 'application/octet-stream'))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(max(0, end - start + 1)))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with target.open('rb') as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>라즈봇 조종판</title>
<style>
:root{--bg:#f3f4f6;--card:#fff;--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--green:#15803d;--red:#dc2626;--amber:#d97706;--orange:#ea580c;--gray:#6b7280;--blue:#2563eb}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",sans-serif}
header{display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:10px 16px;background:#111827;color:#fff}
header h1{font-size:18px;margin:0}
.dot{width:10px;height:10px;border-radius:50%;background:#9ca3af;display:inline-block}
.dot.on{background:#22c55e}
.dot.off{background:#ef4444}
.badge{border-radius:999px;padding:2px 10px;font-size:13px;background:#4b5563}
.badge.run{background:#16a34a}
main{display:grid;gap:12px;padding:12px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}
.card.wide{grid-column:1/-1}
.hd{display:flex;align-items:center;justify-content:space-between;gap:8px;font-weight:700;margin-bottom:8px}
.muted{color:var(--muted);font-size:13px}
.row{margin:8px 0}
label{display:block;font-size:14px;margin:6px 0}
input[type=text],input[type=number],select{width:100%;padding:8px;border:1px solid #d1d5db;border-radius:8px;font-size:15px}
.btn{display:block;width:100%;border:0;border-radius:12px;padding:14px;font-size:17px;font-weight:700;color:#fff;cursor:pointer}
.btn:disabled{opacity:.35;cursor:not-allowed}
.btn.start{background:var(--green)}
.btn.estop{background:var(--red);font-size:26px;padding:22px;letter-spacing:1px}
.btn.gray{background:#4b5563;font-size:14px;padding:10px}
.btn.blue{background:var(--blue);font-size:14px;padding:10px}
.btn.mini{display:inline;width:auto;padding:6px 10px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.state{font-size:38px;font-weight:800;line-height:1.1;word-break:break-all}
.state.green{color:var(--green)} .state.red{color:var(--red)} .state.amber{color:var(--amber)} .state.orange{color:var(--orange)} .state.gray{color:var(--gray)}
.kv{font-size:14px;margin:6px 0;word-break:break-all}
.sensor{display:flex;gap:6px;margin-top:4px}
.sensor div{width:46px;height:46px;border:2px solid #374151;border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px}
.sensor .black{background:#111;color:#fff}
.sensor .white{background:#fff;color:#111}
.wheels{display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:260px;margin-top:4px}
.wheels div{border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:13px}
.wheels b{font-size:18px}
.bar{height:6px;background:#e5e7eb;border-radius:3px;margin-top:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--blue)}
.seen{display:inline-block;background:#dbeafe;border-radius:6px;padding:2px 8px;margin:2px}
.log{height:380px;overflow:auto;background:#0b1020;color:#d1d5db;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;padding:10px;border-radius:8px;white-space:pre-wrap;word-break:break-all}
.log .err{color:#f87171}
.log .st{color:#86efac}
.log .dash{color:#93c5fd}
.log .tm{color:#64748b}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;white-space:nowrap}
.tablewrap{overflow:auto;max-height:320px}
#ruletable td:first-child{font-weight:700;font-family:ui-monospace,Menlo,monospace}
#ruletable tr.now{background:#fef3c7}
#ruletable td{white-space:normal}
.warn{background:#fef2f2;color:#991b1b;border:1px solid #fecaca;border-radius:8px;padding:8px;font-size:13px;margin:8px 0}
#previewimg{width:100%;border-radius:8px;background:#000;min-height:120px;margin-top:8px}
ul{padding-left:18px;font-size:14px}
hr{border:0;border-top:1px solid var(--line);margin:14px 0}
</style>
</head>
<body>
<header>
<h1>라즈봇 자율주행 조종판</h1>
<span class="dot" id="conn" title="조종판 연결"></span>
<span id="host" class="muted"></span>
<span id="runbadge" class="badge">대기</span>
</header>
<main>
<section class="card">
  <button class="btn estop" id="estop">■ 긴급 정지</button>
  <div class="muted" style="margin-top:6px">누르는 즉시 주행 프로그램에 Ctrl+C를 보내고, 끝난 뒤 조종판이 네 바퀴 0을 한 번 더 보냅니다. 키보드 Esc도 같습니다.</div>
  <hr>
  <label><input type="checkbox" id="safe"> 라즈봇을 바닥(출발점)에 놓고 사방 1m를 비웠습니다</label>
  <label>프로그램 <select id="program"></select></label>
  <label>회차 이름 <input type="text" id="trial" placeholder="비우면 dash_시분초"></label>
  <div class="two">
    <label>추가 인자 <input type="text" id="args" placeholder="예: --seconds 10"></label>
    <label>최대 시간(초) <input type="number" id="max" min="1" placeholder="없음"></label>
  </div>
  <button class="btn start" id="start">▶ 자동주행 ON</button>
  <div class="two" style="margin-top:8px">
    <button class="btn gray" id="key">정상 종료 (아무 키)</button>
    <button class="btn gray" id="zeros">바퀴 0 다시 보내기</button>
  </div>
  <div id="others"></div>
  <div class="kv" id="runinfo"></div>
</section>
<section class="card">
  <div class="muted">현재 상태</div>
  <div class="state gray" id="state">—</div>
  <div class="muted" id="statedesc">주행 프로그램이 시작되면 여기에 상태가 나옵니다.</div>
  <div class="kv">경과 <b id="elapsed">0.0</b>초 · PID <span id="pid">—</span> · I2C 오류 <b id="i2c">0</b>회</div>
  <div class="row"><div class="muted">라인센서 — 왼쪽부터 실제 배치 x2 · x1 · x3 · x4 (검정 = 줄 감지)</div><div class="sensor" id="sensors"></div></div>
  <div class="row"><div class="muted">바퀴 명령 (왼앞 · 오앞 / 왼뒤 · 오뒤)</div><div class="wheels" id="wheels"></div></div>
  <div class="row"><div class="muted">본 것 (YOLO)</div><div id="seen">—</div></div>
  <div class="row"><div class="muted">초음파 거리</div><div id="dist">—</div></div>
</section>
<section class="card">
  <div class="hd">규칙표 — 상태 이름과 뜻 <span class="muted">(현재 상태 강조)</span></div>
  <div class="tablewrap"><table id="ruletable"></table></div>
</section>
<section class="card wide">
  <div class="hd">주행 로그 <span><label style="display:inline"><input type="checkbox" id="autoscroll" checked> 자동 스크롤</label> <button class="btn gray mini" id="clear">지우기</button></span></div>
  <div class="log" id="log"></div>
</section>
<section class="card wide">
  <div class="hd">원시 기록 (CSV) 최근 줄 <span class="muted" id="csvname"></span></div>
  <div class="tablewrap"><table id="rows"></table></div>
</section>
<section class="card">
  <div class="hd">카메라 미리보기 <span class="muted">주행 중이 아닐 때만</span></div>
  <button class="btn blue" id="preview">미리보기 켜기</button>
  <img id="previewimg" alt="">
  <div class="muted" id="previewmsg"></div>
</section>
<section class="card">
  <div class="hd">이번 주행 파일 <button class="btn gray mini" id="refresh">새로고침</button></div>
  <div id="runslast" class="muted"></div>
  <ul id="files"></ul>
</section>
</main>
<script>
const $ = id => document.getElementById(id);
// [상태 이름, 색, 언제, 무엇을] — specs/02·04·06·07 규칙표 요약
const RULES = [
  ['WAITVISION','gray','첫 YOLO 인식 결과를 기다림 (06)','정지'],
  ['STALEVISION','gray','인식 결과가 1초 넘게 안 옴 (06)','정지, 새 결과가 오면 재출발'],
  ['READY','gray','시작 직전 (02)','정지'],
  ['FORWARD','green','가운데 두 칸(x1·x3)이 검정','직진 50'],
  ['HOLD','green','양쪽이 모두 검정 (교차점)','직전 바퀴 명령 유지'],
  ['LEFTGENTLE','amber','줄이 왼쪽으로 살짝 치우침','왼쪽 꺾기 안쪽 30 / 바깥 60'],
  ['LEFTMEDIUM','amber','왼쪽 두 칸이 검정','왼쪽 꺾기 15 / 60'],
  ['LEFTSHARP','amber','맨 왼쪽 칸만 검정','왼쪽 꺾기 0 / 60'],
  ['RIGHTGENTLE','amber','줄이 오른쪽으로 살짝 치우침','오른쪽 꺾기 30 / 60'],
  ['RIGHTMEDIUM','amber','오른쪽 두 칸이 검정','오른쪽 꺾기 15 / 60'],
  ['RIGHTSHARP','amber','맨 오른쪽 칸만 검정','오른쪽 꺾기 0 / 60'],
  ['SEARCHLEFT','orange','줄을 놓침, 마지막으로 왼쪽으로 꺾던 중','왼쪽으로 0 / 40 탐색 (최대 1초)'],
  ['SEARCHRIGHT','orange','줄을 놓침, 마지막으로 오른쪽으로 꺾던 중','오른쪽으로 0 / 40 탐색 (최대 1초)'],
  ['LOSTSTOP','red','1초 넘게 줄이 없음 (또는 꺾은 적 없음)','정지. 프로그램을 다시 실행하기 전엔 재출발 없음'],
  ['PERSONWAIT','red','사람을 새 결과에서 연속 2회 인식 (06)','정지. 새 결과에서 1초 안 보이면 재출발'],
  ['STOPWAIT','red','STOP 표지를 연속 2회 인식 (06)','3초 정지 후 재출발, 재출발 뒤 5초 동안 STOP 무시'],
  ['WAITFRAME','gray','첫 카메라 화면 대기 (07)','정지'],
  ['TRACKBOTH','green','노란 선 둘 다 보임 (07)','차선 중심으로 비례 조향 (기본 35, K 0.4, 최대 80)'],
  ['TRACKLEFT','green','왼쪽 선만 보임 (07)','마지막 차선 폭으로 중심 추정 후 조향'],
  ['TRACKRIGHT','green','오른쪽 선만 보임 (07)','마지막 차선 폭으로 중심 추정 후 조향'],
  ['WAITWIDTH','gray','한쪽 선만 보이는데 폭을 모름 (07)','두 선이 보일 때까지 정지 대기'],
  ['FRAMESTOP','red','카메라 화면이 0.5초 넘게 안 바뀜 (07)','정지, 재실행 전 재출발 없음'],
  ['전진','green','초음파 중앙값 35cm 이상 (04)','속도 50 전진'],
  ['거리 조건 정지','red','초음파 중앙값 25cm 이하 (04)','정지 + 삐 0.2초, 35cm 이상이면 재출발'],
  ['정상값 대기','gray','유효한 측정 3개가 모이기 전 (04)','정지'],
  ['연속 오류 정지','red','초음파 오류 3회 연속 (04)','정지'],
  ['WAITLINES','gray','시작 직후 노란 선을 아직 못 봄 (11)','최대 3초 정지 대기, 못 보면 LOSTSTOP'],
  ['WAITLANE','green','로터리 출구 뒤, 흰 실선 전 (11)','07 규칙대로 차선 가운데 주행'],
  ['RIGHTLANE','green','흰 실선 연속 2회 확인 (11)','흰 실선과 오른쪽 노란 선 사이(오른쪽 차선) 중심으로 주행'],
  ['TURNRIGHT','amber','흰 실선이 0.3초 넘게 안 보임 (11)','오른쪽 노란 선 따라 우회전, 선 없으면 호 그리기 55/10'],
  ['FINISHRUN','green','우회전 뒤 큰 길 두 선 확인 (11)','07 규칙대로 주행하며 START 무늬(흰 비율 0.25) 기다림'],
  ['FINISH','red','START 무늬 연속 2회 (11)','정지 + 삐 두 번 + 초록 불빛. 재실행 전 재출발 없음'],
  ['TIMEOUT','red','우회전 뒤 8초 안에 START를 못 봄 (11)','정지, 기록에 시간 초과'],
];
// 규칙표에 없는 새 상태(우리 조 기능 등)는 이름으로 짐작한다
const GENERIC = [
  [/ROTARY|로터리/i,'amber','로터리(회전교차로) 구간','감속 주행 (우리 조 기능)'],
  [/SLOW|천천히/i,'amber','바닥 「천천히」 글씨 구간','감속 주행 (우리 조 기능)'],
  [/OBSTACLE|장애물/i,'red','장애물 인식','정지'],
  [/STOP|정지/i,'red','정지 조건','정지'],
  [/SEARCH|탐색/i,'orange','줄 탐색','탐색'],
  [/WAIT|대기/i,'gray','대기','정지'],
  [/LEFT|RIGHT/i,'amber','꺾기','꺾기'],
  [/FORWARD|TRACK|전진/i,'green','주행','전진'],
];
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const norm = s => String(s).replace(/_/g,'').trim();
function ruleFor(s){
  const n = norm(s);
  for (const r of RULES) if (norm(r[0]) === n) return r;
  for (const g of GENERIC) if (g[0].test(n)) return [s, g[1], g[2], g[3]];
  return [s, 'gray', '(규칙표에 없는 상태)', ''];
}
function renderRules(){
  $('ruletable').innerHTML = '<tr><th>상태</th><th>언제</th><th>무엇을</th></tr>' +
    RULES.map(r => `<tr data-state="${esc(norm(r[0]))}"><td>${esc(r[0])}</td><td>${esc(r[2])}</td><td>${esc(r[3])}</td></tr>`).join('');
}
function setState(s){
  if (s === undefined || s === null || s === '') return;
  const [, color, when, what] = ruleFor(s);
  const el = $('state'); el.textContent = s; el.className = 'state ' + color;
  $('statedesc').textContent = when + ' → ' + what;
  document.querySelectorAll('#ruletable tr').forEach(tr => tr.classList.toggle('now', tr.dataset.state === norm(s)));
}
function setSensors(bits){
  if (!bits || bits.length < 4) return;
  $('sensors').innerHTML = [1,0,2,3].map(i => { const v = String(bits[i]).trim(); return `<div class="${v==='0'?'black':'white'}">x${i+1}</div>`; }).join('');
}
function toWheels(v){
  if (Array.isArray(v)) return v.map(Number);
  if (typeof v === 'number') return [v,v,v,v];
  try { const p = JSON.parse(v); return Array.isArray(p) ? p.map(Number) : [p,p,p,p].map(Number); } catch(e) {}
  const m = String(v).match(/-?\d+/g); if (!m) return null;
  return m.length >= 4 ? m.slice(0,4).map(Number) : Array(4).fill(Number(m[0]));
}
function setWheels(v){
  const arr = toWheels(v); if (!arr) return;
  const names = ['왼앞','왼뒤','오앞','오뒤'];
  $('wheels').innerHTML = [0,2,1,3].map(i => `<div>${names[i]} <b>${esc(arr[i])}</b><div class="bar"><i style="width:${Math.min(100, Math.abs(arr[i]) / 150 * 100)}%"></i></div></div>`).join('');
}
function setSeen(names, scores, heights){
  const n = String(names || '').split(';').filter(Boolean);
  if (!n.length) { $('seen').textContent = '없음'; return; }
  const s = String(scores || '').split(';'), h = String(heights || '').split(';');
  $('seen').innerHTML = n.map((x, i) => `<span class="seen"><b>${esc(x)}</b> ${s[i] ? Number(s[i]).toFixed(2) : ''} ${h[i] ? 'H' + esc(h[i]) : ''}</span>`).join('');
}
let rowFields = null;
function addRowToTable(fields, row){
  const t = $('rows');
  if (!rowFields || rowFields.join() !== fields.join()) { rowFields = fields; t.innerHTML = '<tr>' + fields.map(f => `<th>${esc(f)}</th>`).join('') + '</tr>'; }
  const tr = document.createElement('tr');
  tr.innerHTML = fields.map(f => `<td>${esc(row[f] ?? '')}</td>`).join('');
  t.insertBefore(tr, t.rows[1] || null);
  while (t.rows.length > 31) t.deleteRow(t.rows.length - 1);
}
function applyRow(fields, row){
  const st = row['상태'] ?? row['state'];
  if (st !== undefined) setState(st);
  if (['x1','x2','x3','x4'].every(k => row[k] !== undefined)) setSensors([row.x1, row.x2, row.x3, row.x4]);
  const w = row['바퀴속도'] ?? row['바퀴값'] ?? row['speed'];
  if (w !== undefined && w !== '') setWheels(w);
  if (row['인식이름'] !== undefined) setSeen(row['인식이름'], row['점수'], row['상자높이_px']);
  if (row['raw_cm'] !== undefined) $('dist').textContent = `${row.raw_cm} cm (중앙값 ${row.median_cm || '대기'})`;
  addRowToTable(fields, row);
}
function parseLine(text){
  let m;
  if ((m = text.match(/^상태 (\S+?), 센서 \(([^)]*)\), 바퀴 \(([^)]*)\)/))) { setState(m[1]); setSensors(m[2].split(',')); setWheels('[' + m[3] + ']'); return; }
  if ((m = text.match(/x1\.\.x4=\(([^)]*)\) (\S+) wheels=\(([^)]*)\)/))) { setSensors(m[1].split(',')); setState(m[2]); setWheels('[' + m[3] + ']'); return; }
  if ((m = text.match(/차선 계산: (\{.*\})$/))) { try { const j = JSON.parse(m[1]); if (j.state) setState(j.state); if (j.speeds) setWheels(j.speeds); } catch(e) {} return; }
  if ((m = text.match(/^거리 ([\d.]+)cm \/ 중앙값 (\S+)cm \/ (.+?) \/ 속도 (\d+)/))) { $('dist').textContent = `${m[1]} cm (중앙값 ${m[2]})`; setState(m[3]); setWheels(Number(m[4])); }
}
function addLog(e){
  const box = $('log');
  const div = document.createElement('div');
  let cls = e.src === 'dash' ? 'dash' : '';
  if (/I2C error|Traceback|Error|오류|실패/.test(e.text)) cls = 'err';
  else if (/^상태 |wheels=|차선 계산|^거리 /.test(e.text)) cls = 'st';
  div.className = cls;
  div.innerHTML = `<span class="tm">${new Date(e.t * 1000).toLocaleTimeString('ko-KR', {hour12: false})}</span> ${esc(e.text)}`;
  box.appendChild(div);
  while (box.childElementCount > 2000) box.removeChild(box.firstChild);
  if ($('autoscroll').checked) box.scrollTop = box.scrollHeight;
  parseLine(e.text);
}
let status = {}, offset = 0;
function applyStatus(s){
  status = Object.assign(status, s);
  const running = !!status.running;
  $('start').disabled = running;
  $('key').disabled = !running;
  $('zeros').disabled = running;
  $('preview').disabled = running;
  $('runbadge').textContent = running ? '주행 중 · PID ' + status.pid : '대기';
  $('runbadge').className = 'badge' + (running ? ' run' : '');
  $('pid').textContent = status.pid ?? '—';
  $('i2c').textContent = status.i2c_errors ?? 0;
  $('host').textContent = status.host ? status.host + '.local' : '';
  if (status.programs && !$('program').options.length) {
    for (const p of status.programs) { const o = document.createElement('option'); o.value = p; o.textContent = p; if (p === status.default_program) o.selected = true; $('program').appendChild(o); }
  }
  const info = [];
  if (status.program && status.started_at) info.push(esc(status.program + ' ' + (status.args || []).join(' ')));
  if (status.exit_text && !running) info.push('종료: ' + esc(status.exit_text));
  if (status.zeros_sent === false && !running) info.push('⚠ 조종판의 직접 정지 명령이 실패했거나 생략됐습니다 — 로그를 확인하세요');
  if (status.log_file) info.push(`로그 파일: <a href="/files/${encodeURI(status.log_file.replace(/^results\//, ''))}" target="_blank">${esc(status.log_file)}</a>`);
  $('runinfo').innerHTML = info.join('<br>');
  $('others').innerHTML = (status.others && status.others.length) ? '<div class="warn">다른 주행 프로그램이 이미 실행 중입니다 (터미널에서 끄거나 그대로 두고 판단하세요):<br>' + status.others.map(esc).join('<br>') + '</div>' : '';
  $('csvname').textContent = status.csv_file || '';
  $('preview').textContent = status.preview ? '미리보기 끄기' : '미리보기 켜기';
  if (status.preview) { if (!$('previewimg').getAttribute('src')) $('previewimg').src = '/preview.mjpg?' + Date.now(); }
  else { $('previewimg').removeAttribute('src'); }
  $('previewmsg').textContent = status.preview_error || '';
  if (!running && status.started_at && status.ended_at) $('elapsed').textContent = (status.ended_at - status.started_at).toFixed(1);
  if (!running && status.started_at) loadFiles();
  $('runslast').textContent = status.runs_last ? 'runs.csv 마지막 줄: ' + Object.entries(status.runs_last).filter(([k, v]) => v).map(([k, v]) => k + '=' + v).join(' · ') : '';
}
setInterval(() => { if (status.running && status.started_at) $('elapsed').textContent = (Date.now() / 1000 - offset - status.started_at).toFixed(1); }, 100);
async function loadFiles(){
  try {
    const r = await fetch('/api/files'); const j = await r.json();
    $('files').innerHTML = (j.files || []).map(f => `<li><a href="/files/${encodeURI(f.path)}" target="_blank">${esc(f.path)}</a> <span class="muted">${(f.size / 1024).toFixed(0)} KB</span></li>`).join('') || '<li class="muted">아직 없음</li>';
  } catch(e) {}
}
async function post(url, body){
  let r, j = {};
  try { r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})}); }
  catch(e) { alert('조종판에 연결할 수 없습니다: ' + e); return {error: String(e)}; }
  try { j = await r.json(); } catch(e) {}
  if (!r.ok) alert(j.error || ('오류 ' + r.status));
  return j;
}
function connect(){
  const es = new EventSource('/events');
  es.onopen = () => { $('conn').className = 'dot on'; };
  es.onerror = () => { $('conn').className = 'dot off'; };
  es.addEventListener('snapshot', ev => {
    const s = JSON.parse(ev.data);
    offset = Date.now() / 1000 - s.server_time;
    $('log').innerHTML = '';
    (s.log || []).forEach(addLog);
    rowFields = null; $('rows').innerHTML = '';
    if (s.fields && s.fields.length) (s.rows || []).forEach(r => applyRow(s.fields, r));
    applyStatus(s);
  });
  es.addEventListener('log', ev => addLog(JSON.parse(ev.data)));
  es.addEventListener('row', ev => { const e = JSON.parse(ev.data); applyRow(e.fields, e.row); });
  es.addEventListener('status', ev => { const s = JSON.parse(ev.data); offset = Date.now() / 1000 - s.server_time; applyStatus(s); });
  es.addEventListener('file', ev => { $('csvname').textContent = JSON.parse(ev.data).csv_file; });
}
$('start').onclick = async () => {
  if (!$('safe').checked) { alert('먼저 「바닥에 놓고 사방 1m 비움」 안전 확인을 체크해 주세요.'); return; }
  $('start').disabled = true;
  const j = await post('/api/start', {program: $('program').value, trial: $('trial').value, args: $('args').value,
                                      max_seconds: $('max').value ? Number($('max').value) : null, confirmed: true});
  if (j.error) $('start').disabled = false;
  $('safe').checked = false;
};
$('estop').onclick = () => { $('estop').textContent = '■ 정지 중…'; post('/api/estop').finally(() => { $('estop').textContent = '■ 긴급 정지'; }); };
document.addEventListener('keydown', e => { if (e.key === 'Escape') $('estop').onclick(); });
$('key').onclick = () => post('/api/key');
$('zeros').onclick = () => post('/api/zeros');
$('preview').onclick = () => post('/api/preview', {on: !status.preview});
$('clear').onclick = () => { $('log').innerHTML = ''; };
$('refresh').onclick = loadFiles;
renderRules(); setSensors(['1','1','1','1']); setWheels([0,0,0,0]); connect();
</script>
</body>
</html>
"""


def host_name():
    """브라우저 주소용 이름. macOS는 .local이 이미 붙어 나오므로 한 번만 붙인다."""
    name = socket.gethostname()
    return name[:-6] if name.endswith('.local') else name


def local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('10.255.255.255', 1))
            return s.getsockname()[0]
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description='브라우저 조종판: 자동주행 ON · 긴급 정지 · 실시간 로그')
    parser.add_argument('--host', default='0.0.0.0', help='접속을 받을 주소 (같은 컴퓨터만: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    app = App()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.app = app
    print(f'조종판 시작: http://{host_name()}.local:{args.port}  (같은 와이파이의 브라우저에서 엽니다. 종료는 Ctrl+C)', flush=True)
    ip = local_ip()
    if ip:
        print(f'  안 열리면 주소로: http://{ip}:{args.port}', flush=True)

    def on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Ctrl+C: 조종판을 정리합니다.', flush=True)
    finally:
        errors = app.shutdown()
        server.server_close()
        for exc in errors:
            print(f'정리 오류: {exc}', flush=True)


if __name__ == '__main__':
    main()
