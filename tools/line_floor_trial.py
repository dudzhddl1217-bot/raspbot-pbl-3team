#!/usr/bin/env python3
# 지시서: specs/02-line.md
"""사용자 터미널에서 기존 줄 주행을 실행하고 회차별 원시 기록을 저장한다."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time


class Tee:
    def __init__(self, screen, log):
        self.screen, self.log = screen, log

    def write(self, text):
        self.screen.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.screen.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description="안전 확인 후 줄 주행 회차 기록")
    parser.add_argument('--trial', type=int, required=True, choices=range(1, 6))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('line', root / 'missions/02_line.py')
    line = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(line)
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    base = root / 'results' / f'line_floor_{stamp}_trial{args.trial}'
    base.parent.mkdir(parents=True, exist_ok=True)
    record = {'회차': args.trial, '시각': datetime.now().astimezone().isoformat(),
              '프로그램': 'missions/02_line.py', '직진속도': 50,
              '시간기준': '주행 반복 시작부터 종료 정리 진입까지', '관찰': '사용자 확인 대기'}
    started = None
    stopped = None
    original_stop = line.stop_all

    def timed_stop(bot):
        nonlocal stopped
        stopped = time.monotonic()
        original_stop(bot)

    line.stop_all = timed_stop
    with base.with_suffix('.txt').open('x', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                from Raspbot_Lib import Raspbot
                with line.keyboard_stop() as key_pressed:
                    bot = Raspbot()
                    started = time.monotonic()
                    line.follow(bot, key_pressed)
                record['종료'] = '정상 종료'
            except KeyboardInterrupt:
                record['종료'] = 'Ctrl+C'
                print('Ctrl+C: 정지 정리를 마쳤습니다.', flush=True)
            except BaseException as exc:
                record['종료'] = f'오류: {type(exc).__name__}: {exc}'
                raise
            finally:
                record['시간(초)'] = round(stopped - started, 4) if started is not None and stopped is not None else None
                record['원시기록'] = str(base.with_suffix('.txt').relative_to(root))
                base.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                print(f'기록 저장: {base.with_suffix(".json")} / 주행 시간: {record["시간(초)"]}초', flush=True)


if __name__ == '__main__':
    main()
