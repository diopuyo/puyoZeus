"""B10終了後に低負荷60秒を待ち、B11 longだけを再検収する。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Callable
from unittest.mock import patch

from scripts.measure_live_b6 import wait_idle, save, MAX_WAIT_SECONDS, IDLE_SECONDS
import scripts.measure_live_b9 as b9

OUTPUT = Path('logs/live_b11')
B10_PID = 588
POLL_SEC = 5.0
# 第58試合の暗転リセット3412.217秒より前で切り、第41〜57試合に対応させる。
END_SEC = 3412.0


def b10_running(pid: int) -> bool:
    """PID再利用を別タスクの待機と混同しない。"""
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
    except FileNotFoundError:
        return False
    return b'scripts.measure_live_b10' in command


def run(output: Path, commit: str, idle: Callable = wait_idle,
        execute: Callable = b9.execute, prior_pid: int = B10_PID) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if os.nice(0) != 0:
        raise RuntimeError('認識nice 0のためnice 0で起動してください')
    deadline = time.monotonic()+MAX_WAIT_SECONDS
    save(output/'launch.json', dict(pid=os.getpid(), commit=commit, cases=['long'],
        prior_pid=prior_pid, started_unix=time.time(), max_wait_seconds=MAX_WAIT_SECONDS,
        idle_seconds=IDLE_SECONDS, start_sec=b9.START_SEC, end_sec=END_SEC,
        expected_games=17, expected_internal_boundaries=16, memory_interval_sec=300,
        memory_mode='object_counts', bounded_history=True))
    while b10_running(prior_pid):
        if time.monotonic() >= deadline:
            save(output/'status.json', dict(pid=os.getpid(), state='timeout_waiting_b10'))
            return
        save(output/'status.json', dict(pid=os.getpid(), state='waiting_b10', prior_pid=prior_pid))
        time.sleep(POLL_SEC)
    if not idle(deadline, output, 'B11_long'):
        return
    environment = dict(PUYO_BOUNDED_HISTORY='1', PUYO_MEMORY_PROFILE=str(output/'long'/'memory'),
                       PUYO_MEMORY_INTERVAL='300', PUYO_MEMORY_TRACE='0')
    try:
        with patch.dict(os.environ, environment), patch.object(b9, 'END_SEC', END_SEC):
            result = execute('long', output)
        save(output/'status.json', dict(pid=os.getpid(), state='finished', result=result,
                                       passed=result.get('passed', False)))
    except Exception as error:
        save(output/'status.json', dict(pid=os.getpid(), state='failed', error=str(error)))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--prior-pid', type=int, default=B10_PID)
    options = parser.parse_args()
    run(options.output, options.commit, prior_pid=options.prior_pid)


if __name__ == '__main__':
    main()
