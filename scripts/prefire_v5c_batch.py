"""固定Lの5記録・セット1を、空きメモリと全Python数を監視して再生する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_prefire_replay_20260930 import BASELINE_DIRS

OUT = Path('logs/prefire_prediction/v5c')
MAX_PROCESSES = 8
START_AVAILABLE_KIB = 6 * 1024 * 1024
POLL_SECONDS = 1


def resources() -> tuple[int, int]:
    """WSL全体のavailableと、他worktreeを含むvenv Pythonプロセス数を読む。"""
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    count = 0
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().split(b'\0')
        except OSError:
            continue
        count += bool(command and b'venv/bin/python' in command[0])
    return available, count


def launch(source: str) -> subprocess.Popen:
    """元記録は読むだけで、各記録の出力先を分ける。"""
    with (OUT/f'replay_{source}.log').open('a') as log:
        return subprocess.Popen([sys.executable, '-B', '-m', 'scripts.prefire_v5c_replay', source],
                                stdout=log, stderr=subprocess.STDOUT)


def run(sources: list[str], workers: int) -> None:
    """失敗ジョブを成功扱いせず終了し、完了済み記録だけ再開時に省く。"""
    pending = [s for s in sources if not (OUT/'replay'/s/'DONE.json').exists()]
    running: dict[str, subprocess.Popen] = {}
    minimum, maximum = resources()
    while pending or running:
        for source, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            if code:
                raise RuntimeError(f'{source}: exit {code} (他の実行中記録は継続)')
            del running[source]
            print('completed', source, flush=True)
        available, count = resources()
        minimum, maximum = min(minimum, available), max(maximum, count)
        if pending and len(running) < workers and count < MAX_PROCESSES and available >= START_AVAILABLE_KIB:
            source = pending.pop(0)
            running[source] = launch(source)
            print('started', source, 'available_kib', available, 'processes_before', count, flush=True)
        (OUT/'replay_resources.json').write_text(json.dumps(dict(min_available_kib=minimum,
            max_python_processes=maximum, running=list(running), pending=pending)), encoding='utf-8')
        time.sleep(POLL_SECONDS)


def monitor() -> None:
    """監督終了後も残っている再生を重複起動せず、完了と資源だけ監視する。"""
    path = OUT/'replay_resources.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    backup = OUT/'replay_resources_initial.json'
    if previous and not backup.exists():
        backup.write_text(json.dumps(previous), encoding='utf-8')
    minimum, maximum = resources()
    minimum = min(minimum, previous.get('min_available_kib', minimum))
    maximum = max(maximum, previous.get('max_python_processes', maximum))
    sources = [*BASELINE_DIRS, *(f'c{i}' for i in range(1, 7))]
    while True:
        incomplete = [s for s in sources if not (OUT/'replay'/s/'DONE.json').exists()]
        available, count = resources()
        minimum, maximum = min(minimum, available), max(maximum, count)
        path.write_text(json.dumps(dict(min_available_kib=minimum, max_python_processes=maximum,
            running=incomplete, pending=[], monitor_only=True)), encoding='utf-8')
        if not incomplete:
            print('all records completed', flush=True)
            return
        time.sleep(POLL_SECONDS)


def main() -> None:
    """短区間の完了を必須にし、その後だけ全記録へ広げる。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--monitor-only', action='store_true')
    args = parser.parse_args()
    if not (OUT/'latency.json').exists():
        raise FileNotFoundError('採点前にLの固定が必要')
    if args.monitor_only:
        monitor()
        return
    run(['short'], 1)
    run([*BASELINE_DIRS, *(f'c{i}' for i in range(1, 7))], args.workers)


if __name__ == '__main__':
    main()
