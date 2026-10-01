"""Phase 6の固定順序を再開可能に実行する。採点は係数・L固定後だけ。"""
from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.prefire_v6_replay import OUT
from scripts.prefire_v5c_batch import resources
from scripts.run_prefire_replay_20260930 import BASELINE_DIRS

MAX_PROCESSES, WORKERS = 8, 4
MIN_AVAILABLE_KIB, START_AVAILABLE_KIB = 4*1024*1024, 6*1024*1024
POLL_SEC = 2


def launch(mode: str, source: str, extra: list[str]) -> subprocess.Popen:
    """出力先を分離し、失敗時のログも追記で残す。"""
    with (OUT/f'{mode}_{source}.log').open('a') as log:
        return subprocess.Popen([sys.executable, '-B', '-m', 'scripts.prefire_v6_replay', mode, source, *extra],
                                stdout=log, stderr=subprocess.STDOUT)


def batch(mode: str, sources: list[str], extras: dict | None = None) -> None:
    """全WSLのPython数を見て起動し、低メモリ時は自身の子だけを一時停止する。"""
    pending = [s for s in sources if not (OUT/mode/s/'DONE.json').exists()]
    running: dict[str, subprocess.Popen] = {}
    minimum, maximum = resources()
    suspended = False
    while pending or running:
        for source, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            if code:
                raise RuntimeError(f'{mode}/{source}: exit {code}')
            del running[source]
        available, count = resources()
        minimum, maximum = min(minimum, available), max(maximum, count)
        if available < MIN_AVAILABLE_KIB and not suspended:
            for process in running.values():
                process.send_signal(signal.SIGSTOP)
            suspended = True
        if suspended and available >= START_AVAILABLE_KIB:
            for process in running.values():
                process.send_signal(signal.SIGCONT)
            suspended = False
        if pending and not suspended and len(running) < WORKERS and count < MAX_PROCESSES and available >= START_AVAILABLE_KIB:
            source = pending.pop(0)
            running[source] = launch(mode, source, (extras or {}).get(source, []))
        with (OUT/'resource_history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), mode=mode, available_kib=available,
                python_count=count, min_available_kib=minimum, max_python_count=maximum,
                running=list(running), pending=pending, suspended=suspended))+'\n')
        time.sleep(POLL_SEC)


def cuts() -> None:
    """qの300/600秒打切りを独立再生する。元記録は変更しない。"""
    from scripts.prefire_truncation_audit_20260930 import truncate
    extras = {}
    for cut in (300, 600):
        name = f'q{cut}'
        path = OUT/f'{name}.jsonl.gz'
        if not path.exists():
            truncate(cut, path)
        extras[name] = ['--record', str(path)]
    batch('cut', list(extras), extras)


def main() -> None:
    """短窓→セット1学習→セット2と回帰→打切り→採点の順序を固定する。"""
    if not (OUT/'latency.json').exists() or not json.loads((OUT/'CHECKS_v2.json').read_text())['passed']:
        raise RuntimeError('L固定と本番基準一致が必要')
    if not (OUT/'strength.json').exists():
        batch('train', ['c1'])
        batch('train', [*(f'c{i}' for i in range(2, 7)), 'zenchi'])
        subprocess.run([sys.executable, '-B', '-m', 'scripts.prefire_v6_train'], check=True)
    batch('replay', ['s0'])
    batch('replay', [*(f's{i}' for i in range(1, 7)), *BASELINE_DIRS])
    cuts()
    subprocess.run([sys.executable, '-B', '-m', 'scripts.prefire_v6_score'], check=True)
    subprocess.run([sys.executable, '-B', '-m', 'scripts.prefire_v6_report'], check=True)


if __name__ == '__main__':
    main()
