"""WSL低負荷60秒を最長3時間待ち、B7 CPU条件を一度だけ実行する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from scripts.measure_live_b6 import (
    IDLE_SECONDS, MAX_WAIT_SECONDS, command, run_condition, save, wait_idle,
)

OUTPUT = Path('logs/live_b7_measurement')
TARGET_P95_MS = 500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--commit', required=True)
    options = parser.parse_args()
    output = options.output
    output.mkdir(parents=True, exist_ok=True)
    if os.nice(0) != 0:
        raise RuntimeError('nice 0で起動してください')
    # Windows形式のworktree参照はWSL gitで解決できないため、起動元で取得する。
    commit = options.commit
    deadline = time.monotonic()+MAX_WAIT_SECONDS
    save(output/'launch.json', dict(pid=os.getpid(), started_unix=time.time(),
        nice=os.nice(0), max_wait_seconds=MAX_WAIT_SECONDS, idle_seconds=IDLE_SECONDS,
        parallelism=1, cwd=str(Path.cwd()), python=sys.executable, commit=commit,
        command=command('cpu', output), split_evaluation=True))
    if wait_idle(deadline, output, 'cpu'):
        result = run_condition('cpu', output)
        path = output/'cpu/metrics.json'
        if result['state'] == 'complete' and path.exists():
            metrics = json.loads(path.read_text())
            p95 = metrics['latency_ms']['capture_to_sse']['P95']
            result.update(state_update_ms=metrics['state_update_ms'],
                probability_calculation_ms=metrics['probability_calculation_ms'],
                split_evaluation=metrics['split_evaluation'],
                target_p95_ms=TARGET_P95_MS, target_met=p95 is not None and p95 <= TARGET_P95_MS)
    else:
        result = dict(state='skipped', reason='低負荷待機の3時間上限')
    save(output/'results.json', {'cpu': result})
    save(output/'status.json', dict(pid=os.getpid(), state='finished', results={'cpu': result}))


if __name__ == '__main__':
    main()
