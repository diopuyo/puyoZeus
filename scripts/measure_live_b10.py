"""低負荷待機後、修正したresolution/repeatだけを直列で再検収する。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Callable

from scripts.measure_live_b6 import wait_idle, save, MAX_WAIT_SECONDS, IDLE_SECONDS
from scripts.measure_live_b9 import execute

OUTPUT = Path('logs/live_b10')
CASES = ('resolution', 'repeat')


def run_cases(output: Path, commit: str, idle: Callable = wait_idle, run: Callable = execute) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if os.nice(0) != 0:
        raise RuntimeError('認識nice 0のためnice 0で起動してください')
    save(output/'launch.json', dict(pid=os.getpid(), commit=commit, cases=CASES,
        started_unix=time.time(), max_wait_seconds=MAX_WAIT_SECONDS, idle_seconds=IDLE_SECONDS,
        long_reused_from='logs/live_b9/long'))
    if not idle(time.monotonic()+MAX_WAIT_SECONDS, output, 'B10_serial'):
        return
    results = {}
    for case in CASES:
        try:
            results[case] = run(case, output)
        except Exception as error:
            results[case] = dict(passed=False, error=str(error))
        save(output/'results.json', results)
    save(output/'status.json', dict(pid=os.getpid(), state='finished', results=results,
        passed=all(result.get('passed', False) for result in results.values())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--commit', required=True)
    options = parser.parse_args()
    run_cases(options.output, options.commit)


if __name__ == '__main__':
    main()
