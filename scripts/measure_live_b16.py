"""B15と同じ60分を一度だけ再実行し、評価障害も自動集計する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from scripts.live_b15_plan import START, END, prepare
from scripts.measure_live_b15 import realtime, reference
from scripts.measure_live_b6 import save, wait_idle, MAX_WAIT_SECONDS
from scripts.analyze_live_b14 import source_hashes
from scripts.analyze_live_b15 import runtime_report, quality_report
from src.phase_j.live_eval_supervisor import FAILURE_LIMIT

OUTPUT = Path('logs/live_b16')


def error_report(root: Path) -> dict:
    result = runtime_report(root)
    path = root/'realtime/evaluation_errors.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    errors = [r for r in rows if r['kind'] == 'evaluation_error']
    restarts = [r for r in rows if r['kind'] == 'evaluation_restart']
    for window in result['windows']:
        window['evaluation_errors'] = sum(window['start_sec'] <= r['t_sec'] < window['end_sec']
                                          for r in errors)
    result.update(evaluation_error_count=len(errors), evaluation_restart_count=len(restarts),
                  evaluation_errors=errors, evaluation_restarts=restarts)
    save(root/'report.json', result)
    return result


def run(root: Path, commit: str) -> None:
    if (root/'launch.json').exists() or (root/'realtime').exists():
        raise ValueError('既存の60分運転を上書き・再実行しません')
    plan, hashes = prepare(root), source_hashes()
    save(root/'launch.json', dict(pid=os.getpid(), commit=commit, started_unix=time.time(),
        start=START, end=END, expected_games=plan['expected_games'],
        expected_internal_boundaries=plan['expected_internal_boundaries'], source_hashes=hashes,
        cpu_threads=1, device='cpu', selected='c', realtime_runs=1, failure_limit=FAILURE_LIMIT))
    try:
        if not wait_idle(time.monotonic()+MAX_WAIT_SECONDS, root, 'B16 60min'):
            return
        if source_hashes() != hashes:
            raise ValueError('待機中に認識コードが変わりました')
        realtime(root)
        error_report(root)
        if not wait_idle(time.monotonic()+MAX_WAIT_SECONDS, root, 'B16 offline reference'):
            return
        if source_hashes() != hashes:
            raise ValueError('実時間運転と参照の認識コードが異なります')
        reference(root)
        quality_report(root)
        save(root/'status.json', dict(state='finished', pid=os.getpid()))
    except Exception as error:
        save(root/'status.json', dict(state='failed', pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--commit', required=True)
    options = parser.parse_args()
    run(options.output, options.commit)
