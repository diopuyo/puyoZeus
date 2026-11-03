"""B14検証済み版だけを低負荷待ちでlong再測し、B13との品質差を残す。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from scripts.analyze_live_b14 import source_hashes
from scripts.measure_live_b6 import save, wait_idle, MAX_WAIT_SECONDS
from scripts.measure_live_b12 import execute, START, LONG_END


def check_proof(path: Path) -> dict:
    proof = json.loads(path.read_text())
    if not proof['equal'] or not proof['target_met']:
        raise ValueError('完全一致と速度目標の合格記録が必要です')
    if proof['source_hashes'] != source_hashes():
        raise ValueError('検証後に認識コードが変わっています')
    return proof


def finish_comparison(root: Path, reference: Path) -> None:
    from scripts.analyze_live_b12 import compare
    metrics = json.loads((root/'long/metrics.json').read_text())
    old_metrics = json.loads((reference/'metrics.json').read_text())
    for key, value in old_metrics['assets'].items():
        if key != 'app_build_id' and metrics['assets'].get(key) != value:
            raise ValueError(f'{key}: B13と同じ資産が必要です')
    configs = [json.loads((path/'recognition_config.json').read_text())
               for path in (reference, root/'long')]
    if configs[0] != configs[1]:
        raise ValueError('B13と認識設定が異なります')
    current = compare(reference, root/'long', START, LONG_END)
    baseline = json.loads(Path('logs/live_b13/impact.json').read_text())
    old_drops = json.loads(Path('logs/live_b13/drop_breakdown.json').read_text())['total']
    summary = dict(b13=dict(expected=old_drops['expected'], dropped=old_drops['dropped'],
        drop_fraction=old_drops['drop_fraction'], **baseline['boards']['placements']),
        b14=dict(expected=metrics['expected_frames'],
        dropped=metrics['capture_dropped_in_measured_window'],
        drop_fraction=metrics['capture_drop_fraction'], **current['boards']['placements']))
    save(root/'b13_comparison.json', dict(summary=summary, b14=current, b13=baseline))


def run(args: argparse.Namespace) -> None:
    proof = check_proof(args.proof)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    save(root/'launch.json', dict(pid=os.getpid(), commit=args.commit, started_unix=time.time(),
         cwd=str(Path.cwd()), proof=str(args.proof), source_hashes=proof['source_hashes'],
         start=START, end=LONG_END, cpu_threads=1, device='cpu', selected='c'))
    if not wait_idle(time.monotonic()+MAX_WAIT_SECONDS, root, 'B14 long'):
        return
    try:
        check_proof(args.proof)
        execute('long', root, LONG_END-START, 'c', 0, None)
        finish_comparison(root, args.reference)
        save(root/'status.json', dict(state='finished', pid=os.getpid()))
    except Exception as error:
        save(root/'status.json', dict(state='failed', pid=os.getpid(), error=str(error)))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('logs/live_b14'))
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--reference', type=Path, default=Path('logs/live_b13/reference'))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
