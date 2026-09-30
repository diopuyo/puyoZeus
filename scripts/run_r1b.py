"""R1bのON収集・本番再生を、既存OFFに触れず最大3並列で実行する。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from scripts.run_e17_ablation_20260928 import ALL_SOURCES
from scripts.run_e3_exchange_eval_20260926 import save_json, digest

OUT = Path('logs/r1b')
BASE = Path('logs/r1')
WORKERS = 3


def directory(source: str) -> Path:
    """既存の本番再生と同じ配置にする。"""
    return OUT/'on'/(source if source in ('review', 'zenchi') else f'renders/{source}/on')


def worker(step: str, source: str) -> None:
    """署名維持済みの収集器・本番CLI再生器をそのまま流用する。"""
    assert json.loads((OUT/'stage2/SUMMARY.json').read_text())['passed']
    assert json.loads((BASE/'OFF_INPUT_CHECK.json').read_text())['passed']
    if step == 'collect':
        from scripts import collect_r1 as capture
        capture.OUT = OUT
        capture.collect(source, 'on')
    else:
        from scripts import run_r1_evaluate as evaluate
        assert (OUT/'completed'/f'{source}.json').exists()
        evaluate.OUT = OUT
        evaluate.worker(source, 'on')


def launch(task: tuple[str, str]) -> dict:
    """完了原票を保持し、同一記録の重複起動をロックで防ぐ。"""
    step, source = task
    marker = OUT/'completed'/f'{source}.json' if step == 'collect' else directory(source)/'DONE.json'
    with (OUT/f'{step}_{source}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not marker.exists():
            with (OUT/f'{step}_{source}.log').open('a') as stream:
                subprocess.run([sys.executable, '-B', '-m', 'scripts.run_r1b',
                    '--step', step, '--source', source], stdout=stream,
                    stderr=subprocess.STDOUT, check=True)
    return dict(source=source, completed=True)


def main() -> None:
    """OFFはR1の検収済み原票を流用し、ONだけを起動する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=('collect', 'evaluate'), required=True)
    parser.add_argument('--source', choices=ALL_SOURCES)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.source:
        worker(args.step, args.source)
        return
    save_json(OUT/'BASELINE_REFERENCE.json', {str(p): digest(p) for p in
        (BASE/'OFF_INPUT_CHECK.json', BASE/'SUMMARY.json', BASE/'RECOGNITION_ROWS.json')})
    order = ('zenchi', *(s for s in ALL_SOURCES if s != 'zenchi'))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        result = list(pool.map(launch, [(args.step, s) for s in order]))
    save_json(OUT/f'{args.step.upper()}_COMPLETE.json', result)


if __name__ == '__main__':
    main()
