"""E34bの再収集と新固定入力上の対比較を最大3並列で実行する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts import run_e32 as e32
from scripts import replay_exchange_event_20260926 as replay_module
from scripts.e34b_origin_trace import OriginTrace
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e34b')
WORKERS = 3
SOURCES = (*e32.e31.prior.ALL_SOURCES,)


def worker(source: str, variant: str) -> None:
    """両条件が同じ新しい原票を読み、採点時点の入力得点を保存する。"""
    prior = e32.e31.prior
    prior.OUT = OUT
    dest = prior.directory(variant, source)
    prior.AuditTrace = lambda name: OriginTrace(name, dest)
    options = dict(e32.OPTIONS)
    if variant == 'on':
        options['prefire_origin_guard'] = True
    original_display = replay_module.display_row
    def enriched(record: Path, target: Path, *args: Any, **kwargs: Any) -> dict:
        trace = args[2]
        def display(*values: Any, **named: Any) -> Any:
            row = original_display(*values, **named)
            trace.observe_display(values[0], row)
            return row
        replay_module.display_row = display
        value = replay_module.replay(OUT/'records'/record.name, target, *args, **kwargs)
        save_json(target/'snapshot_final_audit.json', trace.summary())
        return value
    prior.replay = enriched
    prior.locked_worker(variant, source, options)


def launch(task: tuple[str, str]) -> None:
    """一動画一段階の完了を保存し、再開時に完成済み入力を作り直さない。"""
    source, phase = task
    if phase == 'collect' and (OUT/'records'/f'{source}.jsonl.json').exists():
        return
    module = 'scripts.collect_e34b' if phase == 'collect' else 'scripts.run_e34b'
    args = [sys.executable, '-m', module, '--source', source]
    if phase != 'collect':
        args.extend(['--phase', phase])
    with (OUT/f'{phase}_{source}.log').open('a') as stream:
        subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    """新OFFで採点行を固定してから、新ONを開始する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=SOURCES)
    parser.add_argument('--phase', choices=('collect', 'off', 'on', 'all'), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert (OUT/'PREREGISTRATION.md').exists()
    if args.source:
        if args.phase not in ('off', 'on'):
            parser.error('--sourceには--phase offまたはonを指定する')
        worker(args.source, args.phase)
        return
    phases = ('collect', 'off', 'on') if args.phase == 'all' else (args.phase,)
    for phase in phases:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(launch, [(source, phase) for source in SOURCES]))
        save_json(OUT/f'{phase.upper()}_COMPLETE.json', dict(sources=SOURCES))
        if phase == 'off':
            from scripts.report_e34b import prepare_cohort
            prepare_cohort()
    if args.phase in ('all', 'on'):
        from scripts.report_e34b import report
        print(report(), flush=True)
        from scripts.e34b_residuals import residuals
        print(residuals(), flush=True)


if __name__ == '__main__':
    main()
