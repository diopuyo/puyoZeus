"""E33bの固定入力・四条件再生と、表示時点の得点原票を保存する。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from typing import Any
from scripts import run_e32 as e32
from scripts import replay_exchange_event_20260926 as replay_module
from scripts.e33b_score_trace import ScoreTrace, OUT, prepare_cohort
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

WORKERS = 2
VARIANTS = ('off', 'e33', 'on', 'signal')


def signal_only() -> None:
    """原因監査専用の対照。製品フラグは追加せず時間切れの除外だけを抑止する。"""
    from src.exchange_prefire_stage_timeout import StageTimeoutPrefire
    original = StageTimeoutPrefire._exclude_later
    def exclude(engine: Any, entry: dict, stamp: float, reason: str) -> None:
        if reason != 'stage_timeout':
            original(engine, entry, stamp, reason)
    StageTimeoutPrefire._exclude_later = exclude


def worker(source: str, variant: str) -> None:
    """一条件を独立プロセスで実行し、計測フックを他条件へ持ち越さない。"""
    prior = e32.e31.prior
    prior.OUT = OUT
    dest = prior.directory(variant, source)
    prior.AuditTrace = lambda name: ScoreTrace(name, dest)
    options = dict(e32.OPTIONS)
    if variant == 'on':
        options['prefire_stage_timeout_only'] = True
    elif variant in ('e33', 'signal'):
        options['prefire_stage_timeout'] = True
    if variant == 'signal':
        signal_only()
    original_display = replay_module.display_row
    def enriched(record: Path, target: Path, *args: Any, **kwargs: Any) -> dict:
        trace = args[2]
        def display(*values: Any, **named: Any) -> Any:
            row = original_display(*values, **named)
            trace.observe_display(values[0], row)
            return row
        replay_module.display_row = display
        value = replay_module.replay(Path('logs/e31/records')/record.name, target, *args, **kwargs)
        save_json(target/'snapshot_final_audit.json', trace.summary())
        return value
    prior.replay = enriched
    prior.locked_worker(variant, source, options)
    if source in SOURCES:
        from scripts.report_e33b import predictions, read
        predictions(variant, source, read(OUT/'cohort'/f'{source}.json'))
    if variant in ('off', 'e33'):
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        baseline = 'e32' if variant == 'off' else 'e33'
        save_json(OUT/f'{variant.upper()}_{source}.json',
            replay_module.compare(Path(f'logs/{baseline}/on')/suffix, dest))


def launch(task: tuple[str, str]) -> None:
    """既存診断と共存できるよう子プロセスを二つまでに制限する。"""
    source, variant = task
    with (OUT/f'{variant}_{source}.log').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'scripts.run_e33b', '--source', source,
            '--variant', variant], stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    """採点母数を先に固定してから、同じ入力で全比較条件を再生する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=e32.e31.prior.ALL_SOURCES)
    parser.add_argument('--variant', choices=VARIANTS, default='on')
    args = parser.parse_args()
    assert (OUT/'PREREGISTRATION.md').exists()
    if args.source:
        worker(args.source, args.variant)
        return
    prepare_cohort()
    tasks = [('zenchi', 'on'), ('zenchi', 'signal')]
    tasks += [(s, v) for v in ('off', 'e33', 'on') for s in SOURCES]
    tasks += [('review', 'on'), ('review', 'off'), ('zenchi', 'off')]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(launch, tasks))
    from scripts.report_e33b import report
    print(report(), flush=True)


if __name__ == '__main__':
    main()
