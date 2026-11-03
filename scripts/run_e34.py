"""E34の固定入力を検査し、E32と同じ行でON/OFFを再生する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts import run_e32 as e32
from scripts import replay_exchange_event_20260926 as replay_module
from scripts.e33b_score_trace import ScoreTrace
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.exchange_event_record import read_records
from src.match_range_gate import load_ranges

OUT = Path('logs/e34')
WORKERS = 2


def preflight() -> dict:
    """旧原票の保持印と既存試合範囲を調べ、不足を代替判定で埋めない。"""
    result = {}
    for source in e32.e31.prior.ALL_SOURCES:
        stream = read_records(Path('logs/e31/records')/f'{source}.jsonl.gz')
        header = next(stream)
        row = next(r for r in stream if r['kind'] == 'update')
        sides = (row['args'][0].p1, row['args'][0].p2)
        ranges, path = load_ranges(header['video_id'], Path.cwd())
        result[source] = dict(video_id=header['video_id'], match_source=path,
            match_ranges=len(ranges), hold_available=all(
                getattr(s, 'prefire_origin_hold', None) is not None for s in sides))
        stream.close()
    save_json(OUT/'PREFLIGHT.json', result)
    return result


def worker(source: str, variant: str) -> None:
    """OFFも再計算し、E32保存出力の全列とイベントを照合する。"""
    prior = e32.e31.prior
    prior.OUT = OUT
    dest = prior.directory(variant, source)
    prior.AuditTrace = lambda name: ScoreTrace(name, dest)
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
        root = OUT/'records' if variant == 'on' else Path('logs/e31/records')
        value = replay_module.replay(root/record.name, target, *args, **kwargs)
        save_json(target/'snapshot_final_audit.json', trace.summary())
        return value
    prior.replay = enriched
    prior.locked_worker(variant, source, options)
    if variant == 'off':
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        save_json(OUT/f'OFF_{source}.json',
                  replay_module.compare(Path('logs/e32/on')/suffix, dest))


def launch(source: str, variant: str) -> None:
    """動画ごとに独立プロセスを起動する。"""
    with (OUT/f'{variant}_{source}.log').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'scripts.run_e34', '--source', source,
                        '--variant', variant], stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    """再生前に事前登録を要求し、無指定ではOFF互換から確認する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=e32.e31.prior.ALL_SOURCES)
    parser.add_argument('--variant', choices=('off', 'on'), default='off')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert (OUT/'PREREGISTRATION.md').exists()
    if args.preflight:
        print(preflight(), flush=True)
    elif args.source:
        worker(args.source, args.variant)
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(lambda s: launch(s, args.variant), e32.e31.prior.ALL_SOURCES))


if __name__ == '__main__':
    main()
