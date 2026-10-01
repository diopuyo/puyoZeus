"""固定100局面の未処理おじゃまを、本番と同じ記録再生の台帳から補完する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.run_prefire_replay_20260930 import BASELINE_DIRS, RECORDS, options, model_directory
from src.prefire_v5_value import pending_counts

ROOT = Path('logs/prefire_prediction/v5_experiment')
TIME_DIGITS = 6


def collect_source(source: str, samples: list[dict]) -> list[dict]:
    """識別済み局面の盤面・ツモは変更せず、同時刻の観測台帳だけを読む。"""
    from scripts import d5_runtime
    from scripts.replay_exchange_event_20260926 import replay
    d5_runtime.OUT = ROOT/'ledger_runtime'/source
    d5_runtime.install()
    lookup = {(r['game'], round(r['t_sec'], TIME_DIGITS)): r for r in samples if r['source'] == source}
    found = []
    def observe(overlay: Any, inputs: tuple) -> None:
        key = (inputs[4], round(inputs[3], TIME_DIGITS))
        if key in lookup:
            pending = pending_counts(overlay)
            if pending is None:
                raise ValueError(f'台帳欠測: {source} {key}')
            found.append(dict(lookup[key], pending=pending, pending_provenance='replayed_observed_ledger'))
    destination = ROOT/'ledger_replay'/source
    destination.mkdir(parents=True, exist_ok=True)
    replay(RECORDS/f'{source}.jsonl.gz', destination, model_directory(options('off')), True,
           observe, **options('off'))
    if len(found) != len(lookup):
        raise ValueError(f'固定局面の時刻が見つからない: {source}: {len(found)}/{len(lookup)}')
    (ROOT/f'ledger_{source}.json').write_text(json.dumps(found), encoding='utf-8')
    print(source, len(found), flush=True)
    return found


def main() -> None:
    """6本目の枠で順次実行し、既存0固定の試験原票を保持したまま別原票を作る。"""
    samples = json.loads((ROOT/'samples.json').read_text())
    found = [row for source in BASELINE_DIRS for row in collect_source(source, samples)]
    (ROOT/'samples_ledger.json').write_text(json.dumps(found), encoding='utf-8')


if __name__ == '__main__':
    main()
