"""保存入力を配信用の専用processへ配送し、capped_rep原票と全列照合する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from scripts.prefire_v44_data import notifications
from scripts.prefire_v45_origins import sources as old_sources
from scripts.prefire_v48_inputs import sources
from scripts.prefire_v49e_speed_verify import stable, differences, memory_guard
from scripts.prefire_v49e_speed_accounting import rows, sha
from scripts.prefire_v45_data import ASSETS
from src.phase_j.live_v49f_async import AsyncV49F

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(_public_dev_text('<work>/prefire_v49f_20261019/full_fast_clean1'))


def messages(video: int, game: int) -> list[dict]:
    source = ((i, s) for i, s in old_sources(video) if i['game'] == game) if video == 80 else sources(video, {game})
    result = []
    for index, (inp, stream) in enumerate(source):
        result.append(dict(t=inp['input_sec'], priority=-1, order=index, id=index,
                           kind='initial', data=inp, initial_next=stream['initial_next']))
        for priority, (kind, key, time_key) in enumerate((('formula', 'formulas', 't_sec'),
                                                        ('score', 'scores', 't'), ('frame', 'frames', 't'))):
            for value in stream[key]:
                result.append(dict(t=value[time_key], priority=priority, order=index,
                                   id=index, kind=kind, data=value))
    return sorted(result, key=lambda r: (r['t'], r['order'], r['priority']))


def compare(expected: Any, actual: Any, out: Path, kind: str, count: int) -> None:
    left = json.dumps(stable(expected), sort_keys=True, separators=(',', ':'))
    right = json.dumps(stable(actual), sort_keys=True, separators=(',', ':'))
    if left != right:
        value = dict(kind=kind, row=count, differences=differences(stable(expected), stable(actual)),
                     expected=expected, actual=actual)
        (out/'first_mismatch.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
        raise AssertionError(f'{kind} row {count}: {value["differences"][:2]}')


def run(video: int, game: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    memory_guard()
    assets = (ASSETS/'model.joblib', ASSETS/'timing_table.json',
              ROOT/'docs/verification/exch_review_v310/policy_locked.json',
              ROOT/'docs/verification/prefire_v44/model.json')
    process = AsyncV49F(assets, out)
    events, cursor, count, base_count = messages(video, game), 0, 0, 0
    expected = iter(r for r in rows(REFERENCE/f'predictions_{video}.jsonl') if r['game'] == game)
    expected_base = iter(r for r in rows(REFERENCE/f'base_{video}.jsonl') if r['game'] == game)
    began = perf_counter()
    try:
        for index, (stamp, current_game, raw) in enumerate(notifications(video)):
            if current_game != game:
                continue
            delivered = []
            while cursor < len(events) and events[cursor]['t'] <= stamp:
                delivered.append(events[cursor]); cursor += 1
            process.submit(dict(id=index, t=stamp, game=game, raw=raw, origins=delivered))
            if not process.connection.poll(120.):
                raise TimeoutError('配信経路再生の応答なし')
            result = process.connection.recv()
            if result.get('fatal'):
                raise RuntimeError(result)
            if result['base'] is not None:
                compare(next(expected_base, None), dict(result['base'], video=video), out, 'base', base_count)
                base_count += 1
            if result['value'] is not None:
                compare(next(expected, None), dict(result['value'], video=video), out, 'v49f', count)
                count += 1
                if count % 100 == 0:
                    memory_guard()
                    print(video, game, count, stamp, round(perf_counter()-began, 2), flush=True)
        compare(next(expected, None), None, out, 'v49f_tail', count)
        compare(next(expected_base, None), None, out, 'base_tail', base_count)
    finally:
        process.close()
    receipt = dict(video=video, game=game, rows=count, base_rows=base_count,
        non_time_differences=0, seconds=perf_counter()-began, origin_messages=cursor,
        reference=str(REFERENCE), reference_sha256=sha(REFERENCE/f'predictions_{video}.jsonl'))
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--video', required=True, type=int)
    parser.add_argument('--game', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.game, args.out)), flush=True)


if __name__ == '__main__':
    main()
