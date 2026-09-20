"""既存同call NEXT輸送から同色別手の画像確認候補だけを探索。進行認証ではない。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
LIVE = ROOT.parent / 'video38_normal_completion_bridge_live_2026-09-09_v1'
INPUT = LIVE / 'current_scope.jsonl'
STRIDE, PAIR_LENGTH = 2, 2
SIDES = ('1P', '2P')


def pair(value: Any) -> bool:
    return type(value) is list and len(value) == PAIR_LENGTH and all(type(c) is int and 1 <= c <= 5 for c in value)


def add(row: dict[str, Any], opened: dict[str, Any], intervals: list[Any]) -> None:
    side, frame, args = row['side'], row['frame_idx'], row['step_kwargs']
    old = opened.get(side)
    same = pair(args['next_pair']) and args['next_pair'] == args['dnext_pair']
    if old is not None and (not same or old['pair'] != args['next_pair'] or old['last'] + STRIDE != frame):
        intervals.append(opened.pop(side))
        old = None
    if not same:
        return
    if old is None:
        old = {'side': side, 'first': frame, 'last': frame, 'pair': args['next_pair'], 'rows': 0,
               'slide_true_frames': [], 'source_id': row['source_id'], 'run_id': row['run_id']}
        opened[side] = old
    if old['source_id'] != row['source_id'] or old['run_id'] != row['run_id']:
        raise RuntimeError('scope_changed')
    old['last'], old['rows'] = frame, old['rows'] + 1
    if args['slide_motion'] is True:
        old['slide_true_frames'].append(frame)


def main(output: Path) -> int:
    output.mkdir(exist_ok=False)
    before, started, digest = INPUT.stat(), time.perf_counter(), hashlib.sha256()
    opened: dict[str, Any] = {}
    intervals: list[Any] = []
    count = 0
    with INPUT.open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row['kind'] != 'current_entry_step_enter':
                continue
            if row['side'] not in SIDES:
                raise RuntimeError('side')
            count += 1
            add(row, opened, intervals)
    intervals.extend(opened.values())
    after = INPUT.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('input_changed')
    result = {'pid': os.getpid(), 'seconds': time.perf_counter() - started, 'rows': count,
              'input': str(INPUT), 'input_sha256': digest.hexdigest(), 'intervals': intervals,
              'physical_progress_certified': False, 'model_GPU': False, 'quality_gate_clear': False}
    with (output / 'CANDIDATES.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print({'pid': result['pid'], 'seconds': result['seconds'], 'rows': count, 'intervals': len(intervals)})
    print([{k: row[k] for k in ('side', 'first', 'last', 'pair', 'slide_true_frames')} for row in intervals[:8]])
    return 0


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1])))
