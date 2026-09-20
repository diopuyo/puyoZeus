"""同cutoffの2P起点と既存確率観測の一致を調べる。欠測分布は生成しない。"""
import json
from pathlib import Path
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'video38_split_tail_candidate_v22'
FRAME, FPS, SIDE = 35672, 60, '2P'


def rows(name: str) -> Any:
    with (SOURCE / name).open() as stream:
        for line in stream:
            yield json.loads(line)


def main() -> None:
    started = time.perf_counter()
    step = next(r for r in rows('atomic_journal.jsonl')
        if r.get('kind') == 'step' and r['side'] == SIDE and r['frame_idx'] == FRAME)
    origins = [e['active_origin'] for e in step['events'] if e.get('active_origin') is not None]
    assert origins and len({(o['object_id'], o['trigger_sec']) for o in origins}) == 1
    origin = origins[0]
    assert origin['before_board'] is not None
    target = origin['before_board']['grid'][1:]
    matches, nearest = [], []
    for row in rows('hidden_probability.jsonl'):
        if row['side'] != SIDE or row['time_sec'] > origin['trigger_sec']:
            continue
        board = row.get('confirmed')
        if not isinstance(board, dict) or board.get('grid') is None:
            continue
        difference = sum(a != b for left, right in zip(board['grid'][1:], target)
                         for a, b in zip(left, right))
        status = dict(frame=row['frame_idx'], state=row['state_value'], difference=difference,
            probability_present=row['probability']['present'], hold_reasons=row['hold_reasons'])
        nearest.append(status)
        if difference == 0:
            matches.append(status)
    nearest = sorted(nearest, key=lambda r: (r['difference'], -r['frame']))[:5]
    result = dict(cutoff_frame=FRAME, side=SIDE, source_step=step['token'],
        trigger_sec=origin['trigger_sec'], origin_object_id=origin['object_id'],
        exact_visible_matches=matches, nearest=nearest, missing_probability_invented=False,
        live_owner_verified=False, quality_gate_clear=False, seconds=time.perf_counter() - started)
    with (ROOT / 'OPPONENT_ORIGIN_INPUTS_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
