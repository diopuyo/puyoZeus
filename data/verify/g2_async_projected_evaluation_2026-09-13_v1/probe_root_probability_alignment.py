"""初回landing原票と後刻PBの内容/時点を照合する。live所有は認証しない。"""
import hashlib
import json
from pathlib import Path
import time
from probe_opponent_root_projection import rows, deterministic_grid

ROOT = Path(__file__).resolve().parent
ORIGIN_FRAME, PROBABILITY_FRAME, SIDE = 34702, 34718, '2P'


def main() -> None:
    started = time.perf_counter()
    step = next(row for row in rows('atomic_journal.jsonl') if row.get('kind') == 'step'
                and row['frame_idx'] == ORIGIN_FRAME and row['side'] == SIDE)
    origins = [event['active_origin'] for event in step['events'] if event.get('active_origin') is not None]
    origin = next(value for value in origins if value['mechanism'] == 'landing')
    pb = next(row for row in rows('hidden_probability.jsonl')
              if row['frame_idx'] == PROBABILITY_FRAME and row['side'] == SIDE)
    assert all(step[key] == pb[key] for key in ('source_id', 'run_id', 'side'))
    assert pb['state'] == 'STABLE' and not pb['hold_reasons'] and not pb['instrumentation_errors']
    grid = deterministic_grid(pb['probability'])
    expected = tuple(map(tuple, origin['before_board']['grid']))
    changed = [(row, col) for row in range(len(grid)) for col in range(len(grid[row])) if grid[row][col] != expected[row][col]]
    sha = hashlib.sha256(json.dumps(grid, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = dict(origin_frame=ORIGIN_FRAME, probability_frame=PROBABILITY_FRAME,
        earliest_content_available_frame=PROBABILITY_FRAME, origin_trigger_sec=origin['trigger_sec'],
        full_grid_equal=not changed, changed_cells=changed, probability_all_cells_one=True,
        probability_grid_sha=sha, origin_board_sha=origin['before_board']['sha256'],
        source_run_side_match=True, live_owner_verified=False, before_trigger_available=False,
        physical_identity_verified=False, quality_gate_clear=False, seconds=time.perf_counter()-started)
    with (ROOT / 'ROOT_PROBABILITY_ALIGNMENT_v1.json').open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
