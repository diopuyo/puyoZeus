"""2P保存PBの全cellが確率1の場合だけ、非所有の物理診断を行う。"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator
import projected_view as P
from projection_origin_selector import selected_origin

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'video38_split_tail_candidate_v22'
ANCHOR, CALL_FRAME, CUTOFF, FPS = 34718, 34866, 35672, 60
SIDE, TOKEN, ORIGIN_ID = '2P', 'step:5815', 130711948149152


def rows(name: str) -> Iterator[dict[str, Any]]:
    with (SOURCE / name).open() as stream:
        for line in stream:
            yield json.loads(line)


def inputs() -> tuple[dict, dict, dict]:
    found = [r for r in rows('atomic_journal.jsonl') if r.get('token') == TOKEN]
    assert len(found) == 1 and found[0]['frame_idx'] == CALL_FRAME
    step = found[0]
    origin = selected_origin(step, SIDE, TOKEN, ORIGIN_ID)
    found = [r for r in rows('hidden_probability.jsonl')
             if r['frame_idx'] == ANCHOR and r['side'] == SIDE]
    assert len(found) == 1
    source = found[0]
    assert source['state_value'] == 'stable' and not source['hold_reasons']
    assert ANCHOR / FPS <= origin['trigger_sec'] <= CUTOFF / FPS
    capture = source['probability']['capture_scope']
    assert capture['source_id'] == step['source_id'] and capture['run_id'] == step['run_id']
    assert capture['side'] == SIDE and capture['frame_idx'] == ANCHOR
    return source, step, origin


def deterministic_grid(probability: dict) -> tuple:
    assert probability['present'] and probability['type_valid'] and not probability['errors']
    cells = probability['cells']
    assert len(cells) == P.B.BOARD_ROWS and all(len(row) == P.B.BOARD_COLS for row in cells)
    assert all(len(cell) == 1 and len(cell[0]) == 2 and cell[0][1] == 1.0
               and type(cell[0][0]) is int and cell[0][0] in P.B.PROB_COLORS
               for row in cells for cell in row), '非確定PBの独立化や最尤化を禁止'
    grid = tuple(tuple(cell[0][0] for cell in row) for row in cells)
    assert P.B.supported(grid)
    return grid


def main() -> None:
    source, step, origin = inputs()
    grid = deterministic_grid(source['probability'])
    expected = P.B.grid(P.B.Board.from_dict(origin['before_board']))
    assert grid[P.B.HIDDEN_ROWS:] == expected[P.B.HIDDEN_ROWS:]
    assert grid[P.B.HIDDEN_ROWS:] == tuple(map(tuple, source['confirmed']['grid'][P.B.HIDDEN_ROWS:]))
    result = P.B.ChainSimulator(exclude_hidden_row_from_pop=True).simulate(
        P.B.Board.from_dict({'grid': grid}))
    outcome = P.Outcome(P.B.grid(result.final_board), result.chain_count,
                        P.calculate_chain_score(result).total_score, 1.0)
    receipt = dict(kind='non_owned_deterministic_projection_diagnosis/v1', side=SIDE,
        observed_frame=ANCHOR, origin_available_frame=CALL_FRAME, cutoff_frame=CUTOFF,
        origin_trigger_sec=origin['trigger_sec'], origin_object_id=ORIGIN_ID, call_token=TOKEN,
        probability_sha256=source['probability']['sha256'], outcomes=[asdict(outcome)],
        source_row_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        all_cells_probability_one=True, later_identity_metadata_used=False,
        live_belief_created=False, live_owner_verified=False, temporal_probability_propagated=False,
        physical_GT_verified=False, model_evaluated=False, quality_gate_clear=False,
        code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__), Path(P.__file__), Path(P.B.__file__))})
    with (ROOT / 'OPPONENT_ROOT_PROJECTION_v1.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(dict(chain_count=result.chain_count, raw_score=outcome.raw_chain_score,
        worlds=1, live_belief_created=False, quality_gate_clear=False)), flush=True)


if __name__ == '__main__':
    main()
