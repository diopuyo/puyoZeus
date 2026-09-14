"""実保存グリッドと原SMの状態遷移で、短い落下の票不足を反証する限定CPU。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT.parents[2] / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
sys.path.insert(0, str(SNAPSHOT))
import numpy as np
from src import board_state_machine as S
from src.board import Board


def board(value: list) -> Board:
    result = Board()
    result._grid = np.array(value, dtype=np.uint8)
    return result


def case(rows: dict, resume: bool) -> dict[str, Any]:
    sm = S.BoardStateMachine(enable_stable_resume_gate=resume, enable_gravity_filter_support=True)
    sm._ctx.state = S.BoardState.STABLE
    sm._ctx.confirmed_board = board(rows[35166]['sm'])
    trace = []
    for frame in (35168, 35170, 35172, 35174):
        row = rows[frame]
        signals = S.DetectorSignals(cnn_board=board(row['cnn']), time_sec=frame / 60, is_match_active=True,
                    effect_gate_window_active=False, match_just_started=False)
        sm._ctx.frame_idx = frame
        state = S.BoardState.TSUMO_FALL if row['observation']['state'] == 'tsumo_fall' else S.BoardState.STABLE
        before = len(sm._ctx.non_stable_cnn_history)
        sm._apply_transition(state, signals)
        trace.append(dict(frame=frame, history_before=before, history_after=len(sm._ctx.non_stable_cnn_history),
                          col4=sm._ctx.confirmed_board._grid[:3, 4].tolist(),
                          same_saved_sm=sm._ctx.confirmed_board._grid.tolist() == row['sm']))
    return dict(resume_guard=resume, default_votes=sm._empty_to_color_min_votes, trace=trace)


def main() -> None:
    assert Path(S.__file__).resolve() == SNAPSHOT / 'src/board_state_machine.py'
    rows = {r['frame']: r for r in json.loads((ROOT / 'ACTUAL_INPUTS.json').read_bytes())['rows']}
    guarded, control = case(rows, True), case(rows, False)
    assert all(row['same_saved_sm'] for row in guarded['trace'])
    assert guarded['trace'][2]['history_before'] == 1
    assert guarded['trace'][-1]['col4'][1:] == [0, 0]
    assert control['trace'][-1]['col4'][1:] == [4, 3]
    result = dict(original_sm_reexecuted=True, guarded=guarded, guard_off_counterfactual=control,
                  full_pipeline_reexecuted=False, detector_decisions_replayed=True,
                  production_permission=False, quality_gate_clear=False)
    with (ROOT / 'NATIVE_MERGE_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
