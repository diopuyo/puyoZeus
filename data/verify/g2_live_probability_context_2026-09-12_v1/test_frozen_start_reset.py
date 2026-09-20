"""凍結会計の公開APIで開始resetの意味を限定検査する。入力は人工。"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import pytest

REPO = Path(__file__).resolve().parents[3]
SNAPSHOT = REPO / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
SIDES = ('p1', 'p2')
START_SCORE = 6


def check_scenario(scenario: str) -> dict:
    """別processのcold srcで、画像認識を伴わない会計遷移を調べる。"""
    sys.path.insert(0, str(SNAPSHOT))
    from src import ojama_accounting as accounting
    from src.board_state_machine import BoardState
    assert Path(accounting.__file__).resolve() == SNAPSHOT / 'src/ojama_accounting.py'
    tracker = accounting.OjamaAccountingTracker()
    tracker.reset()
    if scenario == 'trace':
        return check_trace(tracker, BoardState)
    for side in SIDES:
        if scenario == 'score_drop':
            tracker.on_state_transition(side, BoardState.STABLE, BoardState.STABLE,
                                        accounting.SCORE_RESET_THRESHOLD + START_SCORE, 0.0)
            tracker.on_state_transition(side, BoardState.STABLE, BoardState.STABLE, START_SCORE, 1.0)
        elif scenario == 'menu_entry':
            tracker.on_state_transition(side, BoardState.STABLE, BoardState.MENU, START_SCORE, 1.0)
        else:
            assert scenario == 'menu_continuation'
            tracker.on_state_transition(side, BoardState.MENU, BoardState.MENU, START_SCORE, 1.0)
    gross, snapshot = tracker.get_gross_counters(1.0), tracker.get_snapshot(1.0)
    counts = [gross.boundary_resets_p1, gross.boundary_resets_p2]
    assert counts == ([0, 0] if scenario == 'menu_continuation' else [1, 1])
    assert snapshot.pending_p1_uncapped == snapshot.pending_p2_uncapped == 0
    return dict(scenario=scenario, resets=counts, artificial_input=True, visual_start_verified=False)


def check_trace(tracker: object, states: object) -> dict:
    from dataclasses import asdict
    from types import SimpleNamespace as N
    import start_trace as trace
    seconds, frame = 1.0, trace.FPS
    before = [asdict(tracker._p1), asdict(tracker._p2)]
    recorder = N(_previous_gross=tracker.get_gross_counters(seconds),
                 _previous_final=tracker.get_attack_finalization_counters(seconds), _previous_pending=(0, 0))
    state = dict(ojama_tracker=tracker, accounting_recorder=recorder, t_sec=seconds, fi=frame,
        shared_game=N(game_idx=0, multisignal_mode=True, require_newmatch_evidence=True),
        result=N(is_match_active=False, p1=N(state=states.MENU, score=6), p2=N(state=states.MENU, score=6)),
        pipeline=N(_active_chain_1p=None, _active_chain_2p=None))
    value = trace.capture(state, frame)
    assert value['scores'] == [6, 6] and value['resets'] == [0, 0]
    assert [asdict(tracker._p1), asdict(tracker._p2)] == before
    return dict(scenario='trace', artificial_input=True, visual_start_verified=False)


@pytest.mark.parametrize('scenario', ['menu_entry', 'menu_continuation', 'score_drop', 'trace'])
def test_reset_count_is_not_visual_start_proof(scenario: str) -> None:
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), scenario],
                            cwd=SNAPSHOT, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value['artificial_input'] and not value['visual_start_verified']


if __name__ == '__main__':
    print(json.dumps(check_scenario(sys.argv[1])))
