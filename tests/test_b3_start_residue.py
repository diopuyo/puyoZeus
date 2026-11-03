"""lean 開始前の台帳・着弾・イベントが次の試合に混入しないことを確認する。"""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from src.b3_features import land_group
from src.b3_column_spec import Mask
from src.b3_state import GameState
from src.b3_timing import TimingTable

START_SEC = 10.0
END_SEC = 20.0
NEXT_START_SEC = 30.0
FPS = 60
OJAMA_COUNT = 18


def _frame(t: float, game: int = 0, count: int = 0, state: str = "STABLE") -> dict:
    board = np.zeros((13, 6), dtype=np.int8)
    board.ravel()[-count:] = 9 if count else 0
    sides = [{"side": side, "state": state, "score": 0, "confirmed_board": board.tolist(),
              "icon": {"icons": [], "total_count": 0}} for side in ("1P", "2P")]
    return {"frame": round(t * FPS), "t_sec": t, "available_at_sec": t, "game_idx": game,
            "match_active": True, "sides": sides}


def _state() -> GameState:
    lean = SimpleNamespace(rows_of=lambda side: ())
    return GameState(lean, {0: {"start_sec": START_SEC, "end_sec": END_SEC},
                            1: {"start_sec": NEXT_START_SEC, "end_sec": NEXT_START_SEC + END_SEC}})


def _assert_empty(gs: GameState) -> None:
    assert gs.sl_snap is not None
    for i, st in enumerate(gs.sides, start=1):
        assert gs.sl_snap[f"board_increase_p{i}"] == 0
        assert gs.sl.core.st[i - 1].board_excess_total == 0
        assert gs.sl_snap['history_waiting'] and np.isnan(gs.sl_snap[f"board_excess_p{i}"])
        assert st.land_count == 0 and st.landing is None and st.fall_start is None
        assert st.first_place_t is None and not st.recent_steps and st.episode is None
        row: dict = {}
        land_group(row, st, f"p{i}", gs.t)
        assert row[f"land_count_p{i}"] == 0 and np.isnan(row[f"land_age_p{i}"])


def test_pregame_board_increase_and_landing_ignored() -> None:
    gs = _state()
    for t, count, phase in [(0, 0, "STABLE"), (1, OJAMA_COUNT, "STABLE"),
                            (2, OJAMA_COUNT, "STABLE"), (3, OJAMA_COUNT, "OJAMA_FALL"),
                            (5, 0, "STABLE")]:
        gs.update_frame(_frame(t, count=count, state=phase))
        assert not gs.in_match and gs.sl_snap is None
    assert gs.update_frame(_frame(START_SEC))
    _assert_empty(gs)
    assert not gs.colors.counts.any()


@pytest.mark.parametrize("next_game_early", [False, True])
def test_previous_landing_does_not_cross_lean_start(next_game_early: bool) -> None:
    gs = _state()
    gs.update_frame(_frame(START_SEC))
    gs.update_frame(_frame(START_SEC + 1, state="OJAMA_FALL"))
    gs.update_frame(_frame(END_SEC - 1, count=OJAMA_COUNT))
    gs.update_frame(_frame(END_SEC, count=OJAMA_COUNT))
    assert gs.sides[0].land_count == 1
    assert gs.sl_snap["board_increase_p1"] == OJAMA_COUNT
    # 観測番号が先に変わっても、その区間を試合内として蓄積しない。
    for t in (END_SEC + 1, END_SEC + 3):
        gs.update_frame(_frame(t, game=int(next_game_early), state="OJAMA_FALL"))
    assert gs.update_frame(_frame(NEXT_START_SEC, game=1))
    _assert_empty(gs)


def test_delayed_pregame_event_is_ignored() -> None:
    gs = _state()
    gs.update_frame(_frame(START_SEC))
    event = {"game_idx": 0, "side": "1P", "kind": "placement_signal",
             "observed_at_sec": START_SEC - 1}
    assert gs.apply_event(event) == 0
    _assert_empty(gs)
    assert gs.apply_event(dict(event, observed_at_sec=START_SEC)) == 1
    assert gs.first_move_t() == START_SEC


def test_scoreless_pregame_inside_lean_range_is_ignored() -> None:
    gs = _state()
    for t, count in ((START_SEC, 0), (START_SEC + 1, OJAMA_COUNT), (START_SEC + 2, OJAMA_COUNT)):
        frame = _frame(t, count=count)
        for side in frame["sides"]:
            side["score"] = None
        gs.update_frame(frame)
        assert not gs.in_match and gs.sl_snap is None
    gs.update_frame(_frame(START_SEC + 3))
    _assert_empty(gs)
    assert gs.game_start_t == START_SEC + 3


def test_in_game_transient_inactivity_does_not_reset() -> None:
    gs = _state()
    gs.update_frame(_frame(START_SEC))
    gs.update_frame(_frame(START_SEC + 1, state="OJAMA_FALL"))
    gs.update_frame(_frame(START_SEC + 3, count=OJAMA_COUNT))
    gs.update_frame(dict(_frame(START_SEC + 4, count=OJAMA_COUNT), match_active=False))
    assert gs.apply_event({"game_idx": 0, "side": "1P", "kind": "placement_signal",
                           "observed_at_sec": START_SEC + 4}) == 1
    assert not gs.update_frame(_frame(START_SEC + 5, count=OJAMA_COUNT))
    assert gs.sides[0].land_count == 1


def test_valid_in_game_history_matches_clean_replay() -> None:
    contaminated, clean = _state(), _state()
    for t in (0.0, 1.0, 2.0):
        contaminated.update_frame(_frame(t, count=OJAMA_COUNT, state="OJAMA_FALL"))
    for t, count, phase in [(START_SEC, 0, "STABLE"), (START_SEC + 1, 0, "OJAMA_FALL"),
                            (START_SEC + 3, OJAMA_COUNT, "STABLE"),
                            (START_SEC + 4, OJAMA_COUNT, "STABLE")]:
        frame = _frame(t, count=count, state=phase)
        contaminated.update_frame(frame)
        clean.update_frame(frame)
        assert contaminated.sl_snap.keys() == clean.sl_snap.keys()
        for key, value in contaminated.sl_snap.items():
            other = clean.sl_snap[key]
            assert np.isnan(other) if isinstance(value, float) and np.isnan(value) else value == other
        for left, right in zip(contaminated.sides, clean.sides):
            assert left.land_count == right.land_count
            if left.landing is not None:
                assert left.landing.finalize_t == right.landing.finalize_t
                assert left.landing.board_net == right.landing.board_net


def test_generated_first_rows_have_no_residue(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import b3_rows
    frames = [_frame(0), _frame(1, count=OJAMA_COUNT), _frame(2, count=OJAMA_COUNT),
              _frame(3, state="OJAMA_FALL"), _frame(5), _frame(START_SEC),
              _frame(START_SEC + 1, state="OJAMA_FALL"), _frame(END_SEC - 1, count=OJAMA_COUNT),
              _frame(END_SEC, count=OJAMA_COUNT), _frame(END_SEC + 1, game=1, state="OJAMA_FALL"),
              _frame(END_SEC + 3, game=1), _frame(NEXT_START_SEC, game=1)]
    monkeypatch.setattr(b3_rows, "iter_frames", lambda *args: iter(frames))
    monkeypatch.setattr(b3_rows, "load_lean", lambda *args: SimpleNamespace(rows_of=lambda side: ()))
    monkeypatch.setattr(b3_rows, "load_events", lambda *args: [])
    rows, _ = b3_rows.generate_rows(Path(), "synthetic", TimingTable(), winners={0: 1, 1: 0},
                                    match_mapping=_state().match_mapping)
    for game in (0, 1):
        first = next(r for r in rows if r["game_idx"] == game)
        for side in ("p1", "p2"):
            assert first[f"sl_board_increase_{side}"] == 0
            assert np.isnan(first[f"sl_board_excess_{side}"])
            assert first[f"mk_sl_{side}"] == Mask.NOT_YET_AVAILABLE
            assert first[f"land_count_{side}"] == 0
            assert np.isnan(first[f"land_board_net_{side}"]) and np.isnan(first[f"land_age_{side}"])
    assert any(r["land_count_p1"] == 1 for r in rows if r["game_idx"] == 0)
