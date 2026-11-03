"""PC2全件監査で発見した生成器矛盾の回帰検査。"""
from __future__ import annotations

import numpy as np
import pytest

from src.b3_chain_prediction import origin_candidates
from src.b3_column_spec import Mask, swap_sides
from src.b3_features import board_group, episode_group
from src.b3_score_ledger import ScoreLedgerV2, margin_rate, margin_rate_table
from src.b3_state import Episode, Fragment, SideState
from src.b3_timeaxis import time_axis_features
from src.b3_timing import TimingTable
from src.scoring import compute_effective_rate

RATES = (70, 52, 39, 29, 21, 15, 11, 8, 6, 4, 3, 2, 1)


def grid(count: int) -> np.ndarray:
    board = np.zeros((13, 6), dtype=np.int8)
    board.flat[-count:] = 1 if count else 0
    return board


@pytest.mark.parametrize("step,rate", enumerate(RATES))
def test_rate_uses_each_truncated_result(step: int, rate: int) -> None:
    elapsed = 0 if step == 0 else 95.5 + (step - 1) * 16 + 0.01
    assert compute_effective_rate(elapsed, from_first_move=True) == margin_rate(elapsed) == rate
    assert margin_rate_table()[step] == rate


def closed(consumed: float, before: int = 13, pre: int = 15) -> SideState:
    st = SideState("1P")
    frag = Fragment("a", 1.0, grid(before), {"board": grid(pre).tolist()}, consumed_board=consumed)
    ep = Episode(1.0, 1.0, frags=[frag], steps=[(1.0, 14.0, 140.0)], closed_t=3.0,
                 pred_l=1.0, pred_final=grid(1).tobytes(), denominator=float(before))
    st.closed.append(ep)
    return st


def test_formula_completed_material_and_board_pre_have_own_denominators() -> None:
    row: dict = {}
    episode_group(row, closed(14), "p1", 4.0, 0.0)
    assert row["ep_denominator_p1"] == row["ep_denominator_formula_p1"] == 15
    assert row["pf_before_color_p1"] == 13
    assert row["ep_ratio_formula_p1"] == row["ep_ratio_board_p1"] == 14 / 15


@pytest.mark.parametrize("consumed", [-1, -7, 16])
def test_impossible_board_consumption_is_missing(consumed: int) -> None:
    row: dict = {}
    episode_group(row, closed(consumed), "p1", 4.0, 0.0)
    assert row["mk_ep_b_p1"] == Mask.RECOGNITION_CONFLICT
    assert np.isnan(row["ep_consumed_board_p1"]) and np.isnan(row["ep_ratio_board_p1"])


def test_negative_fragment_cannot_be_hidden_by_positive_fragment() -> None:
    st = closed(10)
    st.closed[0].frags.append(Fragment("b", 2.0, grid(3), None, consumed_board=-1))
    row: dict = {}
    episode_group(row, st, "p1", 4.0, 0.0)
    assert row["mk_ep_b_p1"] == Mask.RECOGNITION_CONFLICT
    assert np.isnan(row["ep_consumed_board_p1"])


def test_formula_material_outside_zero_to_two_pairs_is_missing() -> None:
    st = closed(1, before=1)
    row: dict = {}
    episode_group(row, st, "p1", 4.0, 0.0)
    assert row["mk_ep_f_p1"] == Mask.RECOGNITION_CONFLICT
    assert np.isnan(row["ep_denominator_formula_p1"]) and np.isnan(row["ep_ratio_formula_p1"])


def five_colors() -> SideState:
    st = SideState("1P")
    st.board = grid(0)
    st.board[-1, :5] = (1, 2, 3, 4, 5)
    st.board_t = 1.0
    return st


def test_five_colors_keep_counts_but_never_search(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("5色盤面で探索してはいけない")
    monkeypatch.setattr("src.b3_features.ideal_fire_search", forbidden)
    row: dict = {}
    st = five_colors()
    board_group(row, st, "p1", 2.0, 0.0, (1, 2, 3, 4))
    assert row["cells_color_p1"] == 5 and row["mk_board_p1"] == Mask.RECOGNITION_CONFLICT
    assert np.isnan(row["ind_conn_pair_count_p1"]) and np.isnan(row["k1_ojama_p1"])
    assert origin_candidates(st.board, (1, 2, 3, 4), ())[1] == "recognition_conflict"
    ta = time_axis_features(st, SideState("2P"), 2, 0, (1, 2, 3, 4), TimingTable(), 0)
    assert ta["__mk"] == Mask.RECOGNITION_CONFLICT and np.isnan(ta["ta_q_med"])


def test_score_regression_and_missing_read_cannot_lower_baseline() -> None:
    st = SideState("1P")
    for i, score in enumerate((415, 0, None, 20, 415, 420)):
        st.update_frame(float(i), {"state": "STABLE", "score": score}, True)
        if score in (0, 20):
            assert st.score == 415 and st.score_conflict
    assert st.score == 420 and not st.score_conflict


def test_ledger_persistent_lower_score_does_not_recount_recovery() -> None:
    core = ScoreLedgerV2()
    core.st[0].adopted = 415
    for i in range(200):
        assert core._adopt_score(0, 20, i * 0.02) == []
    assert core.st[0].adopted == 415
    assert core._adopt_score(0, 415, 4.0) == []
    assert core._adopt_score(0, 415, 4.02) == []


def test_new_denominator_and_conflict_masks_swap_without_sign_change() -> None:
    row = {"ep_denominator_formula_p1": 15, "ep_denominator_formula_p2": 30,
           "mk_board_p1": Mask.RECOGNITION_CONFLICT, "mk_board_p2": Mask.OK}
    swapped = swap_sides(row)
    assert swapped["ep_denominator_formula_p2"] == 15
    assert swapped["mk_board_p2"] == Mask.RECOGNITION_CONFLICT
    assert swap_sides(swapped) == row
