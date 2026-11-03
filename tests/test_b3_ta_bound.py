"""時間軸の上限補完・既存計算の維持・側交換を検証する。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.b3_column_spec import COLUMN_NAMES, Mask, SPEC_BY_NAME, Kind, swap_sides, nonsat
from src.b3_state import Episode, Fragment, SideState
from src.b3_timeaxis import (_bound_q, _episode_generated, _origin_board, _score_upper,
                             time_axis_features)
from src.b3_timing import TimingTable
from src.b3_train_model import feature_columns, to_matrix
from src.b3_search import HMAX_TOTAL_HANDS, IdealSearchResult, ideal_fire_search, points_to_ojama
from src.scoring import ALL_CLEAR_BONUS, BASE_SCORE_PER_PUYO, MAX_BONUS_MULTIPLIER


def sides() -> tuple[SideState, SideState]:
    a, b = SideState("1P"), SideState("2P")
    a.episode = Episode(0, 0, pred_l=2, pred_score=1000)
    b.board = np.zeros((13, 6), dtype=np.int8)
    b.board[-1, :4] = 1
    return a, b


def evaluate(a: SideState, b: SideState, **kwargs: object) -> dict[str, float]:
    return time_axis_features(a, b, 1, 1, (1, 2, 3, 4), TimingTable(),
                              1100, sl_b_pending=1100, **kwargs)


def test_receiver_unknown_fills_bound_and_preserves_reason() -> None:
    a, b = sides()
    b.episode = Episode(0, 0, frags=[Fragment("f", 0, b.board.copy(), None)])
    b.episode.steps = [(0.5, 4, 40)]
    row = evaluate(a, b)
    assert row["ta_bound"] == 1 and row["__mk"] == Mask.UNPREDICTABLE
    assert np.isfinite(row["ta_q_med"]) and np.isfinite(row["ta_m_sl_med"])
    assert row["ta_m_sl_med"] == row["ta_q_med"] - 1100
    assert row["ta_q_med_nf"] == nonsat(row["ta_q_med"])


def test_unknown_attacker_time_uses_generous_horizon() -> None:
    a, b = sides()
    a.episode.pred_l = a.episode.pred_score = float("nan")
    row = evaluate(a, b)
    assert row["ta_bound"] == 1 and row["ta_H_med"] >= HMAX_TOTAL_HANDS
    assert row["ta_H_fast"] >= HMAX_TOTAL_HANDS
    assert np.isfinite(row["ta_q_med"]) and np.isnan(row["ta_i_pred"])


def test_predictable_values_use_unchanged_search() -> None:
    a, b = sides()
    row = evaluate(a, b)
    result = ideal_fire_search(b.board, b.queue, (1, 2, 3, 4))
    q = points_to_ojama(result.at(int(row["ta_H_med"]))[0], 1 + row["ta_a_rem_sec"])
    assert row["ta_bound"] == 0 and row["ta_q_med"] == q
    assert row["ta_m_sl_med"] == q - 1100 and row["__mk"] == Mask.OK


def test_bound_flag_appended_and_swapped_without_complement() -> None:
    assert COLUMN_NAMES[-4:] == ["ta_bound_p1", "ta_bound_p2", "ep_denominator_formula_p1", "ep_denominator_formula_p2"]
    row = {"ta_bound_p1": 1, "ta_bound_p2": 0}
    assert swap_sides(row) == {"ta_bound_p1": 0, "ta_bound_p2": 1}
    assert swap_sides(swap_sides(row)) == row
    assert SPEC_BY_NAME["ta_bound_p1"].kind == Kind.SIDE_ABS
    assert SPEC_BY_NAME["ta_bound_p1"].norm == "0-1"
    assert "ta_bound_p1" in feature_columns(COLUMN_NAMES)
    assert to_matrix(pd.DataFrame([row]), list(row)).tolist() == [[1.0, 0.0]]


def test_observed_attacker_stages_limit_remaining_time() -> None:
    a, b = sides()
    a.episode = Episode(0, 0, frags=[Fragment("a", 0, b.board.copy(), None)],
                        steps=[(0.5, 4, 40)])
    row = evaluate(a, b)
    assert row["ta_bound"] == 1
    assert row["ta_a_rem_sec"] == pytest.approx(TimingTable().tail_sec - 0.5)
    assert row["ta_H_med"] < HMAX_TOTAL_HANDS


def test_origin_uses_first_confirmed_prefire_board() -> None:
    _, b = sides()
    before = np.zeros_like(b.board)
    ep = Episode(0, 0, frags=[Fragment("first", 0, before, {"board": b.board.tolist()}),
                             Fragment("later", 1, before, {"board": before.tolist()})])
    np.testing.assert_array_equal(_origin_board(ep), b.board)


def test_current_chain_does_not_add_generated_or_prefire_search_twice() -> None:
    search = IdealSearchResult((1000.0, 3000.0), (3, 4), 2, 2, False)
    q = _bound_q(20, 3, 1000, 2, 26, 10, True, search)
    score = 24 * BASE_SCORE_PER_PUYO * MAX_BONUS_MULTIPLIER + 3 * ALL_CLEAR_BONUS
    assert q == 26 + points_to_ojama(score, 10) + 3
    assert q - _bound_q(20, 3, 1000, 2, 0, 10, True, search) == 26


def test_unknown_attacker_origin_uses_available_board() -> None:
    a, b = sides()
    a.episode = Episode(0, 0, steps=[(0.5, 4, 40)])
    a.board = b.board.copy()
    row = evaluate(a, b)
    assert row["ta_H_med"] < HMAX_TOTAL_HANDS and row["ta_H_fast"] < HMAX_TOTAL_HANDS
    assert row["ta_a_rem_sec"] == pytest.approx(TimingTable().stage_sec + TimingTable().tail_sec)


@pytest.mark.parametrize("missing", [False, True])
def test_finished_attacker_allows_only_landing_placement(missing: bool) -> None:
    a, b = sides()
    a.episode = None if missing else Episode(0, 0, closed_t=0.5)
    b.episode = Episode(0, 0, frags=[Fragment("f", 0, b.board.copy(), None)])
    row = evaluate(a, b)
    assert row["ta_bound"] == 1 and row["ta_a_rem_sec"] == 0
    assert row["ta_H_med"] == row["ta_H_fast"] == 1
    assert row["ta_b_busy"] == 0 and row["ta_i_pred"] == 0


def test_finished_attacker_preserves_predictable_receiver() -> None:
    a, b = sides()
    a.episode.closed_t = 0.5
    b.episode = Episode(0, 0, pred_l=2, pred_final=b.board.tobytes())
    row = evaluate(a, b)
    assert row["ta_bound"] == 0 and row["__mk"] == Mask.NOT_APPLICABLE


def test_observed_erasures_reduce_post_chain_material() -> None:
    from src.b3_timeaxis import _bound_material
    _, b = sides()
    b.episode = Episode(0, 0, frags=[Fragment("f", 0, b.board.copy(), None)],
                        steps=[(0.5, 4, 40)])
    remaining, _, _, search = _bound_material(b, (1, 2, 3, 4), 20)
    assert remaining == 0 and search is None


def test_generated_is_episode_local_and_includes_closed_fragments() -> None:
    _, b = sides()
    b.episode = Episode(10, 10)
    step = lambda t, n: SimpleNamespace(t_formula=t, generated=n)
    old = SimpleNamespace(side=1, steps=[step(1, 900)])
    closed = SimpleNamespace(side=1, steps=[step(10, 20)])
    wrong = SimpleNamespace(side=0, steps=[step(10, 800)])
    opened = SimpleNamespace(steps=[step(11, 30)])
    ledger = SimpleNamespace(core=SimpleNamespace(chains=[old, closed, wrong],
                              st=[SimpleNamespace(chain=None), SimpleNamespace(chain=opened)]))
    assert _episode_generated(ledger, b) == 50


@pytest.mark.parametrize("cells", [0, 4, 12, 30])
def test_relaxed_bound_is_monotone_in_material_and_hands(cells: int) -> None:
    assert _score_upper(cells + 2) >= _score_upper(cells)
    assert _bound_q(cells, 5, 1000, 3, 20, 10, True) >= _bound_q(cells, 5, 1000, 1, 20, 10, True)


def test_inactive_and_unobserved_do_not_claim_bound() -> None:
    a, b = sides()
    a.episode = None
    assert evaluate(a, b)["ta_bound"] == 0
    a.episode = Episode(0, 0)
    b.board = None
    row = evaluate(a, b)
    assert row["ta_bound"] == 0 and row["__mk"] == Mask.UNPREDICTABLE
