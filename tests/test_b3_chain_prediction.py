"""断片結合後も式の観測と整合する予測だけを公開する回帰検査。"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.b3_chain_prediction import EpisodePrediction, origin_candidates
from src.b3_column_spec import Mask, swap_sides
from src.b3_features import chain_group
from src.b3_state import Episode, SideState
from src.b3_timeaxis import time_axis_features
from src.b3_timing import TimingTable


def option(prefix: tuple[int, ...], color: int = 0, weight: int = 1) -> dict:
    return dict(prefix=prefix, score=prefix[-1], board=np.full((13, 6), color).tolist(), weight=weight)


def formula(st: SideState, t: float, product: int) -> None:
    st.on_formula(dict(t_sec=t, observed_at_sec=t + 0.01, erased=4, product=product))


def start(st: SideState, t: float, fire_id: str = "a") -> None:
    st.on_fire_start(dict(fire_id=fire_id, trigger_sec=t, observed_at_sec=t + 0.02,
                          chain_before=np.zeros((13, 6), dtype=np.int8).tolist()))


def test_scores_follow_each_observed_stage_and_same_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    options = [option((40, 360, 1000)), option((100, 420, 1060)), option((100, 740, 2020))]
    monkeypatch.setattr("src.b3_state.origin_candidates", lambda *_: (deepcopy(options), ""))
    st = SideState("1P")
    start(st, 1.0)
    assert st.episode.pred_score == 1000
    formula(st, 1.1, 100)
    assert st.episode.pred_score == 1060
    formula(st, 2.1, 640)
    assert st.episode.pred_score == 2020
    formula(st, 3.1, 1280)
    assert st.episode.prediction.observed_score == st.episode.pred_score == 2020
    assert st.episode.pred_l == 3
    assert st.episode.prediction.available_at == 3.11


@pytest.mark.parametrize("products", [(40, 320), (100,), (40, 640)])
def test_conflicting_count_or_each_stage_discards_candidate(products: tuple[int, ...]) -> None:
    pred = EpisodePrediction(options=[option((40,))] if len(products) > 1 and products[1] == 320
                             else [option((40, 360))], reason="")
    for i, score in enumerate(products):
        pred.observe(score, float(i))
    assert not pred.options and pred.reason == "observation_conflict"
    assert np.isnan(pred.score) and np.isnan(pred.length) and pred.final is None


def test_a1b_selects_max_chain_and_same_board_not_max_score() -> None:
    pred = EpisodePrediction(options=[option((40, 360, 1000), 2), option((40, 2000), 3, 20)], reason="")
    pred.observe(40, 1.0)
    assert (pred.length, pred.score) == (3, 1000)
    assert np.all(np.frombuffer(pred.final, dtype=np.int8) == 2)


def test_fragment_does_not_reseed_and_prefix_is_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def candidates(*args: object) -> tuple[list[dict], str]:
        calls.append(args)
        return [option((40, 360, 1000))], ""

    monkeypatch.setattr("src.b3_state.origin_candidates", candidates)
    st = SideState("1P")
    start(st, 1.0)
    formula(st, 1.1, 40)
    before: dict = {}
    chain_group(before, st, "p1", 1.2, TimingTable(), 0.0)
    prefix = deepcopy(st)
    start(st, 2.0, "b")
    formula(st, 2.1, 640)
    after: dict = {}
    chain_group(after, prefix, "p1", 1.2, TimingTable(), 0.0)
    assert before == after and len(calls) == 1
    assert len(st.episode.frags) == 2 and st.episode.pred_final is None
    assert st.episode.prediction.reason == "observation_conflict"


def test_backfill_uses_acceptance_time_and_publishes_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.b3_state.origin_candidates", lambda *_: ([option((40,))], ""))
    st = SideState("1P")
    formula(st, 1.0, 100)
    start(st, 1.0)
    row: dict = {}
    chain_group(row, st, "p1", 1.02, TimingTable(), 0.0)
    assert row["av_chain_p1"] == 1.02
    assert row["chain_pred_final_p1"] is None and np.isnan(row["chain_pred_score_p1"])
    assert row["chain_pred_reason_p1"] == "observation_conflict"
    assert row["chain_observed_score_p1"] == 100
    assert row["mk_chain_p1"] == Mask.UNPREDICTABLE
    chain_group(row, SideState("2P"), "p2", 1.02, TimingTable(), 0.0)
    assert swap_sides(row)["chain_observed_score_p2"] == 100


@pytest.mark.parametrize("kind,reason", [("missing", "missing_origin"), ("unknown", "unknown_origin"),
                                        ("palette", "unknown_palette"), ("blocked", "no_completion_candidates")])
def test_missing_candidates_are_explicit(kind: str, reason: str) -> None:
    board = None if kind == "missing" else np.full((13, 6), 10 if kind == "unknown" else 0, dtype=np.int8)
    if kind == "blocked":
        board[:] = 9
    colors = () if kind == "palette" else (1, 2, 3, 4)
    options, actual = origin_candidates(board, colors, ())
    assert not options and actual == reason


def test_direct_origin_and_completion_enumeration() -> None:
    board = np.zeros((13, 6), dtype=np.int8)
    board[-1, :4] = 1
    direct, reason = origin_candidates(board, (), ())
    assert reason == "" and direct[0]["prefix"] == (40,)
    board[-1, 3] = 0
    candidates, reason = origin_candidates(board, (1, 2, 3, 4), ())
    assert reason == "" and candidates
    pred = EpisodePrediction(options=candidates, reason=reason)
    pred.observe(40, 1.0)
    assert pred.score >= 40 and pred.final is not None


def test_unknown_opponent_completion_never_falls_back_to_frozen_board() -> None:
    a, b = SideState("1P"), SideState("2P")
    a.episode = Episode(0.0, 0.0, pred_l=5, pred_score=5000)
    b.episode = Episode(0.0, 0.0)
    b.board = np.zeros((13, 6), dtype=np.int8)
    result = time_axis_features(a, b, 1.0, 1.0, (1, 2, 3, 4), TimingTable(), 0.0)
    assert result["__mk"] == Mask.UNPREDICTABLE and np.isfinite(result["ta_q_med"])
    assert result["ta_bound"] == 1
