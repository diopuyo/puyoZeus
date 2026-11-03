"""現在層への復帰、合成の非累積、死亡後の発火拒否を検証する。"""
from __future__ import annotations

from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board import Board, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.exchange_event_layers import ExchangeEvaluationLayers
from src.exchange_event_overlay import ConfirmedSide, ExchangeEventOverlay
from src.exchange_event_tracker import ExchangeChainRecord, ExchangeRecord

CURRENT, PREDICTED = .2, .8


def context() -> tuple:
    board = Board()
    history = [[ConfirmedSide(2., board.copy(), np.ones(4, int))] for _ in range(2)]
    chain = ExchangeChainRecord("1P", 1, 0., 0., predicted_final_score=100., predicted_chain_count=2)
    record = ExchangeRecord(1, 0, 0., chains=[chain])
    tracker = NS(probability=PREDICTED, source="S3_provisional", current=record,
                 resolver=NS(active=lambda: [], resolved=lambda: []), _chain_aliases={}, layer_rows=[])
    overlay = NS(tracker=tracker, _history=history, _game=0, _landing_projection=NS(death_record=None))
    result = NS(p1=NS(state=BoardState.STABLE), p2=NS(state=BoardState.STABLE))
    layers = ExchangeEvaluationLayers()
    layers._current = lambda *args: CURRENT
    return layers, overlay, result, chain


def test_combination_is_not_accumulated() -> None:
    layers, overlay, result, _ = context()
    layers.apply(overlay, result, None, 2.)
    assert overlay.tracker.probability == pytest.approx(.5)
    result.confirmed_dead_sides = ()
    assert not layers.before(overlay, result, 3.)
    layers.apply(overlay, result, None, 3.)
    assert overlay.tracker.probability == pytest.approx(.5)
    assert overlay.tracker.layer_eval["p1_prediction"] == PREDICTED


@pytest.mark.parametrize("reason", ["unknown_board", "observed_score_exceeds_prediction",
                                   "observed_chain_exceeds_prediction", "prediction_board_mismatch"])
def test_untrusted_prediction_returns_current(reason: str) -> None:
    layers, overlay, result, chain = context()
    if reason == "unknown_board":
        overlay._history[0][-1].board._grid[-1, 0] = COLOR_UNKNOWN
    elif reason == "observed_score_exceeds_prediction":
        chain.formula_total = chain.predicted_final_score+1
    elif reason == "observed_chain_exceeds_prediction":
        overlay.tracker.resolver.active = lambda: [NS(chain_id=1, step_count=3, growth_observed=True)]
    else:
        chain.end_signal_sec = 1.
        chain.predicted_final_board = Board()._grid.tolist()
        overlay._history[0][-1].board._grid[-1, 0] = 1
    layers.apply(overlay, result, None, 2.)
    assert overlay.tracker.probability == CURRENT
    assert reason in overlay.tracker.layer_eval["prediction_reasons"]
    assert overlay.tracker.layer_eval["p1_prediction"] == PREDICTED


def test_observed_death_blocks_baseline_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    overlay = ExchangeEventOverlay(NS(), lambda *args: None, lambda *args: None, e16=True)
    result = NS(p1=NS(chain_event=None), p2=NS(chain_event=NS(trigger_sec=1.)),
                confirmed_dead_sides=("2P",))
    monkeypatch.setattr(overlay._e16, "apply", lambda *args: None)
    overlay.update(result, None, None, 1., 0)
    assert not overlay.tracker.records
    assert overlay.tracker.diagnostics[0]["reason"] == "E16_fire_after_observed_death"
    result.confirmed_dead_sides = ()
    overlay.update(result, None, None, 2., 0)
    assert not overlay.tracker.records


def test_resolved_observed_chain_alias_keeps_excess_reason() -> None:
    layers, overlay, result, _ = context()
    overlay.tracker._chain_aliases = {7: 1}
    overlay.tracker.resolver.resolved = lambda: [NS(chain_id=7, step_count=3, growth_observed=True)]
    layers.apply(overlay, result, None, 3.)
    assert overlay.tracker.probability == CURRENT
    assert "observed_chain_exceeds_prediction" in overlay.tracker.layer_eval["prediction_reasons"]


def test_future_completion_is_not_compared_before_end() -> None:
    layers, overlay, result, chain = context()
    chain.predicted_final_board = Board()._grid.tolist()
    overlay._history[0][-1].board._grid[-1, 0] = 1
    layers.apply(overlay, result, None, 2.)
    assert "prediction_board_mismatch" not in overlay.tracker.layer_eval["prediction_reasons"]
