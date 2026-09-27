"""応手確率の境界、観測区間打ち切り、特徴の左右対称性を検証する。"""
from __future__ import annotations

from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board import Board
from src.exchange_event_overlay import ConfirmedSide, ExchangeEventOverlay
from src.exchange_event_landing import ExchangeLandingProjection, future_send
from src.landing_counter_probability import response_features, LogisticResponseProbability, COLUMNS
from scripts.e19_response_training_rows import observed_label, estimated_send


@pytest.mark.parametrize("reply,expected", [(2., 1), (3., 1), (3.5, None), (4., None), (5., 0)])
def test_label_only_outside_observed_landing_interval(reply: float, expected: int | None) -> None:
    assert observed_label(1., [(3., 4.)], [reply], 8., 10.)[0] == expected


@pytest.mark.parametrize("arrivals,next_attack", [([], 8.), ([(.5, 2.)], 8.), ([(3., 4.)], 3.8)])
def test_missing_or_other_attack_is_not_a_negative(arrivals: list, next_attack: float) -> None:
    assert observed_label(1., arrivals, [], next_attack, 10.)[0] is None


@pytest.mark.parametrize("receiver", [0, 1])
@pytest.mark.parametrize("probability", [0., .25, 1.])
def test_probability_blends_two_gfe_without_changing_current(receiver: int, probability: float,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    model = NS(predict=lambda x: probability)
    projection = ExchangeLandingProjection(counter_probability_model=model)
    monkeypatch.setattr(projection, "_chaining", lambda *a: False)
    latest = tuple(ConfirmedSide(1., Board(), np.ones(4, int)) for _ in range(2))
    incoming, sends = [0, 0], [None, None]
    incoming[receiver], sends[receiver] = 20, 30
    value = dict(response_selected=True, response_send=sends, gfe_no_response_p1=.2, gfe_response_p1=.8)
    result = projection._counter_probability(NS(tracker=NS(probability=.3)), latest, incoming, (2, 2), value, 2.)
    assert result["gfe_weighted_p1"] == pytest.approx(.2+.6*probability)
    assert result["counter_probability"][receiver] == probability


def test_active_receiver_outside_training_support_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    projection = ExchangeLandingProjection(counter_probability_model=NS(predict=lambda x: pytest.fail()))
    monkeypatch.setattr(projection, "_chaining", lambda *a: True)
    value = dict(response_selected=True, response_send=[30, None], gfe_no_response_p1=.2, gfe_response_p1=.8)
    result = projection._counter_probability(NS(tracker=None), (), [20, 0], (2, 2), value, 2.)
    assert result["gfe_weighted_p1"] == .2
    assert result["counter_probability_reasons"][0] == "active_receiver_outside_training_support"


def test_features_bounded_and_receiver_oriented() -> None:
    board = Board()
    board._grid[-2:, 2] = 9
    x = response_features(board._grid, np.array([0, 1, 2, 3]), 900, 30, 10, 50, False)
    assert x.shape == (len(COLUMNS),) and ((0 <= x) & (x <= 1)).all()
    assert x[2] == x[3] == 2/13 and x[6] == .75


def test_logistic_zero_and_extreme_logits() -> None:
    model = LogisticResponseProbability(np.zeros(len(COLUMNS)), np.ones(len(COLUMNS)),
                                         np.ones(len(COLUMNS)), 0.)
    assert model.predict(np.zeros(len(COLUMNS))) == .5
    assert model.predict(np.full(len(COLUMNS), 1000.)) == 1.
    assert model.predict(np.full(len(COLUMNS), -1000.)) == 0.


@pytest.mark.parametrize("hands", [1, 3])
def test_native_training_search_matches_e18(hands: int) -> None:
    grid = np.zeros((13, 6), np.int8)
    grid[-1, :3], grid[-2, :2] = [1, 1, 2], [2, 2]
    queue = np.array([1, 2, 3, 4], np.int8)
    assert estimated_send(grid, queue, 0., hands) == future_send(
        grid.tobytes(), grid.shape, grid.dtype.str, tuple(queue), hands, 0.)


def test_probability_is_off_without_explicit_flag() -> None:
    model = NS(predict=lambda x: .3)
    overlay = ExchangeEventOverlay(NS(), None, None, counter_probability_model=model)
    assert overlay._landing_projection.counter_probability_model is None
    overlay = ExchangeEventOverlay(NS(), None, None, landing_counter_prob=True, counter_probability_model=model)
    assert overlay._landing_projection.counter_probability_model is model
    assert overlay._e16 is None
