"""不確かな連鎖の死判定保留・発火時の完走予測・確定盤面更新を検証する。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src import exchange_event_landing as landing
from src import exchange_event_overlay as adapter
from src.board import Board
from src.board_state_machine import BoardState
from src.exchange_event_tracker import ExchangeChainRecord
from src.exchange_event_features import SIDE_COLUMNS
from tests.test_e10b_exchange_landing import setup_projection
from tests.test_exchange_event_tracker import fire, static, tracker
from tests.test_exchange_event_overlay import Models, Signals, build_static, result


@pytest.mark.parametrize("formula,predicted,expected", [(100, 78150, 78150), (90000, 78150, 90000), (100, 0, 100)])
def test_prediction_is_observed_score_floor(formula: int, predicted: int, expected: int) -> None:
    chain = ExchangeChainRecord('1P', 1, 0, 0, formula_total=formula, predicted_final_score=predicted)
    assert chain.provisional_score == expected
    chain.end_signal_sec, chain.score_ready_sec, chain.score_delta = 1, 1, 1750
    assert chain.provisional_score == 1750


def test_score_notification_does_not_end_active_prediction() -> None:
    chain = ExchangeChainRecord('1P', 1, 0, 0, formula_total=100,
        predicted_final_score=78150, score_ready_sec=1, score_delta=100)
    assert chain.provisional_score == 78150


@pytest.mark.parametrize("state_only", [True, False])
def test_uncertain_active_receiver_never_dies(tracker: object, monkeypatch: pytest.MonkeyPatch, state_only: bool) -> None:
    projection, overlay, observed, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    if state_only:
        observed.p2.state = BoardState.CHAIN
    else:
        fire(tracker, 2.1, (None, 2.1))
    projection.update(overlay, observed, snapshot, 2.1)
    assert projection.death is None
    assert tracker.source != 'unavoidable_death'


def test_same_chain_activity_releases_death(tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, observed, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, observed, snapshot, 2)
    assert projection.death is not None
    observed.p2.state = BoardState.CHAIN
    projection.update(overlay, observed, snapshot, 2.1)
    assert projection.death is None and tracker.source != 'unavoidable_death'
    observed.p2.state = BoardState.STABLE
    projection.update(overlay, observed, snapshot, 2.2)
    assert projection.death is not None


def test_new_activity_after_timeout_clears_display(tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, observed, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, observed, snapshot, 2)
    tracker.finish_frame(6)
    projection.update(overlay, observed, snapshot, 6)
    assert tracker.source == 'unavoidable_death'
    observed.p2.state = BoardState.CHAIN
    projection.update(overlay, observed, snapshot, 6.1)
    assert projection.death is None and tracker.source != 'unavoidable_death'


def test_prediction_drives_s3_and_incoming(tracker: object) -> None:
    fire(tracker, triggers=(2, 2))
    for side, score in zip(('1P', '2P'), (78150, 28960)):
        tracker.latest_chain(side).predicted_final_score = score
    tracker.finish_frame(2)
    assert tracker.source == 'S3_provisional'
    assert tracker.current.values[-1]['score_totals'] == [78150, 28960]
    assert landing.ExchangeLandingProjection()._incoming(tracker, (0, 0))[1] > 0


def test_forecast_alone_cannot_prove_death(tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, observed, snapshot = setup_projection(tracker)
    tracker.latest_chain('1P').predicted_final_score = 7000
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    tracker.finish_frame(2.1)
    projection.update(overlay, observed, snapshot, 2.1)
    assert projection.death is None
    tracker.observe_score('1P', 2.2, None, formula_total=7000)
    tracker.finish_frame(2.2)
    projection.update(overlay, observed, snapshot, 2.2)
    assert projection.death is not None


def test_new_next_queue_reassesses_response(tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, observed, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, observed, snapshot, 2)
    assert projection.death is not None
    old = overlay._history[1][-1]
    overlay._history[1].append(adapter.ConfirmedSide(2.1, old.board, np.full(4, 2)))
    monkeypatch.setattr(landing, 'future_send', lambda *args: 1000)
    projection.update(overlay, observed, snapshot, 2.1)
    assert projection.death is None


def test_refresh_features_invalidates_provisional_cache(tracker: object) -> None:
    fire(tracker)
    tracker.observe_score('1P', 2, None, formula_total=100)
    tracker.finish_frame(2)
    sides = tracker.firing.prefire_sides.copy()
    sides[1, -1] = .7
    tracker.refresh_features(static(), sides)
    tracker.finish_frame(2.1)
    assert tracker.firing.prefire_sides[1, -1] == .7
    assert tracker.current.values[-1]['t_sec'] == 2.1


@pytest.mark.parametrize("recorded", [True, False])
def test_completion_uses_firing_board_without_mutating_it(recorded: bool) -> None:
    board = Board()
    board._grid[-1, :4] = 1
    overlay = adapter.ExchangeEventOverlay(Models(), build_static, Signals)
    overlay._history[0].append(adapter.ConfirmedSide(0, board, np.ones(4, dtype=int)))
    chain = ExchangeChainRecord('1P', 1, 1, 1)
    event = SimpleNamespace(mechanism='formula_read', total_score=0, chain_count=1)
    if not recorded:
        event.before_board = board
    overlay._predict_completion(chain, event, 0)
    assert chain.predicted_final_score == 40
    assert chain.predicted_chain_count == 1
    assert np.count_nonzero(board._grid) == 4


def test_current_d_features_reach_evaluator(tracker: object) -> None:
    fire(tracker)
    updated = static()
    from src.exchange_event_evaluator import StaticInput, build_features
    d = updated.d_features.copy()
    d[0] = .8
    tracker.refresh_features(StaticInput(d, .5, 3), tracker.firing.prefire_sides)
    _, features, _ = build_features(tracker.firing, (1., 2.))
    assert features[0] == .8


def test_latest_confirmed_features_reused_until_board_change(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(adapter, 'prefire_side_features', lambda *args: np.zeros(len(SIDE_COLUMNS)))
    overlay = adapter.ExchangeEventOverlay(Models(), build_static, Signals)
    final = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                            chain_total_score_p1=0, chain_total_score_p2=0)
    overlay.update(result(0), object(), final, 0, 1)
    overlay.update(result(1), object(), final, 1, 1)
    def measured(grid: np.ndarray, queue: np.ndarray, elapsed: float) -> np.ndarray:
        calls.append(grid.copy())
        return np.full(len(SIDE_COLUMNS), np.count_nonzero(grid) / grid.size)
    monkeypatch.setattr(adapter, 'prefire_side_features', measured)
    current = result(1.1)
    current.p2.confirmed_board = Board()
    current.p2.confirmed_board._grid[-1, 0] = 1
    overlay.update(current, object(), final, 1.1, 1)
    overlay.update(current, object(), final, 1.2, 1)
    assert len(calls) == 1
    assert overlay.tracker.firing.prefire_sides[1, 0] > 0
