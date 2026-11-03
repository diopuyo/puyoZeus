"""完走後盤面、純受け量、応手と予測の確実性を独立に検証する。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src import exchange_event_landing as landing
from src import exchange_event_overlay as adapter
from src.board import Board, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.exchange_event_tracker import ExchangeChainRecord
from tests.test_e10_exchange_landing import board_at_height
from tests.test_e10b_exchange_landing import setup_projection
from tests.test_exchange_event_tracker import fire, tracker
from tests.test_exchange_event_overlay import Models, Signals, build_static


def active_projection(tracker: object, height: int = 11) -> tuple:
    """相殺後20個の純受けと、検証済みの完走後盤面を用意する。"""
    projection, overlay, observed, snapshot = setup_projection(tracker)
    fire(tracker, 2.1, (None, 2.1))
    chain = tracker.latest_chain('2P')
    chain.predicted_final_score = 70
    chain.predicted_chain_count = 1
    chain.predicted_final_board = board_at_height(height)._grid.tolist()
    tracker.observe_score('1P', 2.1, None, formula_total=1470)
    tracker.finish_frame(2.1)
    observed.p2.state = BoardState.CHAIN
    return projection, overlay, observed, snapshot, chain


def test_active_receiver_dies_on_completion_only(tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, observed, snapshot, _ = active_projection(tracker)
    overlay._history[1][-1].board._grid[:] = 0
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, observed, snapshot, 2.1)
    assert projection.last['incoming'] == [0, 20]
    assert projection.last['overflow_rows'][1] == 2
    assert projection.last['resolving_send'] == [0, 0]
    assert tracker.probability == .98


def test_frozen_high_board_cannot_kill_safe_completion(tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, observed, snapshot, _ = active_projection(tracker, height=4)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, observed, snapshot, 2.1)
    assert projection.death is None
    assert projection.last['dead_sides'] == []


@pytest.mark.parametrize('available', [0, 1000])
def test_response_search_uses_completion(tracker: object, monkeypatch: pytest.MonkeyPatch,
                                          available: int) -> None:
    projection, overlay, observed, snapshot, chain = active_projection(tracker)
    calls = []
    def response(raw: bytes, shape: tuple, dtype: str, *args: object) -> int:
        calls.append(np.frombuffer(raw, dtype=dtype).reshape(shape))
        return available
    monkeypatch.setattr(landing, 'future_send', response)
    projection.update(overlay, observed, snapshot, 2.1)
    assert calls and all(np.array_equal(b, chain.predicted_final_board) for b in calls)
    assert (projection.death is not None) == (available == 0)


@pytest.mark.parametrize('uncertainty', ['missing', 'score', 'count'])
def test_uncertain_completion_releases_held_death(tracker: object, monkeypatch: pytest.MonkeyPatch,
                                                 uncertainty: str) -> None:
    projection, overlay, observed, snapshot, chain = active_projection(tracker)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, observed, snapshot, 2.1)
    assert projection.death is not None
    if uncertainty == 'missing':
        chain.predicted_final_board = None
    elif uncertainty == 'score':
        chain.formula_total = 140
    else:
        projection.counts[1] = 2
    projection.update(overlay, observed, snapshot, 2.2)
    assert projection.death is None
    assert tracker.source != 'unavoidable_death'


def test_same_active_chain_holds_until_landing(tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, observed, snapshot, _ = active_projection(tracker)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, observed, snapshot, 2.1)
    held = projection.death
    snapshot.total_dropped_to_p2 = 20
    projection.update(overlay, observed, snapshot, 2.2)
    assert projection.death is held and tracker.probability == .98


def test_prediction_cancel_is_not_credited_twice(tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, observed, snapshot, chain = active_projection(tracker)
    chain.predicted_final_score = 1400
    tracker.finish_frame(2.1)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, observed, snapshot, 2.1)
    assert projection.last['incoming'] == [0, 1]
    assert projection.last['resolving_send'] == [0, 0]
    assert projection.death is None


@pytest.mark.parametrize('failure', ['unknown', 'zero_chain', 'exception', 'mismatch'])
def test_prediction_uncertainty_does_not_produce_final_board(failure: str,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    board = Board()
    board._grid[-1, :4] = 1
    overlay = adapter.ExchangeEventOverlay(Models(), build_static, Signals)
    chain = ExchangeChainRecord('1P', 1, 1, 1)
    if failure == 'unknown':
        board._grid[-1, -1] = COLOR_UNKNOWN
    elif failure == 'zero_chain':
        board._grid[:] = 0
    elif failure == 'exception':
        def fail(*args: object) -> None:
            raise ValueError('予測不能')
        monkeypatch.setattr(overlay._landing_projection.simulator, 'simulate', fail)
    overlay._history[0].append(adapter.ConfirmedSide(0, board, np.ones(4, dtype=int)))
    event = SimpleNamespace(mechanism='baseline', total_score=1400, chain_count=2)
    if failure != 'mismatch':
        event.before_board = board
    overlay._predict_completion(chain, event, 0)
    assert chain.predicted_final_board is None


def test_predict_completion_saves_board_without_mutation() -> None:
    board = Board()
    board._grid[-1, :4] = 1
    overlay = adapter.ExchangeEventOverlay(Models(), build_static, Signals)
    chain = ExchangeChainRecord('1P', 1, 1, 1)
    overlay._predict_completion(chain, SimpleNamespace(before_board=board), 0)
    assert np.count_nonzero(chain.predicted_final_board) == 0
    assert np.count_nonzero(board._grid) == 4


def test_drop_bonus_does_not_invalidate_completion(tracker: object) -> None:
    projection, _, _, _, chain = active_projection(tracker)
    chain.score_delta, chain.drop_bonus_score = 72, 2
    assert projection._completion_board(tracker, 1) is not None


def test_raw_board_death_is_rechecked_when_same_chain_becomes_busy(
        tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """静止時の固定を、連鎖再検知時に安全な完走後盤面へ持ち越さない。"""
    projection, overlay, observed, snapshot, chain = active_projection(tracker, height=4)
    chain.end_signal_sec = 2.1
    observed.p2.state = BoardState.STABLE
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, observed, snapshot, 2.1)
    assert projection.death is not None
    observed.p2.state = BoardState.CHAIN
    projection.update(overlay, observed, snapshot, 2.2)
    assert projection.death is None
    assert tracker.source != 'unavoidable_death'
