"""5Cの特徴列と通知間キャッシュの意味論を検証する。"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src import prefire_v5c_value as fast
from src import prefire_v5_search as search
from src.board import Board

SEEDS = range(12)


@pytest.mark.parametrize('seed', SEEDS)
def test_native_features_equal_original(seed: int) -> None:
    from scripts.visualize_advantage_overlay import _side_feats_full_base
    rng = np.random.default_rng(seed)
    board = Board()
    for col in range(6):
        height = int(rng.integers(1, 7))
        board._grid[-height:, col] = rng.choice([1, 2, 3, 4, 9], height)
    if seed == 0:
        board._grid[0, 0] = 1
    expected = _side_feats_full_base(board)
    actual = fast.board_features(board._grid.tobytes(), board._grid.shape, board._grid.dtype.str)
    assert actual.keys() == expected.keys()
    assert np.array_equal(list(actual.values()), list(expected.values()), equal_nan=True)


def overlay() -> SimpleNamespace:
    from scripts.visualize_advantage_overlay import _exchange_static_input
    return SimpleNamespace(_m0=object(), _build_static=_exchange_static_input,
        _snapshots=[(0, SimpleNamespace())], tracker=SimpleNamespace(
            models=SimpleNamespace(elapsed_thresholds=(10., 20.))))


def exchange() -> search.Exchange:
    position = search.Position(bytes(78), (1, 2, 3, 4))
    return search.Exchange((position, position), (0, 0), (0, 0), (False, False))


def test_value_reuses_phase_but_not_pending_queue_or_board(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    def evaluate(proxy: object, item: search.Exchange, time: float) -> float:
        calls.append((item, time))
        return .4
    monkeypatch.setattr(fast.value, 'evaluate', evaluate)
    cache, item = fast.EvaluationCache(overlay()), exchange()
    assert cache(item, 1.) == cache(item, 2.)
    assert len(calls) == 1
    cache(item, 10.)
    assert len(calls) == 1
    cache(item, 10.01)
    for changes in ({'pending': 1}, {'queue': (2, 1)}, {'board': bytes([1])+bytes(77)}):
        cache(replace(item, sides=(replace(item.sides[0], **changes), item.sides[1])), 1.)
    assert len(calls) == 5


def test_models_and_builders_are_instance_scoped() -> None:
    original = overlay()
    cache = fast.EvaluationCache(original)
    assert cache.matches(original)
    assert not cache.matches(overlay())
    original.tracker.models = object()
    assert not cache.matches(original)


def test_custom_static_has_no_phase_assumption(monkeypatch: pytest.MonkeyPatch) -> None:
    original = overlay()
    original._build_static = lambda *args: None
    monkeypatch.setattr(fast.value, 'evaluate', lambda proxy, item, time: time)
    cache = fast.EvaluationCache(original)
    assert cache(exchange(), 1.) == 1.
    assert cache(exchange(), 2.) == 2.
    assert not cache.values


def test_notification_reuses_results_but_keeps_ready_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.prefire_best_play_v5 import BestPlayV5Layer
    from src import prefire_v5b_search as exact
    calls = []
    def choose(states: tuple, side: int, elapsed: float, evaluator: object, **kwargs: object) -> tuple:
        calls.append((side, elapsed))
        return (.4, states[side], states[1-side])
    monkeypatch.setattr(exact, 'choose', choose)
    source, layer = overlay(), BestPlayV5Layer(.3)
    states = exchange().sides
    for key, elapsed in ((1, 1.), (2, 2.), (3, 10.01)):
        layer._schedule_v5(source, (key,), states, (search.OK, search.OK), elapsed, elapsed)
        assert layer._ready[(key,)] == elapsed+.3
    assert len(calls) == 4
    assert layer._cache[(1,)] == layer._cache[(2,)]
    source.tracker.models = SimpleNamespace(elapsed_thresholds=(10., 20.))
    layer._schedule_v5(source, (4,), states, (search.OK, search.OK), 1., 1.)
    assert len(calls) == 6
