"""全列挙と厳密minimaxの独立照合。"""
from dataclasses import replace

import numpy as np
import pytest

from src import prefire_v5b_search as exact
from src import prefire_v5_search as legacy
from src import prefire_v5_value as value
from tests.test_prefire_v5_search import position


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_native_matches_unlimited_reference(depth: int) -> None:
    start = position()
    expected = legacy.candidates(start, depth, None)
    actual = exact.candidates(start, depth)
    assert {p.path: p for p in actual} == {p.path: p for p in expected}
    assert len(actual) == 22 ** depth


def test_k_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match='K制限'):
        exact.candidates(position(), k=8)


@pytest.mark.parametrize('queue,depth', [((1, 2), 1), ((1, 2, 0, 0, 3, 4), 1),
                                     ((1, 2, 3, 4, 0, 0), 2), ((0, 0, 1, 2), 0)])
def test_only_known_prefix(queue: tuple, depth: int) -> None:
    assert exact.known_depth(queue) == depth


@pytest.mark.parametrize('attacker', [0, 1])
@pytest.mark.parametrize('seed', range(8))
def test_pruning_matches_independent_matrix(monkeypatch: pytest.MonkeyPatch, attacker: int, seed: int) -> None:
    options = tuple(replace(position(), score=i, consumed=1, path=((i, 0),)) for i in range(22))
    monkeypatch.setattr(exact, 'candidates', lambda *args: options)
    matrix = np.random.default_rng(seed).random((22, 22))
    monkeypatch.setattr(legacy, 'resolve', lambda left, right, *args: (left, right))
    def evaluator(pair: tuple, elapsed: float) -> float:
        return float(matrix[pair[0].score, pair[1].score])
    found = exact.choose((position(), position()), attacker, 0, evaluator)
    expected = matrix.min(axis=1).max() if attacker == 0 else matrix.max(axis=0).min()
    assert found[0] == expected


@pytest.mark.parametrize('attacker', [0, 1])
def test_full_value_parity(attacker: int) -> None:
    states = (position(), replace(position(), pending=7))
    def evaluator(exchange: legacy.Exchange, elapsed: float) -> float:
        return sum((i+1) * sum(p.board) for i, p in enumerate(exchange.sides)) / 1000
    full = value.choose(states, attacker, 0, evaluator, 1, None)
    assert exact.choose(states, attacker, 0, evaluator, 1)[0] == full[0]


def test_pending_and_side_are_cache_inputs() -> None:
    exact.candidates.cache_clear()
    for side in (0, 1):
        for pending in (0, 1):
            result = exact.candidates(replace(position(), pending=pending), 1, side=side)
            assert all(p.pending == pending for p in result)
    assert exact.candidates.cache_info().misses == 4


def test_unknown_samples_preserve_known_prefix_and_observation() -> None:
    state = position((1, 2, 3, 4, 0, 0))
    states = (state, state)
    samples = exact.response_states(states, 1)
    assert len(samples) == exact.UNKNOWN_ROLLOUTS
    assert samples == exact.response_states(states, 1)
    assert all(p.queue[:4] == state.queue[:4] for p in samples)
    assert all(c in (1, 2, 3, 4) for p in samples for c in p.queue[4:])
    assert state.queue[-2:] == (0, 0)


def test_known_three_pairs_are_never_sampled() -> None:
    states = (position(), position())
    assert exact.response_states(states, 0) == (states[0],)


def test_unknown_game_colors_are_not_invented() -> None:
    states = (position((1, 2, 1, 2, 0, 0)), position((1, 2, 1, 2, 0, 0)))
    assert exact.response_states(states, 0) == (states[0],)


def test_expected_best_response_uses_every_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    attacks = tuple(replace(position(), score=i, consumed=1) for i in range(2))
    responses = tuple(replace(position(), score=i, consumed=1) for i in range(4))
    matrix = np.array([[.2, .8, .9, .7], [.5, .6, .4, .2]])
    monkeypatch.setattr(legacy, 'resolve', lambda left, right, *args: (left, right))
    def evaluator(pair: tuple, elapsed: float) -> float:
        return float(matrix[pair[0].score, pair[1].score])
    result = exact.sampled_choose(attacks, (responses[:2], responses[2:]), 0, 0, evaluator)
    assert result[0] == pytest.approx(.45)
    assert result[1].score == 0
