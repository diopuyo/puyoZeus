"""探索結果の再利用が入力規則・可変戻り値・換算境界を保つことを検証する。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src import indicators_v2 as iv
from src.board import Board
from src.chain import ChainSimulator
from src.phase_j.live_future_cache import FutureSearchCache, cached_future_search


def arguments() -> tuple:
    board = Board()
    for col in range(3):
        board._grid[-1, col] = 1
    return board, (1, 2, 3, 4), (1, 2), (3, 4), 10., ChainSimulator(), 2, (0, 1)


def test_real_search_exact_and_mutation_isolated() -> None:
    cache = FutureSearchCache(iv._near_future_search, 'code', verify_hits=1)
    args = arguments()
    original = deepcopy(args[0]._grid)
    expected = iv._near_future_search(*args)
    first = cache(*args)
    first.values.clear()
    assert cache(*args) == expected
    hit = cache(*args)
    hit.chain_refs.clear()
    assert cache(*args) == expected
    assert (args[0]._grid == original).all()
    assert cache.summary()['verified'] == 1
    assert cache.summary()['hits'] == 3


@pytest.mark.parametrize('index,value', [(1, (4, 3, 2, 1)), (2, (2, 1)), (3, (4, 3)),
                                       (4, 97.), (6, 3), (7, (1,))])
def test_output_dependencies_are_in_key(index: int, value: Any) -> None:
    cache = FutureSearchCache(iv._near_future_search, 'code')
    args = arguments()
    changed = list(args)
    changed[index] = value
    assert cache.key(args, {}) != cache.key(tuple(changed), {})


@pytest.mark.parametrize('setting', ['tiebreak', 'resolve_before_death', 'use_exact_score', 'fast_expand'])
def test_search_switches_are_in_key(setting: str) -> None:
    cache = FutureSearchCache(iv._near_future_search, 'code')
    args = arguments()
    assert cache.key(args, {}) != cache.key(args, {setting: True})


def test_rate_equivalence_and_simulator_rules() -> None:
    cache = FutureSearchCache(iv._near_future_search, 'code')
    args = arguments()
    changed = list(args)
    changed[4] = 95.
    assert cache.key(args, {}) == cache.key(tuple(changed), {})
    changed[5] = ChainSimulator(exclude_hidden_row_from_pop=True)
    assert cache.key(args, {}) != cache.key(tuple(changed), {})
    changed[5] = ChainSimulator(fast_groups=True)
    assert cache.key(args, {}) != cache.key(tuple(changed), {})
    assert cache.key(args, {}) != FutureSearchCache(iv._near_future_search, 'new').key(args, {})


def test_capacity_and_unknown_simulator() -> None:
    cache = FutureSearchCache(iv._near_future_search, 'code', capacity=1, verify_hits=0)
    args = arguments()
    cache(*args)
    changed = list(args)
    changed[2] = (2, 1)
    cache(*changed)
    cache(*args)
    assert cache.misses == 3 and len(cache.entries) == 1
    changed[5] = object()
    assert cache.key(tuple(changed), {}) is None


def test_mismatch_disables_reuse_and_returns_fresh_result() -> None:
    cache = FutureSearchCache(iv._near_future_search, 'code')
    args = arguments()
    expected = cache(*args)
    cache.entries[cache.key(args, {})].values.clear()
    assert cache(*args) == expected
    assert cache.disabled and cache.mismatches == 1 and not cache.entries
    assert cache(*args) == expected


def test_default_off_does_not_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_LIVE_NEAR_FUTURE_CACHE', raising=False)
    original = iv._near_future_search
    with cached_future_search() as cache:
        assert cache is None and iv._near_future_search is original


def test_context_installs_and_restores_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_future_cache
    monkeypatch.setenv('PUYO_LIVE_NEAR_FUTURE_CACHE', '1')
    monkeypatch.setattr(live_future_cache, 'source_sha', lambda: 'code')
    original = iv._near_future_search
    with pytest.raises(RuntimeError):
        with cached_future_search() as cache:
            assert iv._near_future_search is cache
            iv._near_future_search(*arguments())
            iv._near_future_search(*arguments())
            assert cache is not None and cache.verified == 1
            raise RuntimeError('終了経路も復元する')
    assert iv._near_future_search is original


def test_recognition_scope_never_installs_search_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_cache import bounded_chain_caches
    monkeypatch.setenv('PUYO_LIVE_NEAR_FUTURE_CACHE', '1')
    original = iv._near_future_search
    with bounded_chain_caches():
        assert iv._near_future_search is original


def test_evaluation_scope_records_nested_search_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_future_cache, live_stage_timing
    from src.phase_j.live_cache import bounded_chain_caches
    monkeypatch.setenv('PUYO_LIVE_NEAR_FUTURE_CACHE', '1')
    monkeypatch.setattr(live_future_cache, 'source_sha', lambda: 'code')
    record = dict(stages={})
    token = live_stage_timing.CURRENT.set(record)
    try:
        with bounded_chain_caches(future_search=True):
            assert isinstance(iv._near_future_search, FutureSearchCache)
            iv._near_future_search(*arguments())
            assert record['stages']['event_near_future_search']['calls'] == 1
    finally:
        live_stage_timing.CURRENT.reset(token)
