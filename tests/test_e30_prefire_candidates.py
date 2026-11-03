"""全候補の列挙・集約・観測撤回の物理的契約を検証する。"""
from types import SimpleNamespace
import numpy as np
import pytest
from src.board import Board
from src.chain import ChainSimulator
from src.exchange_prefire_candidates import (
    PrefireCandidates, completion, enumerate_candidates, pairs_for, statistics,
)
from src.exchange_event_tracker import ExchangeChainRecord

COLORS = (1, 2, 3, 4)


def board() -> Board:
    """3個の赤から1手・2手の両方で発火できる起点。"""
    value = Board()
    value._grid[-1, :3] = 1
    return value


def option(score: int = 40, weight: int = 1, color: int = 0) -> dict:
    """集約規則を検証する小さな候補。"""
    value = Board()
    value._grid[-1, -1] = color
    return dict(score=score, weight=weight, board=value._grid.tolist(), prefix=(40, score), colors=((1,), (2,)))


def test_pairs_cover_ten_and_next() -> None:
    assert len(pairs_for(COLORS, (1, 2, 2, 1))) == 10
    assert (1, 5) in pairs_for(COLORS, (5, 1))
    assert len(pairs_for(COLORS, (0, 0))) == 10


def test_enumeration_contains_both_depths_and_no_mutation() -> None:
    origin = board()
    before = origin._grid.copy()
    options, trials = enumerate_candidates(origin, COLORS, (), ChainSimulator(cache_enabled=False))
    assert trials > 220
    assert options and any(v['board'][-1][-1] != 0 for v in options)
    assert all(v['prefix'] and v['weight'] > 0 for v in options)
    assert np.array_equal(origin._grid, before)


@pytest.mark.parametrize('colors,unknown', [(COLORS[:3], False), (COLORS, True)])
def test_missing_evidence_rejected(colors: tuple, unknown: bool) -> None:
    origin = board()
    if unknown:
        origin._grid[0, 0] = 10
    assert enumerate_candidates(origin, colors, (), ChainSimulator()) == ([], 0)


def test_weighted_mean_and_tied_board() -> None:
    value = statistics([option(70, 2), option(140, 1, 2)], 0)
    assert value['mean_score'] == pytest.approx(280/3)
    assert value['mean_send'] == pytest.approx(4/3)
    assert value['board'] == option()['board']
    assert statistics([option(), option(color=2)], 0)['board'] is None
    assert statistics([], 0)['mean_score'] is None


def test_prefix_color_filter_and_retraction() -> None:
    engine = PrefireCandidates(ChainSimulator())
    chain = ExchangeChainRecord('2P', 1, 1., 1.)
    original = (0., 0, None)
    chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board = original
    entry = dict(chain=chain, options=[option(320), option(500, color=2)], elapsed=0.,
        original=original, published=None, last=None, audit=dict(observations=[]))
    engine.entries[1] = entry
    engine._publish(entry)
    assert chain.predicted_final_score == 410
    assert chain.predicted_final_board is None
    overlay = SimpleNamespace(tracker=SimpleNamespace(latest_chain=lambda side: chain if side == '2P' else None))
    event = SimpleNamespace(mechanism='formula_read', trigger_sec=1., chain_count=2, total_score=320)
    result = SimpleNamespace(p1=SimpleNamespace(chain_event=None), p2=SimpleNamespace(chain_event=event))
    engine.observe(overlay, result, 2.)
    assert chain.predicted_final_score == 320
    assert engine.maximum(chain)['score'] == 320
    engine.observe(overlay, result, 2.1)
    assert len(entry['audit']['observations']) == 1
    event.erased_colors = (3,)
    engine.observe(overlay, result, 2.2)
    assert not entry['options']
    assert (chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board) == original
    assert engine.maximum(chain) is None


def test_completed_chain_and_boundary_disable_candidates() -> None:
    engine = PrefireCandidates(ChainSimulator())
    chain = ExchangeChainRecord('1P', 1, 0., 0.)
    engine.entries[1] = dict(chain=chain, options=[option()])
    assert engine.active(chain)
    chain.end_signal_sec, chain.score_ready_sec = 1., 1.
    assert engine.active(chain) is None
    engine.reset()
    assert not engine.entries
