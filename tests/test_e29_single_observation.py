"""一枚候補の次段検証、UNKNOWN拒否、予測と死亡への公開境界を検査する。"""
from __future__ import annotations
from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board_state_machine import BoardState as S
from src.chain import ChainSimulator
from src.exchange_midchain_completion import MidchainCompletion
from src.exchange_hidden_row_death import HiddenRowDeathCompletion
from tests.test_e26_midchain import setup, formula
from tests.test_e27_hidden_death import hidden_setup


def prepare(hidden: bool, enabled: bool = True) -> tuple:
    """既存の実盤面を使い、コンストラクタのフラグ経由で実験器を作る。"""
    _, overlay, result, chain, board = hidden_setup() if hidden else setup()
    cls = HiddenRowDeathCompletion if hidden else MidchainCompletion
    engine = cls(ChainSimulator(), single_observation=enabled)
    if hidden:
        engine.color_evidence[0].counts.update({1:10, 2:10, 3:10, 4:10})
    engine.observe(overlay, result, 1.)
    result.p1.state, result.p1.midchain_board = S.GRAVITY_SETTLE, board
    return engine, overlay, result, chain, board


def adopted(engine: object, chain: object, hidden: bool) -> bool:
    """通常予測と死亡専用の公開口から採用状態を調べる。"""
    return engine.maximum(chain) is not None if hidden else engine.verified(chain)


@pytest.mark.parametrize('hidden', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_one_sample_requires_independent_next_formula(hidden: bool, enabled: bool) -> None:
    engine, overlay, result, chain, board = prepare(hidden, enabled)
    saved, prediction = board._grid.copy(), vars(chain).copy()
    engine.observe(overlay, result, 1.1)
    assert not adopted(engine, chain, hidden) and vars(chain) == prediction
    formula(engine, overlay, result)
    assert adopted(engine, chain, hidden) is enabled
    np.testing.assert_array_equal(board._grid, saved)
    if hidden:
        assert vars(chain) == prediction


@pytest.mark.parametrize('hidden', [False, True])
@pytest.mark.parametrize('count,score', [(2,320.), (2,361.), (3,360.), (2,400.)])
def test_next_mismatch_discards_single_candidate(hidden: bool, count: int, score: float) -> None:
    engine, overlay, result, chain, _ = prepare(hidden)
    engine.observe(overlay, result, 1.1)
    formula(engine, overlay, result, count=count, score=score)
    assert not adopted(engine, chain, hidden)
    assert engine.summary()['next_mismatch'] == 1
    assert chain.predicted_final_board is None


@pytest.mark.parametrize('hidden', [False, True])
@pytest.mark.parametrize('row,col', [(1,0), (6,3), (12,4)])
def test_visible_unknown_still_rejected(hidden: bool, row: int, col: int) -> None:
    engine, overlay, result, chain, board = prepare(hidden)
    board._grid[row, col] = 10
    engine.observe(overlay, result, 1.1)
    formula(engine, overlay, result)
    assert not engine.audit and not adopted(engine, chain, hidden)


@pytest.mark.parametrize('hidden', [False, True])
@pytest.mark.parametrize('mechanism', ['formula', 'physics', 'baseline'])
def test_other_mechanisms_cannot_verify(hidden: bool, mechanism: str) -> None:
    engine, overlay, result, chain, _ = prepare(hidden)
    engine.observe(overlay, result, 1.1)
    result.p1.state = S.CHAIN
    result.p1.chain_event = NS(trigger_sec=1., mechanism=mechanism, chain_count=2, total_score=360.)
    engine.observe(overlay, result, 1.3)
    assert not adopted(engine, chain, hidden)


@pytest.mark.parametrize('hidden', [False, True])
@pytest.mark.parametrize('defect', ['missing', 'floating'])
def test_missing_or_floating_is_not_one_valid_observation(hidden: bool, defect: str) -> None:
    engine, overlay, result, chain, board = prepare(hidden)
    if defect == 'missing':
        result.p1.midchain_board = None
    else:
        board._grid[2, 2] = 3
    engine.observe(overlay, result, 1.1)
    formula(engine, overlay, result)
    assert not engine.audit and not adopted(engine, chain, hidden)


@pytest.mark.parametrize('hidden', [False, True])
def test_same_formula_is_not_next_formula(hidden: bool) -> None:
    engine, overlay, result, chain, _ = prepare(hidden)
    engine.observe(overlay, result, 1.1)
    formula(engine, overlay, result, count=1, score=40.)
    assert not adopted(engine, chain, hidden)


@pytest.mark.parametrize('hidden', [False, True])
def test_pending_does_not_cross_boundary(hidden: bool) -> None:
    engine, overlay, result, chain, _ = prepare(hidden)
    engine.observe(overlay, result, 1.1)
    engine.reset()
    formula(engine, overlay, result)
    assert not adopted(engine, chain, hidden)


@pytest.mark.parametrize('hidden', [False, True])
def test_later_formula_mismatch_revokes(hidden: bool) -> None:
    engine, overlay, result, chain, _ = prepare(hidden)
    engine.observe(overlay, result, 1.1)
    formula(engine, overlay, result)
    assert adopted(engine, chain, hidden)
    formula(engine, overlay, result, count=3, score=1000.)
    assert not adopted(engine, chain, hidden) and chain.predicted_final_board is None


@pytest.mark.parametrize('count', [1,2,3])
def test_single_hidden_sample_keeps_every_assignment(count: int) -> None:
    from tests.test_e27_hidden_death import hidden_board
    engine, overlay, result, chain, _ = prepare(True)
    result.p1.midchain_board = hidden_board(count)
    engine.observe(overlay, result, 1.1)
    assert engine.summary()['combinations'] == 5**count
    assert engine.maximum(chain) is None
    formula(engine, overlay, result)
    assert engine.maximum(chain)['score'] == 360
    assert chain.predicted_final_board is None


def test_single_flag_alone_does_not_enable_prediction_engines() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    overlay = object.__new__(ExchangeEventOverlay)
    overlay._landing_projection = NS(simulator=ChainSimulator())
    overlay._initialize_prediction_guards(False, False, False, False, True)
    assert overlay._midchain is None and overlay._hidden_death is None
