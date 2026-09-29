"""隠し段を全列挙しても確率層へ流さず、実測次段で候補を検証する。"""
from __future__ import annotations

from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board import Board
from src.board_state_machine import BoardState as S
from src.chain import ChainSimulator
from src.exchange_hidden_row_death import HiddenRowDeathCompletion, enumerate_hidden
from tests.test_e26_midchain import setup, sample, formula


def hidden_board(count: int = 1) -> Board:
    """右側のおじゃま支柱上を隠し段不明とし、左下に消去群を置く。"""
    board = Board()
    board._grid[-4:, 0] = 1
    for col in range(6-count, 6):
        board._grid[1:, col] = 9
        board._grid[0, col] = 10
    return board


def hidden_setup() -> tuple:
    """E26と同じ時系列入力で別の死亡専用器を使う。"""
    _, overlay, result, chain, _ = setup()
    engine = HiddenRowDeathCompletion(ChainSimulator())
    engine.color_evidence[0].counts.update({1: 10, 2: 10, 3: 10, 4: 10})
    result.p2.confirmed_board = None
    engine.observe(overlay, result, 1.)
    return engine, overlay, result, chain, hidden_board()


@pytest.mark.parametrize('count', [1, 2, 3])
def test_all_five_power_n_assignments(count: int) -> None:
    board = hidden_board(count)
    saved = board._grid.copy()
    values, trials = enumerate_hidden(board, (1, 2, 3, 4), 1, 40., ChainSimulator())
    assert trials == 5**count and values
    np.testing.assert_array_equal(board._grid, saved)


@pytest.mark.parametrize('colors', [(), (1, 2, 3), (1, 2, 3, 4, 5), (1, 1, 2, 3), (1, 2, 3, 9)])
def test_unconfirmed_four_colors_rejected(colors: tuple) -> None:
    assert enumerate_hidden(hidden_board(), colors, 1, 40., ChainSimulator()) == ([], 0)


@pytest.mark.parametrize('count', [0, 4, 5, 6])
def test_unknown_count_out_of_scope(count: int) -> None:
    assert enumerate_hidden(hidden_board(count), (1, 2, 3, 4), 1, 40., ChainSimulator()) == ([], 0)


def test_visible_unknown_rejected() -> None:
    board = hidden_board()
    board._grid[1, 5] = 10
    assert enumerate_hidden(board, (1, 2, 3, 4), 1, 40., ChainSimulator()) == ([], 0)


def test_matching_formula_never_mutates_prediction() -> None:
    engine, overlay, result, chain, board = hidden_setup()
    before = vars(chain).copy()
    sample(engine, overlay, result, board)
    assert engine.maximum(chain) is None
    formula(engine, overlay, result)
    assert engine.maximum(chain)['score'] == 360
    assert vars(chain) == before
    assert engine.summary()['accepted'] == 1
    assert engine.summary()['used'] == 0


@pytest.mark.parametrize('score', [0., 320., 361., 999.])
def test_no_matching_next_score_discards(score: float) -> None:
    engine, overlay, result, chain, board = hidden_setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result, score=score)
    assert engine.maximum(chain) is None and engine.summary()['next_mismatch'] == 1


def test_later_mismatch_revokes() -> None:
    engine, overlay, result, chain, board = hidden_setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    formula(engine, overlay, result, count=3, score=999.)
    assert engine.maximum(chain) is None


def test_single_sample_is_not_used() -> None:
    engine, overlay, result, chain, board = hidden_setup()
    result.p1.state, result.p1.midchain_board = S.GRAVITY_SETTLE, board
    engine.observe(overlay, result, 1.1)
    formula(engine, overlay, result)
    assert engine.summary()['enumerated'] == 0


def test_usage_is_per_candidate_and_reset_clears_active() -> None:
    engine, overlay, result, chain, board = hidden_setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    value = engine.maximum(chain)
    engine.mark_used(value, 1.3)
    engine.mark_used(value, 1.4)
    assert engine.summary()['used'] == 1 and engine.summary()['uses'] == 2
    engine.reset()
    assert engine.palettes == ((), ()) and engine.maximum(chain) is None


def test_maximum_retains_all_tied_final_boards() -> None:
    engine, overlay, result, chain, board = hidden_setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    assert len(engine.maximum(chain)['options']) == 5


def test_finished_chain_excluded() -> None:
    engine, overlay, result, chain, board = hidden_setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    chain.end_signal_sec, chain.score_ready_sec, chain.end_confirmed = 2., 2., True
    assert engine.maximum(chain) is None


def test_larger_counterattack_wins_even_when_other_candidates_match() -> None:
    engine, overlay, result, chain, _ = hidden_setup()
    engine.simulator = ChainSimulator(exclude_hidden_row_from_pop=True)
    board = Board()
    board._grid[0, 5] = 10
    board._grid[1:4, 5] = 2
    board._grid[4:, 5] = 1
    values, trials = enumerate_hidden(board, (1, 2, 3, 4), 1, 40., engine.simulator)
    assert trials == 5 and len({v['prefix'][2] for v in values}) == 1
    assert len({v['score'] for v in values}) > 1
    sample(engine, overlay, result, board)
    formula(engine, overlay, result, score=values[0]['prefix'][2])
    maximum = engine.maximum(chain)
    assert [v['hidden'] for v in maximum['options']] == [[2]]
    assert chain.predicted_final_board is None


def test_palette_change_invalidates_previous_bound() -> None:
    engine, overlay, result, chain, board = hidden_setup()
    sample(engine, overlay, result, board)
    formula(engine, overlay, result)
    assert engine.maximum(chain) is not None
    engine.color_evidence[0].counts.update({5: 100})
    engine.observe(overlay, result, 1.4)
    assert engine.maximum(chain) is None
