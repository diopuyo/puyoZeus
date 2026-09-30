"""E35の上限・着弾順序・生存枝・不確実入力を検証する。"""
from collections import Counter

import numpy as np
import pytest

from src.board import Board, DEATH_COL, DEATH_ROW
from src.chain import ChainSimulator
from src.post_counter_death_bound import Bound, prove_post_counter, score_upper, usable_colors
from src.scoring import calculate_chain_score

PALETTE = (1, 2, 3, 4)


def garbage(height: int) -> Board:
    """全列同高のおじゃま床を作る。"""
    board = Board()
    if height:
        board._grid[-height:] = 9
    return board


@pytest.mark.parametrize('height', range(8, 12))
def test_no_clear_even_favorable_remainder(height: int) -> None:
    board = garbage(height)
    value = prove_post_counter(board, (1, 2, 3, 4), 171, 1, 0, PALETTE)
    assert value['dead'] and value['pruned'] == 1
    assert board._grid[DEATH_ROW, DEATH_COL] == 0


@pytest.mark.parametrize('height', range(1, 8))
def test_repeated_five_rows_without_clear(height: int) -> None:
    assert prove_post_counter(garbage(height), (1, 2, 3, 4), 171, 1, 0, PALETTE)['dead']


@pytest.mark.parametrize('amount', (1, 2, 5, 6, 11, 29, 30))
def test_surviving_branch_refuses_death(amount: int) -> None:
    assert not prove_post_counter(garbage(1), (1, 2, 3, 4), amount, 1, 0, PALETTE)['dead']


def test_chain_hand_postpones_landing(monkeypatch: pytest.MonkeyPatch) -> None:
    board = garbage(9)
    board._grid[3, :3] = 1
    proof = Bound((1, 2), PALETTE, 70, 100)
    remaining = []
    def visit(board: Board, pending: int, turn: int, grace: int) -> bool:
        remaining.append(pending)
        return False
    monkeypatch.setattr(proof, 'visit', visit)
    assert not proof.expand(board, 171, 0, 0, [(1, 2)])
    # 楽観火力31個だけを相殺し、連鎖手に着弾30個をさらに引かない。
    assert remaining[0] == 140


def test_buried_colors_have_no_immediate_firepower() -> None:
    board = garbage(8)
    board._grid[-1] = [1, 1, 2, 2, 3, 3]
    assert usable_colors(board, Counter({1: 2, 2: 2, 3: 2, 4: 2})) == 0


def test_exposed_trigger_keeps_downstream_buried_colors() -> None:
    board = garbage(8)
    board._grid[4, :2] = 1
    board._grid[-1] = [2, 2, 3, 3, 4, 4]
    assert usable_colors(board, Counter({1: 2})) == 8


def test_new_pair_can_bridge_two_groups() -> None:
    board = garbage(1)
    board._grid[11, 0] = board._grid[11, 2] = 1
    assert usable_colors(board, Counter({1: 2})) == 2


@pytest.mark.parametrize('count', (4, 5, 6, 8, 11, 12, 24, 40))
def test_score_upper_includes_large_simultaneous_groups(count: int) -> None:
    board = Board()
    board._grid.flat[-count:] = 1
    actual = calculate_chain_score(ChainSimulator().simulate(board)).total_score
    assert score_upper(count) >= actual


@pytest.mark.parametrize('kind', ('unknown', 'floating', 'dead', 'palette', 'hands', 'amount'))
def test_uncertain_inputs_fail_open(kind: str) -> None:
    board = garbage(8)
    palette, hands, amount = PALETTE, 1, 171
    if kind == 'unknown':
        board._grid[-1, 0] = 10
    elif kind == 'floating':
        board._grid[2, 0] = 1
    elif kind == 'dead':
        board._grid[DEATH_ROW:, DEATH_COL] = 9
    elif kind == 'palette':
        palette = (1, 2, 3)
    elif kind == 'hands':
        hands = 0
    else:
        amount = 0
    value = prove_post_counter(board, (), amount, hands, 0, palette)
    assert not value['dead'] and value['reason'] == 'invalid_bound_input'


def test_budget_exhaustion_is_not_death() -> None:
    value = prove_post_counter(garbage(1), (), 171, 1, 0, PALETTE, node_limit=0)
    assert not value['dead'] and value['reason'] == 'bound_limit'


def test_no_mutation() -> None:
    board = garbage(6)
    before = board._grid.copy()
    prove_post_counter(board, (1, 2, 3, 4), 171, 1, 0, PALETTE)
    np.testing.assert_array_equal(board._grid, before)


def test_d4_post_counter_board_is_proved_without_twenty_thousand_nodes() -> None:
    """D4保存候補の一例。場面値は回帰用入力だけで、判定器の定数に使わない。"""
    board = Board.from_list([[0]*6]*6 + [
        [0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 3, 4],
        [5, 0, 0, 4, 4, 5]])
    result = prove_post_counter(board, (4, 5, 1, 5), 171, 1, 0, (1, 3, 4, 5))
    assert result['dead'] and result['pruned'] > 0 and result['nodes'] < 20000
