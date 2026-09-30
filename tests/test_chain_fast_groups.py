"""ChainSimulator(fast_groups=True) が従来経路と全出力で一致することを確かめる (出力同一の高速化)。"""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, COLOR_UNKNOWN, Board
from src.chain import ChainSimulator
from src.scoring import calculate_chain_score

SEEDS = range(4)
BOARDS_PER_SEED = 120
COLORS = (1, 2, 3, 4)
FILL_PROBABILITY = 0.55
OJAMA_PROBABILITY = 0.12
UNKNOWN_PROBABILITY = 0.02


def random_board(rng: random.Random, with_unknown: bool) -> Board:
    """連結が起きやすいよう 3〜4 色に絞った、重力済みでない乱雑な盤面 (隠し段・おじゃま含む)。"""
    board = Board()
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=board._grid.dtype)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if rng.random() > FILL_PROBABILITY:
                continue
            roll = rng.random()
            if roll < OJAMA_PROBABILITY:
                grid[row, col] = COLOR_OJAMA
            elif with_unknown and roll < OJAMA_PROBABILITY + UNKNOWN_PROBABILITY:
                grid[row, col] = COLOR_UNKNOWN
            else:
                grid[row, col] = rng.choice(COLORS)
    board._grid = grid
    return board


def group_key(groups: list) -> list:
    """順序を含めて比較できる形。"""
    return [(g.color, sorted(g.cells), g.size, sorted(g.ojama_adjacent)) for g in groups]


@pytest.mark.parametrize('ghost', (False, True))
@pytest.mark.parametrize('with_unknown', (False, True))
def test_groups_identical(ghost: bool, with_unknown: bool) -> None:
    reference = ChainSimulator(exclude_hidden_row_from_pop=ghost)
    fast = ChainSimulator(exclude_hidden_row_from_pop=ghost, fast_groups=True)
    checked = with_groups = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        for _ in range(BOARDS_PER_SEED):
            board = random_board(rng, with_unknown)
            expected = group_key(reference.find_groups(board))
            assert group_key(fast.find_groups(board)) == expected
            checked += 1
            with_groups += any(size >= 4 for _, _, size, _ in expected)
    assert checked == len(SEEDS) * BOARDS_PER_SEED
    assert with_groups > checked // 10  # 消去が起きる盤面が十分含まれる


@pytest.mark.parametrize('ghost', (False, True))
@pytest.mark.parametrize("with_unknown", (False, True))
def test_simulate_identical(ghost: bool, with_unknown: bool) -> None:
    reference = ChainSimulator(exclude_hidden_row_from_pop=ghost)
    fast = ChainSimulator(exclude_hidden_row_from_pop=ghost, fast_groups=True)
    chains = 0
    for seed in SEEDS:
        rng = random.Random(100 + seed)
        for _ in range(BOARDS_PER_SEED):
            board = random_board(rng, with_unknown=with_unknown)
            a, b = reference.simulate(board), fast.simulate(board)
            assert (a.chain_count, a.total_erased, a.total_ojama) == (b.chain_count, b.total_erased, b.total_ojama)
            assert np.array_equal(a.final_board._grid, b.final_board._grid)
            assert calculate_chain_score(a).total_score == calculate_chain_score(b).total_score
            assert [group_key(s.erased_groups) for s in a.steps] == [group_key(s.erased_groups) for s in b.steps]
            for x, y in zip(a.steps, b.steps):
                assert (x.chain_index, x.erased_ojama, x.erased_count) == (y.chain_index, y.erased_ojama, y.erased_count)
                assert np.array_equal(x.board_before._grid, y.board_before._grid)
                assert np.array_equal(x.board_after._grid, y.board_after._grid)
            chains += a.chain_count > 0
    assert chains > 20


def test_default_is_reference_path() -> None:
    assert ChainSimulator()._fast_groups is False
