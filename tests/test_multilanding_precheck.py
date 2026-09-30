"""multilanding 証明の打切り予告 (出力同一の高速化) が従来経路と完全に一致することを確かめる。"""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, Board
from src.chain import ChainSimulator
from src.exchange_event_multilanding import EXACT_PRECHECK, prove_multilanding
from src.exchange_multilanding_precheck import (column_heights, exceeds_limit,
                                                ojama_column_combinations, placement_count)
from src.indicators_v2 import _enumerate_placements

SEEDS = range(6)
BOARDS_PER_SEED = 40
FILL_CHOICES = (0, 3, 8, 10, 11, 12, 13)
PIECE_COLORS = (1, 2, 3, 4)
OJAMA = 9
SMALL_LIMITS = (60, 400, 2500)
PLACEMENT_PAIR = (1, 2)


def random_board(rng: random.Random, floating: bool = False) -> Board:
    """高さがばらつく盤面。floating=True は柱の途中に空きがある (高さは最上段で決まる)。"""
    board = Board()
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=board._grid.dtype)
    for col in range(BOARD_COLS):
        height = rng.choice(FILL_CHOICES + (rng.randint(0, BOARD_ROWS),))
        for k in range(height):
            grid[BOARD_ROWS - 1 - k, col] = rng.choice(PIECE_COLORS + (OJAMA,))
        if floating and height > 3:
            grid[BOARD_ROWS - 2, col] = 0
    board._grid = grid
    return board


def stub_optimistic(board: Board, queue: np.ndarray, hands: int, elapsed: float) -> float:
    """相殺不能側 (0) の楽観値。証明を最後まで走らせるための固定スタブ。"""
    return 0.0


@pytest.mark.parametrize('floating', (False, True))
def test_placement_count_matches_enumeration(floating: bool) -> None:
    sim = ChainSimulator()
    for seed in SEEDS:
        rng = random.Random(seed)
        for _ in range(BOARDS_PER_SEED):
            board = random_board(rng, floating)
            assert placement_count(board._grid) == len(_enumerate_placements(board, PLACEMENT_PAIR, sim))


def test_column_heights_matches_board_height_of() -> None:
    rng = random.Random(7)
    for _ in range(BOARDS_PER_SEED):
        board = random_board(rng, floating=True)
        assert list(column_heights(board._grid)) == [board.height_of(c) for c in range(BOARD_COLS)]


def test_helpers() -> None:
    assert ojama_column_combinations(30) == 1 and ojama_column_combinations(31) == BOARD_COLS
    assert exceeds_limit(10, 5, 14) and not exceeds_limit(10, 4, 14)


def _prove(board: Board, incoming: int, hands: int, limit: int, precheck: bool) -> dict:
    queue = (1, 2, 3, 4)
    return prove_multilanding(board, queue, incoming, hands, 0.0, stub_optimistic,
                              credit=0, node_limit=limit, precheck=precheck)


def test_precheck_output_identical_to_reference() -> None:
    """打切り・全滅・生存のどの結論でも、結果 dict (rounds / nodes を含む) が完全一致する。"""
    rng = random.Random(11)
    reasons: set[str] = set()
    cases = 0
    for _ in range(BOARDS_PER_SEED):
        board = random_board(rng)
        if board.is_dead():
            continue
        for incoming, hands, limit in ((40, 1, 400), (120, 2, 400), (200, 3, 2500), (60, 2, 60)):
            reference = _prove(board, incoming, hands, limit, precheck=False)
            fast = _prove(board, incoming, hands, limit, precheck=True)
            assert fast == reference
            reasons.add(reference['reason'])
            cases += 1
    assert cases > 40
    assert 'node_limit' in reasons  # 打切り経路が実際に検証されている (母数ゼロを合格と誤読しない)


def test_default_flag_is_on_and_off_path_still_works() -> None:
    assert EXACT_PRECHECK is True
    board = random_board(random.Random(3))
    assert _prove(board, 30, 1, 60, precheck=False)['nodes'] >= 0
