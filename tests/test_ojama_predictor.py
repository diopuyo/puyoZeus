"""C-2 OjamaPredictor (W-γ) のテスト."""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_RED,
    Board,
)
from src.ojama_predictor import OJAMA_DIVISOR_GAME, OjamaPredictor


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _board_with_ojama(n: int) -> Board:
    """下から n 個のおじゃまセルを置いた盤面."""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    placed = 0
    for r in range(BOARD_ROWS - 1, -1, -1):
        for c in range(BOARD_COLS):
            if placed >= n:
                break
            grid[r][c] = COLOR_OJAMA
            placed += 1
        if placed >= n:
            break
    return Board.from_list(grid)


def test_initial_state_zero() -> None:
    p = OjamaPredictor()
    assert p.pending_for("1P") == 0
    assert p.pending_for("2P") == 0


def test_p1_chain_adds_pending_to_p2() -> None:
    """1P の score 増 → 2P 側 pending 追加."""
    p = OjamaPredictor()
    p.update(p1_score_delta=1400, p2_score_delta=0)
    # 1400 / 70 = 20 個
    assert p.pending_for("2P") == 20
    assert p.pending_for("1P") == 0


def test_p2_chain_adds_pending_to_p1() -> None:
    p = OjamaPredictor()
    p.update(p1_score_delta=0, p2_score_delta=2100)
    # 2100 / 70 = 30 個
    assert p.pending_for("1P") == 30
    assert p.pending_for("2P") == 0


def test_simultaneous_chain_cancels() -> None:
    """両者同時発火 → pending 相殺."""
    p = OjamaPredictor()
    # 1P→2P 20 個、2P→1P 30 個 → 相殺後 1P 受 10 個、2P 受 0 個
    p.update(p1_score_delta=1400, p2_score_delta=2100)
    assert p.pending_for("1P") == 10
    assert p.pending_for("2P") == 0


def test_pending_accumulates_across_frames() -> None:
    """連続 score 増加で pending 蓄積."""
    p = OjamaPredictor()
    p.update(p1_score_delta=700, p2_score_delta=0)  # 10 個 → 2P
    p.update(p1_score_delta=350, p2_score_delta=0)  # 5 個 → 2P
    assert p.pending_for("2P") == 15


def test_negative_score_delta_ignored() -> None:
    """負の score_delta (異常値) は無視."""
    p = OjamaPredictor()
    p.update(p1_score_delta=-500, p2_score_delta=0)
    assert p.pending_for("2P") == 0


def test_ojama_drop_decrements_pending() -> None:
    """盤面のおじゃま数が増 → 落下発生 → pending 減少."""
    p = OjamaPredictor()
    p1_board0 = _empty_board()
    p1_board1 = _board_with_ojama(5)  # 5 個落下
    # 2P が連鎖 → 1P pending = 30
    p.update(
        p1_score_delta=0, p2_score_delta=2100,
        p1_board=p1_board0, p2_board=_empty_board(),
    )
    assert p.pending_for("1P") == 30
    # 次 frame: 1P 盤面に 5 個おじゃま増 (落下) → pending = 25
    p.update(
        p1_score_delta=0, p2_score_delta=0,
        p1_board=p1_board1, p2_board=_empty_board(),
    )
    assert p.pending_for("1P") == 25


def test_reset_clears_state() -> None:
    p = OjamaPredictor()
    p.update(p1_score_delta=1400, p2_score_delta=0)
    assert p.pending_for("2P") == 20
    p.reset()
    assert p.pending_for("2P") == 0
    assert p.pending_for("1P") == 0


def test_unknown_side_returns_zero() -> None:
    p = OjamaPredictor()
    assert p.pending_for("3P") == 0


def test_ojama_drop_overshoot_clamps_to_zero() -> None:
    """既存 pending より多くおじゃまが出現しても、pending は負にならない."""
    p = OjamaPredictor()
    # 1P pending = 5
    p.update(
        p1_score_delta=0, p2_score_delta=350,
        p1_board=_empty_board(), p2_board=_empty_board(),
    )
    assert p.pending_for("1P") == 5
    # 1P 盤面に 20 個おじゃま (相手がトリガしてないが盤面状態が変わった想定)
    # → pending=0 にクランプ
    p.update(
        p1_score_delta=0, p2_score_delta=0,
        p1_board=_board_with_ojama(20), p2_board=_empty_board(),
    )
    assert p.pending_for("1P") == 0
