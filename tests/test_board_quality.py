"""src/board_quality.py のユニットテスト (幻盤面フィルタ、2026-08-08)。

非試合画面 (対戦カード紹介・ロビー・順位表) で誤認識された満杯おじゃま盤面を
学習データから除外するための判定を検証する。 実戦であり得る盤面を誤って
幻と判定しないこと (偽陽性の抑止) を特に重視する。
"""
from __future__ import annotations

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_RED,
)
from src.board_quality import (
    PHANTOM_MIN_NONEMPTY,
    VISIBLE_ROW_LO,
    count_phantom_boards,
    is_phantom_board,
    phantom_board_mask,
)


def _grid(fill: int = COLOR_EMPTY) -> np.ndarray:
    """全セルを fill で埋めた盤面配列を返す。"""
    return np.full((BOARD_ROWS, BOARD_COLS), fill, dtype=np.int8)


def _fill_visible_rows(grid: np.ndarray, n_rows: int, color: int) -> np.ndarray:
    """可視段の下から n_rows 行を color で埋める。"""
    for r in range(BOARD_ROWS - n_rows, BOARD_ROWS):
        grid[r, :] = color
    return grid


class TestPhantomDetection:
    """幻盤面 (非試合画面由来) の検出を検証する。"""

    def test_empty_board_is_not_phantom(self) -> None:
        """空盤面は幻ではない (0 除算もしない)。"""
        assert is_phantom_board(_grid()) is False

    def test_full_ojama_board_is_phantom(self) -> None:
        """全面おじゃま盤面 = 実戦で安定継続し得ない → 幻。"""
        assert is_phantom_board(_grid(COLOR_OJAMA)) is True

    def test_full_color_board_is_not_phantom(self) -> None:
        """満杯でも色ぷよ主体なら実戦であり得る → 幻ではない。"""
        assert is_phantom_board(_grid(COLOR_RED)) is False

    def test_ojama_but_sparse_is_not_phantom(self) -> None:
        """おじゃま比率が高くてもセル数が少なければ幻ではない。

        相手の連鎖で数個だけおじゃまが降った通常の局面を守る。
        """
        g = _fill_visible_rows(_grid(), 3, COLOR_OJAMA)  # 18 セル
        assert is_phantom_board(g) is False

    def test_mixed_board_below_ratio_is_not_phantom(self) -> None:
        """満杯でもおじゃま比率が閾値未満なら幻ではない。

        おじゃまを大量に受けた終盤の実戦盤面 (色ぷよの土台が残る) を守る。
        """
        g = _grid(COLOR_OJAMA)
        # 下 5 行を色ぷよにする → おじゃま比率 = (12-5)/12 ≈ 0.58 < 0.7
        _fill_visible_rows(g, 5, COLOR_BLUE)
        assert is_phantom_board(g) is False

    def test_hidden_row_is_excluded_from_judgement(self) -> None:
        """隠し段 (row0) は判定に含めない (窒息判定と同じ扱い)。"""
        g = _grid(COLOR_OJAMA)
        g[0, :] = COLOR_EMPTY  # 隠し段だけ空
        # 可視段は全ておじゃまのままなので判定は変わらない
        assert is_phantom_board(g) is True

    def test_boundary_nonempty_count(self) -> None:
        """非空セル数が閾値ちょうどなら幻、 1 つ少なければ幻でない。"""
        rows_needed = PHANTOM_MIN_NONEMPTY // BOARD_COLS  # 48/6 = 8 行
        g = _fill_visible_rows(_grid(), rows_needed, COLOR_OJAMA)
        assert is_phantom_board(g) is True
        g2 = _fill_visible_rows(_grid(), rows_needed, COLOR_OJAMA)
        g2[BOARD_ROWS - rows_needed, 0] = COLOR_EMPTY  # 1 セル減らす
        assert is_phantom_board(g2) is False


class TestPhantomMaskShapes:
    """配列入力時の形状・集計を検証する。"""

    def test_mask_shape_for_batch(self) -> None:
        """(n,13,6) 入力で shape (n,) の bool マスクが返ること。"""
        grids = np.stack([_grid(), _grid(COLOR_OJAMA), _grid(COLOR_RED)])
        mask = phantom_board_mask(grids)
        assert mask.shape == (3,)
        assert mask.dtype == np.bool_
        assert mask.tolist() == [False, True, False]

    def test_mask_shape_for_single(self) -> None:
        """単体盤面 (13,6) 入力でも shape (1,) で返ること。"""
        assert phantom_board_mask(_grid(COLOR_OJAMA)).shape == (1,)

    def test_count_phantom_boards(self) -> None:
        """(幻盤面数, 総数) が正しく返ること。"""
        grids = np.stack([_grid(COLOR_OJAMA), _grid(), _grid(COLOR_OJAMA)])
        assert count_phantom_boards(grids) == (2, 3)

    def test_visible_row_lo_matches_death_row(self) -> None:
        """可視段開始行は窒息判定と同じ 1 (隠し段 row0 を除く)。"""
        assert VISIBLE_ROW_LO == 1
