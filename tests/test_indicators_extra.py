"""
拡張4指標のテスト (mayah/puyoai 先行研究ベース)

- ShapeScoreIndicator: 形評価 (U字+土台)
- TouchingDensityIndicator: 接ぷよ密度
- TailHeightIndicator: 連鎖発火点低さ
- ColorVarianceIndicator: 色集中度

各指標 4-5 件で 0/1 端値・典型盤面・空盤面・端値耐性を検証する。
"""

from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.old.indicators import (
    INDICATOR_COLOR_VARIANCE,
    INDICATOR_SHAPE_SCORE,
    INDICATOR_TAIL_HEIGHT,
    INDICATOR_TOUCHING_DENSITY,
    SHAPE_IDEAL_HEIGHTS,
    ColorVarianceIndicator,
    IndicatorCalculator,
    ShapeScoreIndicator,
    TailHeightIndicator,
    TouchingDensityIndicator,
)


# ============================
# テスト用ヘルパー
# ============================


def empty_grid() -> list[list[int]]:
    """13×6 の全空グリッド。"""
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def board_from_grid(grid: list[list[int]]) -> Board:
    return Board.from_list(grid)


def make_empty_board() -> Board:
    return board_from_grid(empty_grid())


def make_ushape_board() -> Board:
    """SHAPE_IDEAL_HEIGHTS と一致する U 字型盤面 (理想形)。"""
    grid = empty_grid()
    for col, h in enumerate(SHAPE_IDEAL_HEIGHTS):
        for d in range(h):
            # 適度に色を変えて連鎖が起きないようにする (高さ評価のみ確認)
            color = COLOR_RED if (col + d) % 2 == 0 else COLOR_BLUE
            grid[BOARD_ROWS - 1 - d][col] = color
    return board_from_grid(grid)


def make_flat_tall_board() -> Board:
    """全列ほぼ満タン (U字とは逆形)。"""
    grid = empty_grid()
    for col in range(BOARD_COLS):
        for d in range(11):  # 高さ 11 で揃える
            color = COLOR_RED if (col + d) % 2 == 0 else COLOR_BLUE
            grid[BOARD_ROWS - 1 - d][col] = color
    return board_from_grid(grid)


def make_chain_board_low_trigger() -> Board:
    """発火点が最下段 (row=12) に来る単発消し盤面。"""
    grid = empty_grid()
    # row 12 に赤4個 → 1連鎖、発火点 row=12 (= 最下段)
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_RED
    grid[12][2] = COLOR_RED
    grid[12][3] = COLOR_RED
    return board_from_grid(grid)


def make_chain_board_high_trigger() -> Board:
    """発火点が高い位置 (row=2) に来る単発消し盤面。

    赤4個を row=2 に浮かせ、下段は色を交互にして 4 連結を作らないよう支える。
    これにより連鎖シミュレータの最初のステップは row=2 の赤4個のみ消去する。
    """
    grid = empty_grid()
    # col0-3 の row 3〜12 を 緑/黄/緑/黄... で埋める (4 連結作らない)
    fill_pattern = [COLOR_GREEN, COLOR_YELLOW, COLOR_GREEN, COLOR_YELLOW]
    for col in range(4):
        for r in range(3, BOARD_ROWS):
            # row * col の値で色をローテーション → 4連結不成立
            grid[r][col] = fill_pattern[(r + col) % 4]
        grid[2][col] = COLOR_RED
    return board_from_grid(grid)


def make_one_color_dense_board() -> Board:
    """単色クラスター (低分散)。"""
    grid = empty_grid()
    # 1 色クラスター (左下に固める) → 重心からの平均距離が小さい
    for r in range(11, BOARD_ROWS):
        for c in range(2):
            grid[r][c] = COLOR_RED
    return board_from_grid(grid)


def make_one_color_scattered_board() -> Board:
    """単色を盤面の四隅に散らす (高分散)。"""
    grid = empty_grid()
    grid[12][0] = COLOR_RED
    grid[12][5] = COLOR_RED
    grid[7][0] = COLOR_RED
    grid[7][5] = COLOR_RED
    return board_from_grid(grid)


def make_high_touching_board() -> Board:
    """同色接続が多い盤面 (横一列)。"""
    grid = empty_grid()
    # row 12 に赤6個 → 5 ペアの水平接続
    for c in range(BOARD_COLS):
        grid[12][c] = COLOR_RED
    return board_from_grid(grid)


def make_low_touching_board() -> Board:
    """色がバラバラで接続がほぼ無い盤面。"""
    grid = empty_grid()
    # 1 行に交互に色 → 水平接続 0
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_BLUE
    grid[12][2] = COLOR_GREEN
    grid[12][3] = COLOR_YELLOW
    grid[12][4] = COLOR_RED
    grid[12][5] = COLOR_BLUE
    return board_from_grid(grid)


# ============================
# ShapeScoreIndicator
# ============================


class TestShapeScoreIndicator:
    def test_name(self):
        assert ShapeScoreIndicator().name == INDICATOR_SHAPE_SCORE

    def test_empty_board_high_score(self):
        """空盤面: 全列高さ 0 → 偏差 = sum(IDEAL) で必ずしも満点ではない。"""
        ind = ShapeScoreIndicator()
        result = ind.compute(make_empty_board())
        # 偏差 = 10+9+7+7+9+10 = 52 ≫ MAX(24) → score=0
        assert result.score == 0.0
        assert result.detail["heights"] == [0, 0, 0, 0, 0, 0]

    def test_ushape_ideal_full_score(self):
        """理想 U 字形: deviation=0 → score=1.0。"""
        ind = ShapeScoreIndicator()
        result = ind.compute(make_ushape_board())
        assert result.score == pytest.approx(1.0)
        assert result.raw_value == 0.0

    def test_flat_tall_low_score(self):
        """全列満タン: 偏差大 → score 低い。"""
        ind = ShapeScoreIndicator()
        result = ind.compute(make_flat_tall_board())
        assert result.score < 0.5
        assert result.raw_value > 0

    def test_score_in_range(self):
        """スコアは常に [0,1]。"""
        ind = ShapeScoreIndicator()
        for board in [make_empty_board(), make_ushape_board(),
                      make_flat_tall_board()]:
            r = ind.compute(board)
            assert 0.0 <= r.score <= 1.0


# ============================
# TouchingDensityIndicator
# ============================


class TestTouchingDensityIndicator:
    def test_name(self):
        assert TouchingDensityIndicator().name == INDICATOR_TOUCHING_DENSITY

    def test_empty_board_zero(self):
        """空盤面: ぷよ無しなので 0。"""
        result = TouchingDensityIndicator().compute(make_empty_board())
        assert result.score == 0.0

    def test_high_touching(self):
        """横一列同色: 5 ペア / 6 puyo ≈ 0.83 ratio → score ≈ 0.55。"""
        result = TouchingDensityIndicator().compute(make_high_touching_board())
        assert result.score > 0.4
        assert result.detail["pairs"] == 5

    def test_low_touching(self):
        """全部別色: ペア 0 → score 0。"""
        result = TouchingDensityIndicator().compute(make_low_touching_board())
        assert result.score == 0.0
        assert result.detail["pairs"] == 0

    def test_score_in_range(self):
        for board in [make_empty_board(), make_high_touching_board(),
                      make_low_touching_board(), make_one_color_dense_board()]:
            r = TouchingDensityIndicator().compute(board)
            assert 0.0 <= r.score <= 1.0


# ============================
# TailHeightIndicator
# ============================


class TestTailHeightIndicator:
    def test_name(self):
        assert TailHeightIndicator().name == INDICATOR_TAIL_HEIGHT

    def test_no_chain_neutral(self):
        """連鎖無し: 中立値 0.5。"""
        result = TailHeightIndicator().compute(make_empty_board())
        assert result.score == pytest.approx(0.5)
        assert result.detail.get("no_chain") is True

    def test_low_trigger_high_score(self):
        """最下段 (row=12) で発火 → 高さ=1 → score ≈ 1 - 1/13 ≈ 0.92。"""
        result = TailHeightIndicator().compute(make_chain_board_low_trigger())
        assert result.score > 0.85
        assert result.detail["trigger_row"] == 12

    def test_high_trigger_low_score(self):
        """row=2 発火 → 高さ=11 → score ≈ 1 - 11/13 ≈ 0.15。"""
        result = TailHeightIndicator().compute(make_chain_board_high_trigger())
        assert result.score < 0.3
        assert result.detail["trigger_row"] == 2

    def test_score_in_range(self):
        for board in [make_empty_board(), make_chain_board_low_trigger(),
                      make_chain_board_high_trigger()]:
            r = TailHeightIndicator().compute(board)
            assert 0.0 <= r.score <= 1.0


# ============================
# ColorVarianceIndicator
# ============================


class TestColorVarianceIndicator:
    def test_name(self):
        assert ColorVarianceIndicator().name == INDICATOR_COLOR_VARIANCE

    def test_empty_board_neutral(self):
        """空盤面: 色データ無し → 中立値 0.5。"""
        result = ColorVarianceIndicator().compute(make_empty_board())
        assert result.score == pytest.approx(0.5)

    def test_concentrated_color_high_score(self):
        """密集した単色クラスター → 距離小 → score 高い。"""
        result = ColorVarianceIndicator().compute(make_one_color_dense_board())
        assert result.score > 0.7

    def test_scattered_color_low_score(self):
        """四隅散らばし → 距離大 → score 低い。"""
        result = ColorVarianceIndicator().compute(make_one_color_scattered_board())
        assert result.score < 0.5

    def test_score_in_range(self):
        for board in [make_empty_board(), make_one_color_dense_board(),
                      make_one_color_scattered_board()]:
            r = ColorVarianceIndicator().compute(board)
            assert 0.0 <= r.score <= 1.0


# ============================
# IndicatorCalculator 統合
# ============================


class TestExtraIndicatorIntegration:
    def test_extra_results_in_set(self):
        """compute_all() の結果に拡張4指標が全部含まれる。"""
        calc = IndicatorCalculator()
        s = calc.compute_all(make_ushape_board())
        for name in (INDICATOR_SHAPE_SCORE, INDICATOR_TOUCHING_DENSITY,
                     INDICATOR_TAIL_HEIGHT, INDICATOR_COLOR_VARIANCE):
            assert name in s.results
            assert 0.0 <= s.results[name].score <= 1.0

    def test_extra_attrs_consistent_with_results(self):
        """IndicatorSet の属性と results 辞書スコアが一致する。"""
        calc = IndicatorCalculator()
        s = calc.compute_all(make_high_touching_board())
        assert s.shape_score == s.results[INDICATOR_SHAPE_SCORE].score
        assert s.touching_density == s.results[INDICATOR_TOUCHING_DENSITY].score
        assert s.tail_height_score == s.results[INDICATOR_TAIL_HEIGHT].score
        assert s.color_variance_score == s.results[INDICATOR_COLOR_VARIANCE].score
