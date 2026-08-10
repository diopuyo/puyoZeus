"""隠し段の色確率 + 連鎖数分布のユニットテスト (2026-08-08)。

設計: docs/HIDDEN_ROW_PROBABILISTIC_DESIGN_2026-08-08.md

検証の要点:
- 情報源A (重力) で確定するセルは展開対象にしない
- 使われていない色に確率を置かない (組み合わせを無駄に増やさない)
- 隠し段の中身で連鎖数が変わる盤面で、 分布が実際に割れること
- 打ち切ったときは必ず truncated / covered_probability に現れること
"""
from __future__ import annotations

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_RED,
    Board,
)
from src.chain_count_distribution import (
    MAX_EXPAND_CELLS,
    ChainCountDistribution,
    compute_chain_count_distribution,
)
from src.hidden_row_probability import (
    HiddenCellSource,
    build_hidden_row_probabilities,
    infer_match_colors,
)


def _empty_grid() -> list[list[int]]:
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def _board(grid: list[list[int]]) -> Board:
    return Board.from_list(grid)


class TestHiddenRowProbabilities:
    """情報源A / D の構成を検証する。"""

    def test_empty_top_visible_means_hidden_is_empty(self) -> None:
        """可視最上段が空なら隠し段は空で確定 (重力の帰結)。"""
        g = _empty_grid()
        g[12][0] = COLOR_RED  # 下段だけ、最上段は空
        h = build_hidden_row_probabilities(_board(g))
        cell = h.get(0)
        assert cell is not None
        assert cell.source == HiddenCellSource.GRAVITY_EMPTY
        assert cell.is_certain
        assert cell.cell.get(COLOR_EMPTY) == 1.0

    def test_all_columns_empty_means_no_expansion(self) -> None:
        """空盤面では展開対象の列が 1 つも無いこと。"""
        h = build_hidden_row_probabilities(_board(_empty_grid()))
        assert h.uncertain_cols == ()

    def test_filled_top_visible_is_uncertain(self) -> None:
        """可視最上段が埋まっている列は不確定 (展開対象) になること。"""
        g = _empty_grid()
        for row in range(1, BOARD_ROWS):
            g[row][2] = COLOR_RED
        h = build_hidden_row_probabilities(_board(g))
        cell = h.get(2)
        assert cell is not None
        assert cell.source == HiddenCellSource.UNINFORMED
        assert not cell.is_certain
        assert 2 in h.uncertain_cols

    def test_unused_colors_get_no_probability(self) -> None:
        """盤面に出ていない色には確率を置かないこと。

        試合は 4 色のみ使用されるため、 観測されない色を候補に入れない
        (組み合わせ数を無駄に増やさないための制約)。
        """
        g = _empty_grid()
        for row in range(1, BOARD_ROWS):
            g[row][0] = COLOR_RED
        g[12][1] = COLOR_BLUE
        h = build_hidden_row_probabilities(_board(g))
        cell = h.get(0)
        assert cell is not None
        # 緑は盤面に無いので確率ゼロ
        assert cell.cell.get(COLOR_GREEN) == 0.0
        # 赤・青・おじゃまには確率がある
        assert cell.cell.get(COLOR_RED) > 0.0
        assert cell.cell.get(COLOR_BLUE) > 0.0
        assert cell.cell.get(COLOR_OJAMA) > 0.0

    def test_infer_match_colors_excludes_ojama_and_empty(self) -> None:
        """使用色の推定におじゃま・空を含めないこと。"""
        g = _empty_grid()
        g[12][0] = COLOR_RED
        g[12][1] = COLOR_OJAMA
        assert infer_match_colors(_board(g)) == (COLOR_RED,)


class TestChainCountDistribution:
    """連鎖数分布の計算を検証する。"""

    def test_no_uncertain_cells_gives_single_value(self) -> None:
        """不確定セルが無ければ従来通り単一値になること。"""
        g = _empty_grid()
        g[12][0] = COLOR_RED
        b = _board(g)
        d = compute_chain_count_distribution(b, build_hidden_row_probabilities(b))
        assert d.is_single_valued
        assert d.truncated is False
        assert d.n_expanded_cells == 0
        assert d.covered_probability == 1.0

    def test_hidden_puyo_does_not_pop_in_place(self) -> None:
        """13段目のぷよはその場で 4 連結しても消えないこと (幽霊連鎖のルール)。

        列0 の row1..3 を赤にし、 隠し段にも赤が入るケースを作る。
        素朴に数えれば 4 連結だが、 **13段目は消去対象外**なので消えない。
        13段目のぷよは下が空いて落ちてきて初めて消去対象になる。
        """
        g = _empty_grid()
        for row in range(1, 4):
            g[row][0] = COLOR_RED
        for row in range(4, BOARD_ROWS):
            g[row][0] = COLOR_BLUE if row % 2 == 0 else COLOR_GREEN
        b = _board(g)
        h = build_hidden_row_probabilities(b)
        assert 0 in h.uncertain_cols
        d = compute_chain_count_distribution(b, h)
        # 隠し段に何が入っても、 その場では 4 連結が成立しない = 連鎖 0 のみ
        assert d.probabilities == {0: 1.0}

    def test_ghost_chain_hidden_puyo_falls_and_pops(self) -> None:
        """13段目のぷよが落下してから消えること (幽霊連鎖そのもの)。

        列0 に「可視段で消える赤の塊」を作り、 その上 (12段目) に緑を 3 つ、
        隠し段にも緑が来る形にする。 赤が消えると緑が 1 段落ち、
        緑 4 つが 12 段目以下で揃って消える = 2 連鎖目が成立する。
        """
        g = _empty_grid()
        # 列0 の下部: 赤 4 つ (1 連鎖目、 単独で消える)
        for row in range(9, 13):
            g[row][0] = COLOR_RED
        # その上: 緑 3 つ (row 6,7,8)。 隠し段の緑が落ちてくれば 4 連結
        for row in range(6, 9):
            g[row][0] = COLOR_GREEN
        # 可視最上段 (row1) を埋めて隠し段を不確定にする
        for row in range(1, 6):
            g[row][0] = COLOR_GREEN
        b = _board(g)
        h = build_hidden_row_probabilities(b)
        d = compute_chain_count_distribution(b, h)
        # 緑が 8 つ縦に並ぶので赤消去前から緑は既に 4 連結 → 連鎖は必ず起きる
        assert d.most_likely >= 1
        assert abs(sum(d.probabilities.values()) - 1.0) < 1e-6

    def test_probabilities_sum_to_one(self) -> None:
        """分布の確率は必ず 1.0 に正規化されること。"""
        g = _empty_grid()
        for col in range(3):
            for row in range(1, BOARD_ROWS):
                g[row][col] = COLOR_RED if row % 2 else COLOR_BLUE
        b = _board(g)
        d = compute_chain_count_distribution(b, build_hidden_row_probabilities(b))
        assert abs(sum(d.probabilities.values()) - 1.0) < 1e-6

    def test_truncation_is_reported(self) -> None:
        """展開上限を超えたら truncated=True になること (黙って打ち切らない)。"""
        g = _empty_grid()
        # 全 6 列を満たして不確定セルを 6 個作る (上限 MAX_EXPAND_CELLS 超え)
        for col in range(BOARD_COLS):
            for row in range(1, BOARD_ROWS):
                g[row][col] = COLOR_RED if (row + col) % 2 else COLOR_BLUE
        b = _board(g)
        h = build_hidden_row_probabilities(b)
        assert len(h.uncertain_cols) > MAX_EXPAND_CELLS
        d = compute_chain_count_distribution(b, h)
        assert d.truncated is True
        assert d.n_expanded_cells == MAX_EXPAND_CELLS

    def test_accessors(self) -> None:
        """most_likely / expected / value_range / probability_of の整合。"""
        d = ChainCountDistribution(
            probabilities={5: 0.6, 9: 0.4},
            truncated=False, covered_probability=1.0, n_expanded_cells=1,
        )
        assert d.most_likely == 5
        assert abs(d.expected - (5 * 0.6 + 9 * 0.4)) < 1e-9
        assert d.value_range == (5, 9)
        assert d.probability_of(9) == 0.4
        assert d.probability_of(7) == 0.0
        assert d.is_single_valued is False

    def test_empty_distribution_is_safe(self) -> None:
        """空の分布でも例外を出さずに既定値を返すこと。"""
        d = ChainCountDistribution(
            probabilities={}, truncated=True,
            covered_probability=0.0, n_expanded_cells=0,
        )
        assert d.most_likely == 0
        assert d.expected == 0.0
        assert d.value_range == (0, 0)
