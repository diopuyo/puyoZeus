"""
chain.py のテスト

連鎖シミュレーション・グループ検出・重力処理・おじゃま落下を検証する。
"""

from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.chain import (
    MIN_ERASE_COUNT,
    ChainResult,
    ChainSimulator,
    ChainStep,
    PuyoGroup,
)


# ============================
# テスト用ヘルパー
# ============================


def empty_grid() -> list[list[int]]:
    """13×6 の全空グリッドを生成する。"""
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def board_from_grid(grid: list[list[int]]) -> Board:
    """グリッドリストから Board を生成する。"""
    return Board.from_list(grid)


def place_horizontal(
    grid: list[list[int]], row: int, col_start: int, color: int, count: int
) -> None:
    """指定行に color のぷよを count 個横に並べる (in-place)。"""
    for col in range(col_start, col_start + count):
        grid[row][col] = color


def place_vertical(
    grid: list[list[int]], row_start: int, col: int, color: int, count: int
) -> None:
    """指定列に color のぷよを count 個縦に並べる (in-place)。"""
    for row in range(row_start, row_start + count):
        grid[row][col] = color


@pytest.fixture
def sim() -> ChainSimulator:
    return ChainSimulator()


# ============================
# TestFindGroups
# ============================


class TestFindGroups:
    def test_empty_board_no_groups(self, sim: ChainSimulator):
        board = Board()
        assert sim.find_groups(board) == []

    def test_single_puyo_is_group_of_size_1(self, sim: ChainSimulator):
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        board = board_from_grid(grid)
        groups = sim.find_groups(board)
        assert len(groups) == 1
        assert groups[0].size == 1
        assert groups[0].color == COLOR_RED

    def test_four_horizontal_same_color(self, sim: ChainSimulator):
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, 4)
        board = board_from_grid(grid)
        groups = sim.find_groups(board)
        assert len(groups) == 1
        assert groups[0].size == 4

    def test_four_vertical_same_color(self, sim: ChainSimulator):
        grid = empty_grid()
        place_vertical(grid, 9, 0, COLOR_BLUE, 4)
        board = board_from_grid(grid)
        groups = sim.find_groups(board)
        assert len(groups) == 1
        assert groups[0].size == 4

    def test_l_shaped_group(self, sim: ChainSimulator):
        """
        L字型 (5個):
          RR
          R
          R
          R
        """
        grid = empty_grid()
        grid[8][0] = COLOR_RED
        grid[9][0] = COLOR_RED
        grid[10][0] = COLOR_RED
        grid[11][0] = COLOR_RED
        grid[11][1] = COLOR_RED
        board = board_from_grid(grid)
        groups = sim.find_groups(board)
        assert len(groups) == 1
        assert groups[0].size == 5

    def test_two_separate_groups_same_color(self, sim: ChainSimulator):
        """同色でも離れていれば2グループ。"""
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        grid[12][2] = COLOR_RED  # col 1 が空なので繋がっていない
        board = board_from_grid(grid)
        groups = sim.find_groups(board)
        assert len(groups) == 2

    def test_ojama_not_in_group(self, sim: ChainSimulator):
        """おじゃまはグループを形成しない。"""
        grid = empty_grid()
        for col in range(BOARD_COLS):
            grid[12][col] = COLOR_OJAMA
        board = board_from_grid(grid)
        groups = sim.find_groups(board)
        assert groups == []

    def test_ojama_adjacent_detected(self, sim: ChainSimulator):
        """4連グループに隣接するおじゃまが ojama_adjacent に入る。"""
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, 4)
        grid[11][0] = COLOR_OJAMA  # 4連の真上
        board = board_from_grid(grid)
        groups = sim.find_groups(board)
        assert len(groups) == 1
        assert (11, 0) in groups[0].ojama_adjacent

    def test_different_colors_separate_groups(self, sim: ChainSimulator):
        """赤と青が隣接しても別グループ。"""
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        grid[12][1] = COLOR_BLUE
        board = board_from_grid(grid)
        groups = sim.find_groups(board)
        assert len(groups) == 2
        colors = {g.color for g in groups}
        assert colors == {COLOR_RED, COLOR_BLUE}

    def test_ojama_adjacent_dedup_across_groups(self, sim: ChainSimulator):
        """複数グループが同じおじゃまに隣接できる (ojama_adjacent 重複許容)。"""
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        grid[12][2] = COLOR_BLUE
        grid[12][1] = COLOR_OJAMA  # 赤グループと青グループの両方に隣接
        board = board_from_grid(grid)
        groups = sim.find_groups(board)
        assert len(groups) == 2
        for g in groups:
            assert (12, 1) in g.ojama_adjacent


# ============================
# TestFindErasableGroups
# ============================


class TestFindErasableGroups:
    def test_group_of_4_is_erasable(self, sim: ChainSimulator):
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, MIN_ERASE_COUNT)
        board = board_from_grid(grid)
        erasable = sim.find_erasable_groups(board)
        assert len(erasable) == 1

    def test_group_of_3_is_not_erasable(self, sim: ChainSimulator):
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, 3)
        board = board_from_grid(grid)
        erasable = sim.find_erasable_groups(board)
        assert erasable == []

    def test_only_large_groups_returned(self, sim: ChainSimulator):
        """サイズ3 (非対象) とサイズ5 (対象) が混在 → サイズ5のみ返る。"""
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, 3)    # 非対象
        place_horizontal(grid, 11, 0, COLOR_BLUE, 5)   # 対象
        board = board_from_grid(grid)
        erasable = sim.find_erasable_groups(board)
        assert len(erasable) == 1
        assert erasable[0].color == COLOR_BLUE


# ============================
# TestApplyGravity
# ============================


class TestApplyGravity:
    def test_empty_board_unchanged(self, sim: ChainSimulator):
        board = Board()
        sim.apply_gravity(board)
        assert board.count_puyos() == 0

    def test_floating_puyo_falls_to_bottom(self, sim: ChainSimulator):
        """中段に浮いたぷよが最下段に落ちる。"""
        grid = empty_grid()
        grid[5][0] = COLOR_RED  # row=5 に浮かせる
        board = board_from_grid(grid)
        sim.apply_gravity(board)
        assert board.get(BOARD_ROWS - 1, 0) == COLOR_RED
        assert board.get(5, 0) == COLOR_EMPTY

    def test_gap_in_middle_fills(self, sim: ChainSimulator):
        """中間に空白があるとき上のぷよが落ちる。"""
        grid = empty_grid()
        grid[10][0] = COLOR_RED    # 上
        grid[11][0] = COLOR_EMPTY  # 空白 (gap)
        grid[12][0] = COLOR_BLUE   # 下 (すでに底にある)
        board = board_from_grid(grid)
        sim.apply_gravity(board)
        assert board.get(BOARD_ROWS - 1, 0) == COLOR_BLUE
        assert board.get(BOARD_ROWS - 2, 0) == COLOR_RED

    def test_relative_order_preserved(self, sim: ChainSimulator):
        """落下後も上下の相対順序が保たれる。"""
        grid = empty_grid()
        grid[0][2] = COLOR_RED
        grid[1][2] = COLOR_BLUE
        grid[2][2] = COLOR_GREEN
        board = board_from_grid(grid)
        sim.apply_gravity(board)
        # 下から: GREEN, BLUE, RED の順
        assert board.get(BOARD_ROWS - 1, 2) == COLOR_GREEN
        assert board.get(BOARD_ROWS - 2, 2) == COLOR_BLUE
        assert board.get(BOARD_ROWS - 3, 2) == COLOR_RED

    def test_no_cross_column_movement(self, sim: ChainSimulator):
        """重力で隣列には移動しない。"""
        grid = empty_grid()
        grid[0][1] = COLOR_RED
        board = board_from_grid(grid)
        sim.apply_gravity(board)
        assert board.get(BOARD_ROWS - 1, 1) == COLOR_RED
        assert board.get(BOARD_ROWS - 1, 0) == COLOR_EMPTY
        assert board.get(BOARD_ROWS - 1, 2) == COLOR_EMPTY


# ============================
# TestSimulate
# ============================


class TestSimulate:
    def test_no_chain_no_erasable(self, sim: ChainSimulator):
        """消えるグループがない盤面 → chain_count=0。"""
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        board = board_from_grid(grid)
        result = sim.simulate(board)
        assert result.chain_count == 0
        assert result.steps == []

    def test_single_chain_4_puyos(self, sim: ChainSimulator):
        """4ぷよ一列 → chain_count=1, total_erased=4。"""
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, 4)
        board = board_from_grid(grid)
        result = sim.simulate(board)
        assert result.chain_count == 1
        assert result.total_erased == 4

    def test_two_chain_sequence(self, sim: ChainSimulator):
        """
        1消しで重力落下 → さらに4つ揃う → chain_count=2。

        配置:
          col:  0  1  2  3
          row 8: R              ← 浮いている赤3個
          row 9: R
          row10: R
          row11: B  B  B  B    ← 1連鎖目で消える青4個
          row12: R              ← 1個の赤

        青が消えると col0 の赤3個が落下 → row9-12 に赤4個 → 2連鎖目
        """
        grid = empty_grid()
        grid[8][0] = COLOR_RED
        grid[9][0] = COLOR_RED
        grid[10][0] = COLOR_RED
        place_horizontal(grid, 11, 0, COLOR_BLUE, 4)
        grid[12][0] = COLOR_RED
        board = board_from_grid(grid)
        result = sim.simulate(board)
        assert result.chain_count == 2

    def test_chain_with_ojama(self, sim: ChainSimulator):
        """4連に隣接するおじゃまが消える → erased_ojama=1。"""
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, 4)
        grid[11][0] = COLOR_OJAMA
        board = board_from_grid(grid)
        result = sim.simulate(board)
        assert result.chain_count == 1
        assert result.steps[0].erased_ojama == 1
        assert result.total_ojama == 1

    def test_original_board_not_mutated(self, sim: ChainSimulator):
        """simulate 後、引数の board が変更されていない。"""
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, 4)
        board = board_from_grid(grid)
        original_count = board.count_puyos()
        sim.simulate(board)
        assert board.count_puyos() == original_count

    def test_steps_have_correct_chain_index(self, sim: ChainSimulator):
        """steps[0].chain_index==1, steps[1].chain_index==2。"""
        grid = empty_grid()
        # 2連鎖になる配置 (test_two_chain_sequence と同じ)
        grid[8][0] = COLOR_RED
        grid[9][0] = COLOR_RED
        grid[10][0] = COLOR_RED
        place_horizontal(grid, 11, 0, COLOR_BLUE, 4)
        grid[12][0] = COLOR_RED
        board = board_from_grid(grid)
        result = sim.simulate(board)
        assert result.steps[0].chain_index == 1
        assert result.steps[1].chain_index == 2

    def test_participating_cells_equals_total_erased(self, sim: ChainSimulator):
        """participating_cells == total_erased。"""
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, 4)
        board = board_from_grid(grid)
        result = sim.simulate(board)
        assert result.participating_cells == result.total_erased

    def test_final_board_has_no_erasable_groups(self, sim: ChainSimulator):
        """final_board に消えるグループがない。"""
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, 4)
        board = board_from_grid(grid)
        result = sim.simulate(board)
        assert sim.find_erasable_groups(result.final_board) == []

    def test_board_before_after_in_step(self, sim: ChainSimulator):
        """board_before と board_after は異なる (消去・落下が反映)。"""
        grid = empty_grid()
        place_horizontal(grid, 12, 0, COLOR_RED, 4)
        board = board_from_grid(grid)
        result = sim.simulate(board)
        step = result.steps[0]
        assert step.board_before != step.board_after

    def test_three_chain_known_board(self, sim: ChainSimulator):
        """
        既知の3連鎖盤面で chain_count==3 を確認。

        配置:
          col:  0  1  2  3
          row 5: G              ← 浮いている緑3個
          row 6: G
          row 7: G
          row 8: R              ← 浮いている赤3個
          row 9: R
          row10: R
          row11: B  B  B  B    ← step1: 青4個消え
          row12: R  G  G  G    ← 赤1個 + 緑3個

        step1: 青消え → col0 の赤3個落下 → 赤4個(row9-12) → step2
        step2: 赤消え → col0 の緑3個落下(row10-12) + col1-3の緑(row12) が連結 → 緑6個 → step3
        """
        grid = empty_grid()
        grid[5][0] = COLOR_GREEN
        grid[6][0] = COLOR_GREEN
        grid[7][0] = COLOR_GREEN
        grid[8][0] = COLOR_RED
        grid[9][0] = COLOR_RED
        grid[10][0] = COLOR_RED
        place_horizontal(grid, 11, 0, COLOR_BLUE, 4)
        grid[12][0] = COLOR_RED
        grid[12][1] = COLOR_GREEN
        grid[12][2] = COLOR_GREEN
        grid[12][3] = COLOR_GREEN
        board = board_from_grid(grid)
        result = sim.simulate(board)
        assert result.chain_count == 3


# ============================
# TestDropOjama
# ============================


class TestDropOjama:
    def test_drop_zero_ojama_unchanged(self, sim: ChainSimulator):
        """0個落とすと元盤面と同一。"""
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        board = board_from_grid(grid)
        result = sim.drop_ojama(board, 0)
        assert result == board

    def test_drop_6_fills_one_per_column(self, sim: ChainSimulator):
        """空盤面に6個 → 重力で各列の最下段に1個ずつ。"""
        board = Board()
        result = sim.drop_ojama(board, 6)
        for col in range(BOARD_COLS):
            assert result.get(BOARD_ROWS - 1, col) == COLOR_OJAMA

    def test_drop_7_fills_6_plus_1(self, sim: ChainSimulator):
        """7個 = 6個 (全列の最下段) + 1個 (col 0 の2段目)。"""
        board = Board()
        result = sim.drop_ojama(board, 7)
        # col 0 は2個 (row 12, 11)
        assert result.get(BOARD_ROWS - 1, 0) == COLOR_OJAMA
        assert result.get(BOARD_ROWS - 2, 0) == COLOR_OJAMA
        # col 1-5 は1個 (row 12)
        for col in range(1, BOARD_COLS):
            assert result.get(BOARD_ROWS - 1, col) == COLOR_OJAMA
            assert result.get(BOARD_ROWS - 2, col) == COLOR_EMPTY

    def test_drop_on_existing_puyos(self, sim: ChainSimulator):
        """ぷよが積まれた盤面にはその上に落ちる。"""
        grid = empty_grid()
        grid[12][0] = COLOR_RED  # col 0 の最下段に赤
        board = board_from_grid(grid)
        result = sim.drop_ojama(board, 1)
        # col 0 の最上段の空きは row 11
        assert result.get(11, 0) == COLOR_OJAMA
        assert result.get(12, 0) == COLOR_RED  # 元のぷよは残る

    def test_original_board_not_mutated(self, sim: ChainSimulator):
        """drop_ojama 後、引数の board が変更されない。"""
        board = Board()
        sim.drop_ojama(board, 6)
        assert board.count_puyos() == 0

    def test_negative_ojama_raises(self, sim: ChainSimulator):
        """負の個数は ValueError。"""
        board = Board()
        with pytest.raises(ValueError, match="おじゃま数が負の値"):
            sim.drop_ojama(board, -1)


# ============================
# TestIntegration
# ============================


class TestIntegration:
    def test_harassment_resistance_scenario(self, sim: ChainSimulator):
        """
        おじゃま30個落下後も本線が連鎖できるか。

        本線として縦4赤を仕込み、おじゃま30個を落とした後でも
        盤面が窒息していなければ連鎖が成立しうる。
        """
        grid = empty_grid()
        # col 5 に縦4赤 (おじゃまが落ちやすい col 0-4 を使わない列)
        place_vertical(grid, 9, 5, COLOR_RED, 4)
        board = board_from_grid(grid)

        board_after_ojama = sim.drop_ojama(board, 30)
        result = sim.simulate(board_after_ojama)

        # 30個のおじゃまで窒息しているか、または連鎖が成立
        is_dead = board_after_ojama.is_dead()
        has_chain = result.chain_count >= 1
        # 少なくとも一方が成立する
        assert is_dead or has_chain

    def test_full_chain_simulation_ojama_and_erase(self, sim: ChainSimulator):
        """
        おじゃまを消した後に連鎖が続くシナリオ。

        配置:
          col:  0  1  2  3
          row 8: R              ← 浮いている赤3個
          row 9: R
          row10: R
          row11: B  B  B  B    ← step1: 青消え + 隣接おじゃま3個消え
          row12: R  O  O  O    ← 赤1個 + おじゃま3個

        step1: 青4個消え → 隣接おじゃま(row12 col1-3)3個消え
               → col0の赤3個落下 → 赤4個(row9-12) → step2
        step2: 赤消え → chain_count=2
        """
        grid = empty_grid()
        grid[8][0] = COLOR_RED
        grid[9][0] = COLOR_RED
        grid[10][0] = COLOR_RED
        place_horizontal(grid, 11, 0, COLOR_BLUE, 4)
        grid[12][0] = COLOR_RED
        for col in range(1, 4):
            grid[12][col] = COLOR_OJAMA
        board = board_from_grid(grid)
        result = sim.simulate(board)
        assert result.chain_count == 2
        assert result.total_ojama == 3
