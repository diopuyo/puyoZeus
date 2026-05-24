"""V2.1 NextLinkedColorRefiner のテスト。"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    Board,
)
from src.next_linked_refiner import (
    NextLinkedColorRefiner,
    SKIP_COLORS,
)


def _make_board(cells: dict[tuple[int, int], int]) -> Board:
    """{(row, col): color} 形式から Board を生成。"""
    b = Board()
    for (row, col), color in cells.items():
        b.set(row, col, color)
    return b


def test_no_change_when_colors_match() -> None:
    """新規 2 セルが next_pair と一致 → 補正なし。"""
    prev = _make_board({})
    cur = _make_board({(11, 2): COLOR_RED, (12, 2): COLOR_BLUE})
    refiner = NextLinkedColorRefiner()
    res = refiner.refine(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.n_new_cells == 2
    assert res.n_corrected == 0
    assert res.refined.get(11, 2) == COLOR_RED
    assert res.refined.get(12, 2) == COLOR_BLUE


def test_correct_one_misclassified_color() -> None:
    """1 セルが誤認識 → next_pair に合わせて補正。"""
    prev = _make_board({})
    # 期待: (RED, BLUE) なのに (RED, PURPLE) と認識
    cur = _make_board({(11, 2): COLOR_RED, (12, 2): COLOR_PURPLE})
    refiner = NextLinkedColorRefiner()
    res = refiner.refine(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.n_new_cells == 2
    assert res.n_corrected == 1
    # RED は維持、PURPLE → BLUE 補正
    assert res.refined.get(11, 2) == COLOR_RED
    assert res.refined.get(12, 2) == COLOR_BLUE


def test_correct_both_misclassified() -> None:
    """2 セルとも誤認識 → 両方補正。"""
    prev = _make_board({})
    cur = _make_board({(11, 3): COLOR_GREEN, (12, 3): COLOR_PURPLE})
    refiner = NextLinkedColorRefiner()
    res = refiner.refine(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.n_corrected == 2
    out_colors = {res.refined.get(11, 3), res.refined.get(12, 3)}
    assert out_colors == {COLOR_RED, COLOR_BLUE}


def test_same_color_pair() -> None:
    """同色ペア (赤・赤) で 1 セル誤認 → 両方赤に補正。"""
    prev = _make_board({})
    cur = _make_board({(11, 2): COLOR_RED, (12, 2): COLOR_PURPLE})
    refiner = NextLinkedColorRefiner()
    res = refiner.refine(prev, cur, (COLOR_RED, COLOR_RED))
    assert res.n_corrected == 1
    assert res.refined.get(11, 2) == COLOR_RED
    assert res.refined.get(12, 2) == COLOR_RED


def test_skip_when_next_contains_empty() -> None:
    """next_pair に EMPTY → 補正スキップ。"""
    prev = _make_board({})
    cur = _make_board({(11, 2): COLOR_RED, (12, 2): COLOR_PURPLE})
    refiner = NextLinkedColorRefiner()
    res = refiner.refine(prev, cur, (COLOR_RED, COLOR_EMPTY))
    assert res.skipped_reason == "next_pair_not_definite"
    assert res.n_corrected == 0


def test_skip_when_next_contains_unknown() -> None:
    prev = _make_board({})
    cur = _make_board({(11, 2): COLOR_RED, (12, 2): COLOR_BLUE})
    refiner = NextLinkedColorRefiner()
    res = refiner.refine(prev, cur, (COLOR_RED, COLOR_UNKNOWN))
    assert res.skipped_reason == "next_pair_not_definite"


def test_skip_when_new_cell_count_not_two() -> None:
    """新規 1 セル / 3 セルなら補正スキップ。"""
    prev = _make_board({})
    cur = _make_board({(12, 2): COLOR_RED})
    refiner = NextLinkedColorRefiner()
    res = refiner.refine(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.skipped_reason == "new_cell_count=1"
    assert res.n_corrected == 0

    cur3 = _make_board({
        (10, 2): COLOR_RED, (11, 2): COLOR_BLUE, (12, 2): COLOR_GREEN,
    })
    res3 = refiner.refine(prev, cur3, (COLOR_RED, COLOR_BLUE))
    assert res3.skipped_reason == "new_cell_count=3"


def test_no_change_when_no_new_cells() -> None:
    """prev == cur (静止状態) → 補正なし。"""
    cells = {(12, 2): COLOR_RED, (12, 3): COLOR_BLUE}
    prev = _make_board(cells)
    cur = _make_board(cells)
    refiner = NextLinkedColorRefiner()
    res = refiner.refine(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.skipped_reason == "new_cell_count=0"
    assert res.n_corrected == 0


def test_hidden_row_excluded() -> None:
    """隠し段の変化は新規出現に含めない (UNKNOWN 混入回避)。"""
    prev = _make_board({})
    # row 0 = 隠し段、row 11/12 = 可視
    cur = _make_board({
        (0, 2): COLOR_PURPLE,  # 隠し段、無視されるべき
        (11, 2): COLOR_RED,
        (12, 2): COLOR_BLUE,
    })
    refiner = NextLinkedColorRefiner()
    res = refiner.refine(prev, cur, (COLOR_RED, COLOR_BLUE))
    # 隠し段除外で新規 = 2、一致 → 補正なし
    assert res.n_new_cells == 2
    assert res.n_corrected == 0


def test_skip_color_set_includes_ojama() -> None:
    """OJAMA を含む next_pair は確定的でない (= 不正な使い方) としてスキップ。"""
    assert COLOR_OJAMA in SKIP_COLORS
    assert COLOR_EMPTY in SKIP_COLORS
    assert COLOR_UNKNOWN in SKIP_COLORS
    # 通常色は SKIP に含めない
    assert COLOR_RED not in SKIP_COLORS
    assert COLOR_YELLOW not in SKIP_COLORS
