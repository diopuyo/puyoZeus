"""W3.0 hidden_row_inferrer のテスト。"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    Board,
)
from src.hidden_row_inferrer import HiddenInferenceResult, infer_hidden_row


def _make_board(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (r, c), color in cells.items():
        b.set(r, c, color)
    return b


def test_two_new_cells_hidden_is_certain_empty() -> None:
    """新規 2 セル → 隠し段 EMPTY 確定。"""
    prev = _make_board({})
    cur = _make_board({(11, 2): COLOR_RED, (12, 2): COLOR_BLUE})
    pb, res = infer_hidden_row(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.n_new_cells == 2
    assert res.skipped_reason is None
    # 隠し段 row 0 全列 EMPTY 確定
    for c in range(BOARD_COLS):
        cell = pb.cell(0, c)
        assert cell.most_likely() == (COLOR_EMPTY, 1.0)


def test_one_new_cell_hidden_distribution() -> None:
    """新規 1 セル + observed が row=HIDDEN_ROWS → 同列の隠し段に missing color 分布。"""
    prev = _make_board({})
    # row=1 (= HIDDEN_ROWS) に RED 1 つ → ペアの bot が同列の隠し段にある
    cur = _make_board({(1, 2): COLOR_RED})
    pb, res = infer_hidden_row(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.n_new_cells == 1
    assert res.skipped_reason is None
    # 列 2 の row 0 は BLUE 確率 1.0
    cell_2 = pb.cell(0, 2)
    assert cell_2.get(COLOR_BLUE) == 1.0
    # 他の列の row 0 は EMPTY 確定
    for c in range(BOARD_COLS):
        if c == 2:
            continue
        assert pb.cell(0, c).most_likely() == (COLOR_EMPTY, 1.0)


def test_one_new_cell_below_hidden_three_candidates() -> None:
    """新規 1 セル + row > HIDDEN_ROWS → 自列+左右の 3 列分散。"""
    prev = _make_board({})
    # row=12 に RED → 横置きペアの可能性、左右の隣接列も候補
    cur = _make_board({(12, 2): COLOR_RED})
    pb, res = infer_hidden_row(prev, cur, (COLOR_RED, COLOR_BLUE))
    # 候補列は 1, 2, 3 (横置きペアで隣接)
    # ただし、これは設計仕様上、row > HIDDEN_ROWS の単独出現は稀
    # candidate_cols が 3、prob = 1/3 ずつ
    assert res.n_new_cells == 1
    cell_1 = pb.cell(0, 1)
    cell_2 = pb.cell(0, 2)
    cell_3 = pb.cell(0, 3)
    assert abs(cell_1.get(COLOR_BLUE) - 1/3) < 0.01
    assert abs(cell_2.get(COLOR_BLUE) - 1/3) < 0.01
    assert abs(cell_3.get(COLOR_BLUE) - 1/3) < 0.01


def test_skip_when_next_pair_none() -> None:
    """next_pair=None → スキップ。"""
    prev = _make_board({})
    cur = _make_board({(12, 2): COLOR_RED})
    pb, res = infer_hidden_row(prev, cur, None)
    assert res.skipped_reason == "no_next_pair"


def test_skip_when_next_pair_contains_ojama() -> None:
    """next_pair に OJAMA → スキップ。"""
    prev = _make_board({})
    cur = _make_board({(12, 2): COLOR_RED})
    pb, res = infer_hidden_row(prev, cur, (COLOR_RED, COLOR_OJAMA))
    assert res.skipped_reason == "next_pair_not_definite"


def test_no_new_cells_returns_empty_result() -> None:
    """変化なし → 推論結果なし。"""
    cells = {(12, 2): COLOR_RED}
    prev = _make_board(cells)
    cur = _make_board(cells)
    pb, res = infer_hidden_row(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.n_new_cells == 0
    assert "n_new_cells=0" in res.skipped_reason


def test_three_new_cells_skipped() -> None:
    prev = _make_board({})
    cur = _make_board({
        (10, 2): COLOR_RED, (11, 2): COLOR_BLUE, (12, 2): COLOR_RED,
    })
    pb, res = infer_hidden_row(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.n_new_cells == 3
    assert "n_new_cells=3" in res.skipped_reason
