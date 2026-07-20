"""scripts/collect_indicators_v2.py の CSV 列定義・指標充填ロジックのユニットテスト。

テスト方針 (tests/test_board_npz_dump.py に準拠):
- 動画認識・重い collect 実行は一切しない。
- 合成 Board で _fill_indicator_columns を直接呼び、列の存在・値域を検証する。

対象: XII board sim 本命指標 (飽和連鎖量・発火点・多色発火・副砲・同時消し
リッチネス) を CSV 列として追加した際の統合が正しいことを確認する。
"""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_RED,
    COLOR_BLUE,
    Board,
)
import src.indicators_v2 as iv
import scripts.collect_indicators_v2 as collect_mod


# ============================
# 盤面ビルダー
# ============================


def _empty_grid() -> list[list[int]]:
    return [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]


def _empty_board() -> Board:
    return Board.from_list(_empty_grid())


def _chain_ready_board() -> Board:
    """4 連結 (赤) を最下段に置いた発火可能盤面。連鎖シミュが有意に走る。"""
    g = _empty_grid()
    g[12][0] = COLOR_RED
    g[12][1] = COLOR_RED
    g[11][0] = COLOR_RED
    g[11][1] = COLOR_RED
    # 副砲候補: 別色を近傍に配置
    g[12][3] = COLOR_BLUE
    g[12][4] = COLOR_BLUE
    return Board.from_list(g)


# ============================
# XII 列が ALL_COLUMNS / INDICATOR_COLUMNS に含まれること
# ============================


# CSV 末尾に追加された XII 5 指標の列名 (順序保持で確認)
_XII_COLUMNS: tuple[str, ...] = (
    "saturated_chain_count", "saturated_chain_count_raw",
    "ignition_point_count", "ignition_point_count_raw",
    "multi_color_ignition", "multi_color_ignition_raw",
    "sub_chain_count", "sub_chain_count_raw",
    "simultaneous_pop_richness", "simultaneous_pop_richness_raw",
)


def test_xii_columns_present_in_all_columns() -> None:
    """XII 5 指標 (score/raw 計 10 列) が ALL_COLUMNS に含まれること。"""
    for col in _XII_COLUMNS:
        assert col in collect_mod.ALL_COLUMNS, f"{col} が ALL_COLUMNS に無い"


def test_xii_columns_are_tail_of_indicator_columns() -> None:
    """XII 5 指標が INDICATOR_COLUMNS の末尾 10 列であること (順序保持ルール)。

    既存列の位置を変更していないことの回帰検知。
    """
    tail = collect_mod.INDICATOR_COLUMNS[-len(_XII_COLUMNS):]
    assert tail == _XII_COLUMNS


def test_ukeyasusa_still_precedes_xii_columns() -> None:
    """既存最終列 (ukeyasusa/ukeyasusa_raw) が XII 列の直前に維持されていること。"""
    cols = collect_mod.INDICATOR_COLUMNS
    idx = cols.index("ukeyasusa_raw")
    assert cols[idx + 1] == "saturated_chain_count"


# ============================
# _fill_indicator_columns が XII 列を 0-1 範囲で埋めること
# ============================


def _call_fill(board: Board) -> dict[str, object]:
    """_fill_indicator_columns を呼び出す共通ヘルパ。"""
    row: dict[str, object] = {}
    total_conn, _ = iv.connectivity_observation(board)
    collect_mod._fill_indicator_columns(
        row, board, tsumo=10, elapsed_sec=30.0, net=0, forecast=0,
        total_conn=total_conn,
    )
    return row


@pytest.mark.parametrize("board", [_empty_board(), _chain_ready_board()])
def test_fill_indicator_columns_xii_score_in_range(board: Board) -> None:
    """XII 5 指標の score が 0-1 範囲・NaN なしで埋まること。"""
    row = _call_fill(board)
    for score_col in _XII_COLUMNS[0::2]:  # score 列のみ (偶数 index)
        val = row[score_col]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0, f"{score_col}={val} が 0-1 範囲外"
        assert val == val, f"{score_col} が NaN"


@pytest.mark.parametrize("board", [_empty_board(), _chain_ready_board()])
def test_fill_indicator_columns_xii_raw_present(board: Board) -> None:
    """XII 5 指標の raw が非負の数値で埋まること (NaN なし)。"""
    row = _call_fill(board)
    for raw_col in _XII_COLUMNS[1::2]:  # raw 列のみ (奇数 index)
        val = row[raw_col]
        assert isinstance(val, float)
        assert val >= 0.0, f"{raw_col}={val} が負値"
        assert val == val, f"{raw_col} が NaN"


def test_fill_indicator_columns_empty_board_zero_saturated_chain() -> None:
    """空盤面は 1 個追加しても発火できないため saturated_chain_count=0。"""
    row = _call_fill(_empty_board())
    assert row["saturated_chain_count_raw"] == pytest.approx(0.0)
    assert row["saturated_chain_count"] == pytest.approx(0.0)


def test_fill_indicator_columns_chain_ready_board_positive_saturated_chain() -> None:
    """発火可能盤面 (4 連結あり) は 1 個追加無しでも既に発火するため raw > 0。"""
    row = _call_fill(_chain_ready_board())
    assert row["saturated_chain_count_raw"] > 0.0


def test_fill_indicator_columns_does_not_break_existing_columns() -> None:
    """XII 追加後も既存列 (例: ukeyasusa) が引き続き埋まること (後方互換)。"""
    row = _call_fill(_chain_ready_board())
    assert "ukeyasusa" in row
    assert 0.0 <= row["ukeyasusa"] <= 1.0
    assert "board_puyo_total" in row
