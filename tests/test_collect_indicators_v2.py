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
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_PURPLE,
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

# CSV 末尾に追加された XIV (近未来最大火力 K=1..5) 5 指標の列名
# (2026-07-22 本番統合、順序保持で確認)。
_XIV_COLUMNS: tuple[str, ...] = (
    "near_future_fire_k1", "near_future_fire_k1_raw",
    "near_future_fire_k2", "near_future_fire_k2_raw",
    "near_future_fire_k3", "near_future_fire_k3_raw",
    "near_future_fire_k4", "near_future_fire_k4_raw",
    "near_future_fire_k5", "near_future_fire_k5_raw",
)


def test_xii_columns_present_in_all_columns() -> None:
    """XII 5 指標 (score/raw 計 10 列) が ALL_COLUMNS に含まれること。"""
    for col in _XII_COLUMNS:
        assert col in collect_mod.ALL_COLUMNS, f"{col} が ALL_COLUMNS に無い"


def test_xii_columns_precede_xiv_columns() -> None:
    """XII 5 指標が XIV (近未来最大火力) 追加前と同じ相対位置を保っていること。

    XIV が新たな末尾になったため「XII が末尾10列」という旧アサーションは
    そのままでは成立しない (これは正当な末尾追加の結果であり退行ではない)。
    XII ブロックが XIV ブロックの直前に連続していることを確認し、
    既存列の位置自体は変更していないことを回帰検知する。
    """
    total = len(_XII_COLUMNS) + len(_XIV_COLUMNS)
    tail_block = collect_mod.INDICATOR_COLUMNS[-total:]
    assert tail_block == _XII_COLUMNS + _XIV_COLUMNS


def test_xiv_columns_are_tail_of_indicator_columns() -> None:
    """XIV (近未来最大火力) 5 指標が INDICATOR_COLUMNS の末尾10列であること。

    2026-07-22 本番統合、新指標は常に末尾追加 (CLAUDE.md 規約)。
    """
    tail = collect_mod.INDICATOR_COLUMNS[-len(_XIV_COLUMNS):]
    assert tail == _XIV_COLUMNS


def test_xiv_columns_present_in_all_columns() -> None:
    """XIV 5 指標 (score/raw 計 10 列) が ALL_COLUMNS に含まれること。"""
    for col in _XIV_COLUMNS:
        assert col in collect_mod.ALL_COLUMNS, f"{col} が ALL_COLUMNS に無い"


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


# ============================
# _fill_indicator_columns が XIV (近未来最大火力) 列を 0-1 範囲で埋めること
# ============================


@pytest.mark.parametrize("board", [_empty_board(), _chain_ready_board()])
def test_fill_indicator_columns_xiv_score_in_range(board: Board) -> None:
    """XIV (near_future_fire_k1..k5) の score が 0-1 範囲・NaN なしで埋まること。"""
    row = _call_fill(board)
    for score_col in _XIV_COLUMNS[0::2]:  # score 列のみ (偶数 index)
        val = row[score_col]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0, f"{score_col}={val} が 0-1 範囲外"
        assert val == val, f"{score_col} が NaN"


@pytest.mark.parametrize("board", [_empty_board(), _chain_ready_board()])
def test_fill_indicator_columns_xiv_raw_present(board: Board) -> None:
    """XIV 5 指標の raw が非負の数値で埋まること (NaN なし)。"""
    row = _call_fill(board)
    for raw_col in _XIV_COLUMNS[1::2]:  # raw 列のみ (奇数 index)
        val = row[raw_col]
        assert isinstance(val, float)
        assert val >= 0.0, f"{raw_col}={val} が負値"
        assert val == val, f"{raw_col} が NaN"


def test_fill_indicator_columns_xiv_k_monotonic_non_decreasing() -> None:
    """K を増やすほど near_future_fire_k{K}_raw が単調非減少であること。"""
    row = _call_fill(_chain_ready_board())
    raws = [row[f"near_future_fire_k{k}_raw"] for k in range(1, 6)]
    for prev, cur in zip(raws, raws[1:]):
        assert cur >= prev


def test_fill_indicator_columns_uses_next_pair_for_near_future() -> None:
    """next_pair/dnext_pair を渡すと near_future 側にも伝播すること (フォールバック無し確認)。"""
    row: dict[str, object] = {}
    board = _chain_ready_board()
    total_conn, _ = iv.connectivity_observation(board)
    collect_mod._fill_indicator_columns(
        row, board, tsumo=10, elapsed_sec=30.0, net=0, forecast=0,
        total_conn=total_conn,
        next_pair=(COLOR_RED, COLOR_BLUE), dnext_pair=(COLOR_RED, COLOR_BLUE),
    )
    for col in _XIV_COLUMNS:
        assert col in row


# ============================
# _GameColorTracker (試合単位 active_colors、2026-07-22 stateless修正)
# ============================
# コーディネータ指示: near_future_fire_power は stateless 純関数のまま維持し、
# 試合単位の色頻度計算 (プロトの _compute_active_colors_by_game と同じ
# 「頻度上位4色採用」ロジック) は収集パイプライン側の外部トラッカーが担う。


def test_game_color_tracker_returns_none_when_insufficient_data() -> None:
    """観測色数が GAME_COLOR_MIN_DISTINCT (4) 未満なら None を返すこと (フォールバック委譲)。"""
    tracker = collect_mod._GameColorTracker()
    tracker.update(_chain_ready_board())  # 赤・青の2色のみ含む盤面
    assert tracker.active_colors() is None


def test_game_color_tracker_returns_top4_when_sufficient_data() -> None:
    """5色全てが観測されても、出現頻度下位1色を除いた4色を返すこと。

    色1(赤)は1セルのみ (最少 = 除外される想定)、他4色は複数セルずつ配置し、
    「出現した5色全てのうち上位4色を採用」することを確認する。
    """
    g = _empty_grid()
    g[12][0] = COLOR_RED  # 最少 (1セルのみ) → 除外される想定
    g[12][1] = COLOR_BLUE
    g[11][1] = COLOR_BLUE
    g[12][2] = COLOR_GREEN
    g[11][2] = COLOR_GREEN
    g[12][3] = COLOR_YELLOW
    g[11][3] = COLOR_YELLOW
    g[12][4] = COLOR_PURPLE
    g[11][4] = COLOR_PURPLE
    board = Board.from_list(g)

    tracker = collect_mod._GameColorTracker()
    tracker.update(board)
    active = tracker.active_colors()
    assert active is not None
    assert len(active) == 4
    assert COLOR_RED not in active  # 最少観測色 (1セルのみ) は除外される


def test_game_color_tracker_reset_clears_counts() -> None:
    """reset() で累積カウントがクリアされること (試合境界での再初期化)。"""
    tracker = collect_mod._GameColorTracker()
    tracker.update(_chain_ready_board())
    assert tracker.counts  # 何かしら累積されている
    tracker.reset()
    assert tracker.counts == {}


def test_update_game_idx_resets_color_tracker_on_boundary() -> None:
    """score 大幅減少 (試合境界) で _SideTracker.color_tracker も reset されること。"""
    tracker = collect_mod._SideTracker()
    tracker.color_tracker.update(_chain_ready_board())
    assert tracker.color_tracker.counts
    collect_mod._update_game_idx(tracker, score=100)  # prev_score未設定 (初回)
    collect_mod._update_game_idx(tracker, score=100 - collect_mod.SCORE_RESET_THRESHOLD - 1)
    assert tracker.game_idx == 1
    assert tracker.color_tracker.counts == {}


def test_fill_indicator_columns_active_colors_propagates_to_near_future() -> None:
    """active_colors を渡すと near_future_fire_k* の値が変化しうること (伝播確認)。"""
    board = _chain_ready_board()
    total_conn, _ = iv.connectivity_observation(board)

    row_default: dict[str, object] = {}
    collect_mod._fill_indicator_columns(
        row_default, board, tsumo=10, elapsed_sec=0.0, net=0, forecast=0,
        total_conn=total_conn,
    )
    row_restricted: dict[str, object] = {}
    collect_mod._fill_indicator_columns(
        row_restricted, board, tsumo=10, elapsed_sec=0.0, net=0, forecast=0,
        total_conn=total_conn,
        active_colors=(COLOR_GREEN, COLOR_YELLOW),
    )
    diffs = [
        row_default[f"near_future_fire_k{k}_raw"] != row_restricted[f"near_future_fire_k{k}_raw"]
        for k in range(1, 6)
    ]
    assert any(diffs), "active_colors 指定が near_future_fire_k* に反映されていない"
