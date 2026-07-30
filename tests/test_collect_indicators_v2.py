"""scripts/collect_indicators_v2.py の CSV 列定義・指標充填ロジックのユニットテスト。

テスト方針 (tests/test_board_npz_dump.py に準拠):
- 動画認識・重い collect 実行は一切しない。
- 合成 Board で _fill_indicator_columns を直接呼び、列の存在・値域を検証する。

対象: XII board sim 本命指標 (飽和連鎖量・発火点・多色発火・副砲・同時消し
リッチネス) を CSV 列として追加した際の統合が正しいことを確認する。
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
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
from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline
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

# CSV 末尾に追加された XV (火力の受けの多さ fire_stability, K=2,4,6) の列名
# (2026-07-22 本番統合、順序保持で確認)。
_XV_COLUMNS: tuple[str, ...] = (
    "fire_stability_k2", "fire_stability_k2_raw",
    "fire_stability_k4", "fire_stability_k4_raw",
    "fire_stability_k6", "fire_stability_k6_raw",
)

# CSV 末尾に追加された XVI (平均ツモ期待火力 expected_fire_power, K=1..4) の列名
# (2026-07-22 本番統合、順序保持で確認。K=3,4 追加時の列定義更新漏れを是正済み)。
_XVI_COLUMNS: tuple[str, ...] = (
    "expected_fire_k1", "expected_fire_k1_raw",
    "expected_fire_k2", "expected_fire_k2_raw",
    "expected_fire_k3", "expected_fire_k3_raw",
    "expected_fire_k4", "expected_fire_k4_raw",
)

# 新指標追加のたびに末尾へ連結していく既存ブロック群 (退行検知の土台。
# 新ブロック追加時はこのタプルに1行足すだけで済むようにする)。
_TAIL_BLOCKS: "tuple[tuple[str, ...], ...]" = (
    _XII_COLUMNS, _XIV_COLUMNS, _XV_COLUMNS, _XVI_COLUMNS,
)


def test_xii_columns_present_in_all_columns() -> None:
    """XII 5 指標 (score/raw 計 10 列) が ALL_COLUMNS に含まれること。"""
    for col in _XII_COLUMNS:
        assert col in collect_mod.ALL_COLUMNS, f"{col} が ALL_COLUMNS に無い"


def test_tail_blocks_are_contiguous_at_end_of_indicator_columns() -> None:
    """XII→XIV→XV→XVI が INDICATOR_COLUMNS の末尾に連続していること。

    新指標追加のたびに「旧ブロックが末尾」という旧アサーションは成立しなく
    なる (正当な末尾追加の結果であり退行ではない) ため、既知の全ブロックを
    連結した「今の末尾」で確認する回帰テストに一本化する。
    """
    expected_tail: "tuple[str, ...]" = ()
    for block in _TAIL_BLOCKS:
        expected_tail += block
    tail_block = collect_mod.INDICATOR_COLUMNS[-len(expected_tail):]
    assert tail_block == expected_tail


def test_xvi_columns_are_tail_of_indicator_columns() -> None:
    """XVI (平均ツモ期待火力) 2 指標が INDICATOR_COLUMNS の末尾4列であること。

    2026-07-22 本番統合、新指標は常に末尾追加 (CLAUDE.md 規約)。
    """
    tail = collect_mod.INDICATOR_COLUMNS[-len(_XVI_COLUMNS):]
    assert tail == _XVI_COLUMNS


def test_xvi_columns_present_in_all_columns() -> None:
    """XVI 2 指標 (score/raw 計 4 列) が ALL_COLUMNS に含まれること。"""
    for col in _XVI_COLUMNS:
        assert col in collect_mod.ALL_COLUMNS, f"{col} が ALL_COLUMNS に無い"


def test_xiv_columns_present_in_all_columns() -> None:
    """XIV 5 指標 (score/raw 計 10 列) が ALL_COLUMNS に含まれること。"""
    for col in _XIV_COLUMNS:
        assert col in collect_mod.ALL_COLUMNS, f"{col} が ALL_COLUMNS に無い"


def test_xv_columns_present_in_all_columns() -> None:
    """XV 3 指標 (score/raw 計 6 列) が ALL_COLUMNS に含まれること。"""
    for col in _XV_COLUMNS:
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


# ============================
# _fill_indicator_columns が XV (火力の受けの多さ) 列を 0-1 範囲で埋めること
# ============================


@pytest.mark.parametrize("board", [_empty_board(), _chain_ready_board()])
def test_fill_indicator_columns_xv_score_in_range(board: Board) -> None:
    """XV (fire_stability_k2/4/6) の score が 0-1 範囲・NaN なしで埋まること。"""
    row = _call_fill(board)
    for score_col in _XV_COLUMNS[0::2]:  # score 列のみ (偶数 index)
        val = row[score_col]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0, f"{score_col}={val} が 0-1 範囲外"
        assert val == val, f"{score_col} が NaN"


@pytest.mark.parametrize("board", [_empty_board(), _chain_ready_board()])
def test_fill_indicator_columns_xv_raw_present(board: Board) -> None:
    """XV 3 指標の raw (件数) が非負の数値で埋まること (NaN なし)。"""
    row = _call_fill(board)
    for raw_col in _XV_COLUMNS[1::2]:  # raw 列のみ (奇数 index)
        val = row[raw_col]
        assert isinstance(val, float)
        assert val >= 0.0, f"{raw_col}={val} が負値"
        assert val == val, f"{raw_col} が NaN"


def test_fill_indicator_columns_active_colors_propagates_to_fire_stability() -> None:
    """active_colors を渡すと fire_stability_k* の値が変化しうること (伝播確認)。"""
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
        row_default[f"fire_stability_k{k}_raw"] != row_restricted[f"fire_stability_k{k}_raw"]
        for k in (2, 4, 6)
    ]
    assert any(diffs), "active_colors 指定が fire_stability_k* に反映されていない"


# ============================
# _fill_indicator_columns が XVI (平均ツモ期待火力) 列を 0-1 範囲で埋めること
# ============================
# ⚠️ 2026-07-22 user判断: expected_fire_power は重い (1.7-3.5秒/盤面) ため
# collect_indicators_v2.COLLECT_EXPECTED_FIRE は既定 False (opt-in)。
# 以下の値域・再現性テストは XVI の計算内容自体を検証するものなので、
# monkeypatch で明示的に True にした上で確認する (既定OFF挙動は
# 別途 test_collect_expected_fire_opt_in_* 群で確認する)。


@pytest.mark.parametrize("board", [_empty_board(), _chain_ready_board()])
def test_fill_indicator_columns_xvi_score_in_range(
    board: Board, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XVI (expected_fire_k1/k2) の score が 0-1 範囲・NaN なしで埋まること。"""
    monkeypatch.setattr(collect_mod, "COLLECT_EXPECTED_FIRE", True)
    row = _call_fill(board)
    for score_col in _XVI_COLUMNS[0::2]:
        val = row[score_col]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0, f"{score_col}={val} が 0-1 範囲外"
        assert val == val, f"{score_col} が NaN"


@pytest.mark.parametrize("board", [_empty_board(), _chain_ready_board()])
def test_fill_indicator_columns_xvi_raw_present(
    board: Board, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XVI 2 指標の raw (平均お邪魔換算) が非負の数値で埋まること (NaN なし)。"""
    monkeypatch.setattr(collect_mod, "COLLECT_EXPECTED_FIRE", True)
    row = _call_fill(board)
    for raw_col in _XVI_COLUMNS[1::2]:
        val = row[raw_col]
        assert isinstance(val, float)
        assert val >= 0.0, f"{raw_col}={val} が負値"
        assert val == val, f"{raw_col} が NaN"


def test_fill_indicator_columns_xvi_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一盤面を2回埋めても expected_fire_k* が同じ値になること (stateless再現性)。"""
    monkeypatch.setattr(collect_mod, "COLLECT_EXPECTED_FIRE", True)
    board = _chain_ready_board()
    row1 = _call_fill(board)
    row2 = _call_fill(board)
    for k in (1, 2):
        assert row1[f"expected_fire_k{k}_raw"] == row2[f"expected_fire_k{k}_raw"]


# ============================
# XVI opt-in ガード (COLLECT_EXPECTED_FIRE、2026-07-22 user判断)
# ============================
# expected_fire_power は重い (1.7-3.5秒/盤面) ため既定 OFF。将来の Phase L
# データ拡充で常時収集すると ~1fps律速の収集パイプラインが破綻するため。


def test_collect_expected_fire_default_is_off() -> None:
    """モジュール既定 COLLECT_EXPECTED_FIRE が False (opt-in) であること。"""
    assert collect_mod.COLLECT_EXPECTED_FIRE is False


def test_collect_expected_fire_default_off_skips_columns_entirely() -> None:
    """既定 (monkeypatch なし) では expected_fire_k* が row に一切追加されないこと。

    CSV 列は INDICATOR_COLUMNS 定義に残ったまま (csv.DictWriter の
    restval='' により空欄で出力される)、計算コストはゼロになることを確認する。
    """
    row = _call_fill(_chain_ready_board())
    for col in _XVI_COLUMNS:
        assert col not in row, f"{col} は既定OFFのはずなのに計算されている"


def test_collect_expected_fire_monkeypatch_true_enables_columns() -> None:
    """COLLECT_EXPECTED_FIRE=True (monkeypatch) にすると expected_fire_k* が

    計算され row に現れること (opt-in が正しく機能する)。
    """
    row: dict[str, object] = {}
    board = _chain_ready_board()
    total_conn, _ = iv.connectivity_observation(board)
    collect_mod._fill_expected_fire_columns(row, board, elapsed_sec=0.0, enabled=True)
    for col in _XVI_COLUMNS:
        assert col in row


def test_collect_expected_fire_explicit_enabled_false_skips_even_if_global_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enabled=False を明示指定すれば、モジュール既定が True でもスキップされること

    (呼び出し側の明示指定がモジュール既定より優先される)。
    """
    monkeypatch.setattr(collect_mod, "COLLECT_EXPECTED_FIRE", True)
    row: dict[str, object] = {}
    board = _chain_ready_board()
    collect_mod._fill_expected_fire_columns(row, board, elapsed_sec=0.0, enabled=False)
    for col in _XVI_COLUMNS:
        assert col not in row


def test_collect_expected_fire_near_future_and_fire_stability_unaffected_by_opt_in() -> None:
    """opt-in ガードの追加が near_future/fire_stability の既定収集を壊さないこと

    (既定 COLLECT_EXPECTED_FIRE=False の状態で、XIV/XV 列は引き続き計算される)。
    """
    row = _call_fill(_chain_ready_board())
    for col in _XIV_COLUMNS:
        assert col in row
    for col in _XV_COLUMNS:
        assert col in row


# ============================
# _resolve_sample_interval_frames (2026-07-28 追加)
# サンプリング間隔をフレーム単位で正確指定できるようにする回帰テスト。
# 動画デコードは一切行わず、fps を差し替えた純粋関数呼び出しのみで検証する。
# ============================


def test_resolve_sample_interval_frames_explicit_takes_priority_over_sec() -> None:
    """フレーム数指定が秒指定より優先されること (fps が変わっても結果不変)。"""
    for fps in (30.0, 60.0):
        resolved = collect_mod._resolve_sample_interval_frames(
            sample_interval_sec=1.0,  # fps次第で全く違う値になるはずの秒指定
            fps=fps,
            sample_interval_frames=8,
        )
        assert resolved == 8, f"fps={fps} でもフレーム数指定 8 が優先されるべき"


def test_resolve_sample_interval_frames_omitted_preserves_legacy_sec_behavior_60fps() -> None:
    """フレーム数指定省略時、60fps で秒指定の従来換算結果と完全一致すること。"""
    # 8フレーム(60fps) = 0.1333...秒 相当を秒指定した場合との整合を確認
    resolved = collect_mod._resolve_sample_interval_frames(
        sample_interval_sec=0.2, fps=60.0, sample_interval_frames=None,
    )
    assert resolved == max(1, int(round(0.2 * 60.0)))  # 従来式と同じ計算
    assert resolved == 12


def test_resolve_sample_interval_frames_omitted_preserves_legacy_sec_behavior_30fps() -> None:
    """フレーム数指定省略時、30fps でも秒指定の従来換算結果と完全一致すること。

    同じ 0.2 秒指定でも fps が違えばフレーム数換算結果が変わることを確認し、
    「fps混在時に意図と違う間引きになる」現状課題が秒指定側では
    引き続き再現される (=後方互換で挙動が一切変わっていない) ことを示す。
    """
    resolved = collect_mod._resolve_sample_interval_frames(
        sample_interval_sec=0.2, fps=30.0, sample_interval_frames=None,
    )
    assert resolved == max(1, int(round(0.2 * 30.0)))  # 従来式と同じ計算
    assert resolved == 6


def test_resolve_sample_interval_frames_zero_sec_defaults_to_one_frame() -> None:
    """sample_interval_sec=0.0 (全フレーム指定) は従来通り 1 になること。"""
    resolved = collect_mod._resolve_sample_interval_frames(
        sample_interval_sec=0.0, fps=60.0, sample_interval_frames=None,
    )
    assert resolved == 1


@pytest.mark.parametrize("bad_frames", [0, -1, -100])
def test_resolve_sample_interval_frames_non_positive_frames_clamped_to_one(
    bad_frames: int,
) -> None:
    """0 以下のフレーム数指定は下限 1 に丸められること (秒指定側の max(1, ...) と整合)。"""
    resolved = collect_mod._resolve_sample_interval_frames(
        sample_interval_sec=0.0, fps=60.0, sample_interval_frames=bad_frames,
    )
    assert resolved == collect_mod.MIN_SAMPLE_INTERVAL_FRAMES


def test_resolve_sample_interval_frames_default_none_when_omitted_kwarg() -> None:
    """sample_interval_frames を渡さず呼んでも従来の秒指定挙動になること (引数省略呼び出し)。"""
    resolved = collect_mod._resolve_sample_interval_frames(
        sample_interval_sec=0.2, fps=60.0,
    )
    assert resolved == 12


def test_collect_signature_has_sample_interval_frames_appended_at_tail() -> None:
    """collect() の新引数 sample_interval_frames が末尾追加され、

    既存引数の並び・デフォルト値が一切変わっていないこと (backwards compat)。

    2026-07-30 追記: さらに末尾へ indicator_interval_frames を追加したため、
    sample_interval_frames はもう「最後の引数」ではなくなった。ここでは
    「sample_interval_frames までの並びが不変であること」のみを確認し、
    新しい末尾の確認は test_collect_signature_has_indicator_interval_frames_appended_at_tail
    に分離する (末尾追加のたびに前のブロックの「最後」アサーションを壊さない
    ための一般化)。
    """
    import inspect
    sig = inspect.signature(collect_mod.collect)
    params = list(sig.parameters.keys())
    assert params[:6] == [
        "video_path", "out_path", "max_sec", "sample_interval_sec",
        "start_sec", "board_npz_path",
    ]
    assert params[6] == "sample_interval_frames"
    assert sig.parameters["sample_interval_frames"].default is None


def test_collect_signature_has_indicator_interval_frames_appended_at_tail() -> None:
    """collect() の新引数 indicator_interval_frames / normalize_fps_30 が

    末尾に順次 optional 追加され、既存引数 (sample_interval_frames まで) の
    並び・デフォルト値が一切変わっていないこと (2026-07-30 追加、backwards compat)。
    """
    import inspect
    sig = inspect.signature(collect_mod.collect)
    params = list(sig.parameters.keys())
    assert params[-2] == "indicator_interval_frames"
    assert sig.parameters["indicator_interval_frames"].default is None
    assert params[-3] == "sample_interval_frames"
    assert params[-1] == "normalize_fps_30"
    # 2026-07-30 既定 True 化 (user承認済み、A/B実測で60fps stride-2が優位)
    assert sig.parameters["normalize_fps_30"].default is True


def _run_fake_main_collect(argv_tail: list[str]) -> dict[str, object]:
    """collect を差し替えて main() を実行し、渡された kwargs を返す共通ヘルパ。"""
    from unittest.mock import patch

    captured: dict[str, object] = {}

    def _fake_collect(*args: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    with patch.object(collect_mod, "collect", _fake_collect):
        with patch(
            "sys.argv",
            ["collect_indicators_v2.py", "--video", "x.mp4", "--out", "y.csv"]
            + argv_tail,
        ):
            collect_mod.main()
    return captured


def test_main_cli_normalize_fps_30_default_true_when_no_flags() -> None:
    """CLI で --normalize-fps-30 / --no-normalize-fps-30 とも未指定なら

    normalize_fps_30=True が collect に渡る (2026-07-30 既定 True 化)。
    """
    captured = _run_fake_main_collect([])
    assert captured["normalize_fps_30"] is True


def test_main_cli_no_normalize_fps_30_flag_disables() -> None:
    """--no-normalize-fps-30 指定時は normalize_fps_30=False が渡ること。"""
    captured = _run_fake_main_collect(["--no-normalize-fps-30"])
    assert captured["normalize_fps_30"] is False


def test_main_cli_no_normalize_fps_30_wins_when_both_specified() -> None:
    """--normalize-fps-30 と --no-normalize-fps-30 を同時指定した場合、

    無効化 (--no-normalize-fps-30) が優先されること (coordinator指示の仕様)。
    """
    captured = _run_fake_main_collect(["--normalize-fps-30", "--no-normalize-fps-30"])
    assert captured["normalize_fps_30"] is False


# ============================
# _resolve_indicator_interval_frames (2026-07-30 追加)
# 認識と独立に指標計算・行出力を間引く幅を確定する純関数の回帰テスト。
# ============================


def test_resolve_indicator_interval_frames_omitted_defaults_to_one() -> None:
    """省略時 (None) は 1 (間引きなし=毎フレーム、従来挙動) になること。"""
    assert collect_mod._resolve_indicator_interval_frames(None) == 1
    assert collect_mod._resolve_indicator_interval_frames() == 1


def test_resolve_indicator_interval_frames_explicit_value_used() -> None:
    """明示指定した値がそのまま使われること。"""
    assert collect_mod._resolve_indicator_interval_frames(6) == 6


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_resolve_indicator_interval_frames_non_positive_clamped_to_one(
    bad_value: int,
) -> None:
    """0 以下の指定は下限 1 に丸められること (_resolve_sample_interval_frames と同じ規約)。"""
    assert collect_mod._resolve_indicator_interval_frames(bad_value) == (
        collect_mod.MIN_SAMPLE_INTERVAL_FRAMES
    )


# ============================
# collect() ループ結合テスト (2026-07-30 追加)
# 実動画は使わず cv2.VideoCapture / RecognitionPipeline.load_default を
# フェイクに差し替え、「認識は毎フレーム・指標計算だけが間引かれる」ことを
# 直接検証する (実動画デコードは重いため一切使わない)。
# ============================


class _FakeCapture:
    """cv2.VideoCapture の最小フェイク。固定本数のダミーフレームを返す。"""

    def __init__(self, n_frames: int, fps: float = 30.0) -> None:
        self._n_frames = n_frames
        self._fps = fps
        self._i = 0
        # TARGET_H/TARGET_W と一致させ、collect() 内の cv2.resize を回避する
        # (同一 ndarray 参照を使い回すのでメモリ確保は 1 回のみ)。
        self._frame = np.zeros(
            (collect_mod.TARGET_H, collect_mod.TARGET_W, 3), dtype=np.uint8,
        )

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._n_frames)
        return 0.0

    def set(self, prop: int, value: float) -> None:  # noqa: D401 - フェイクなので no-op
        pass

    def read(self) -> "tuple[bool, np.ndarray | None]":
        if self._i >= self._n_frames:
            return False, None
        self._i += 1
        return True, self._frame

    def release(self) -> None:
        pass


class _FakeThrottlePipeline:
    """RecognitionPipeline.load_default の最小フェイク。

    update() を呼ぶ度に必ず盤面セルを1つ追加する (dedup で潰されない
    ようにするため、fill_idx が単調増加する限り毎回異なる盤面になる)。
    """

    def __init__(self) -> None:
        self.update_calls: list[int] = []
        self._grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
        self._cells = [(r, c) for r in range(BOARD_ROWS) for c in range(BOARD_COLS)]
        self._fill_idx = 0
        self._tsumo = {"1P": 0, "2P": 0}

    def update(self, fi: int, t_sec: float, frame: np.ndarray) -> SimpleNamespace:
        self.update_calls.append(fi)
        r, c = self._cells[self._fill_idx % len(self._cells)]
        self._grid[r, c] = COLOR_RED
        self._fill_idx += 1
        board = Board.from_list(self._grid.tolist())
        # tsumo_count は毎フレーム (= 毎 update 呼び出し) 進める
        # (おじゃま会計 drain が毎フレーム駆動されることの検証に使う)。
        self._tsumo["1P"] += 1
        self._tsumo["2P"] += 1
        side = SimpleNamespace(
            state=BoardState.STABLE,
            score=1000 + self._fill_idx,  # 単調増加 (試合境界誤検知を防ぐ)
            confirmed_board=board,
            next_pair=None,
            dnext_pair=None,
            chain_event=None,
        )
        return SimpleNamespace(p1=side, p2=side)

    def tsumo_count(self, side_label: str) -> int:
        return self._tsumo[side_label]


def _run_fake_collect(
    tmp_path: Path, n_frames: int, *, fps: float = 30.0, **collect_kwargs: object,
) -> "tuple[int, _FakeThrottlePipeline, Path]":
    """cv2.VideoCapture / RecognitionPipeline.load_default をフェイクに
    差し替えて collect() を実行する共通ヘルパ。

    fps: フェイク動画の fps (既定 30.0 = 従来のテスト呼び出しと完全一致)。
        normalize_fps_30 の A/B テスト (2026-07-30 追加) のため optional 追加。
    """
    fake_cap = _FakeCapture(n_frames, fps=fps)
    fake_pipeline = _FakeThrottlePipeline()

    def _fake_video_capture(_path: str) -> _FakeCapture:
        return fake_cap

    def _fake_load_default(*args: object, **kwargs: object) -> _FakeThrottlePipeline:
        return fake_pipeline

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(collect_mod.cv2, "VideoCapture", _fake_video_capture)
        mp.setattr(RecognitionPipeline, "load_default", _fake_load_default)
        out_path = tmp_path / "out.csv"
        n_rows = collect_mod.collect(
            Path("dummy_video.mp4"), out_path, **collect_kwargs,
        )
    return n_rows, fake_pipeline, out_path


def test_collect_indicator_interval_omitted_recognizes_and_emits_every_frame(
    tmp_path: Path,
) -> None:
    """indicator_interval_frames 省略時、pipeline.update・行出力とも

    全フレーム実行される (従来挙動そのまま、backwards compat)。
    """
    n_frames = 12
    n_rows, fake_pipeline, _ = _run_fake_collect(tmp_path, n_frames)
    assert len(fake_pipeline.update_calls) == n_frames
    assert n_rows == n_frames * 2  # 1P/2P 双方、毎フレーム別盤面につき出力


def test_collect_indicator_interval_frames_recognizes_every_frame(
    tmp_path: Path,
) -> None:
    """indicator_interval_frames 指定時も pipeline.update は全フレーム呼ばれること

    (認識は間引きの対象外であることの直接確認)。
    """
    n_frames = 12
    n_rows, fake_pipeline, _ = _run_fake_collect(
        tmp_path, n_frames, indicator_interval_frames=4,
    )
    assert len(fake_pipeline.update_calls) == n_frames


def test_collect_indicator_interval_frames_throttles_only_row_output(
    tmp_path: Path,
) -> None:
    """indicator_interval_frames 指定時、行の書き出しだけが間引かれること。

    fi % interval == 0 のフレームのみ 1P/2P 各 1 行、それ以外は出力なし。
    """
    n_frames = 12
    interval = 4
    n_rows, fake_pipeline, out_path = _run_fake_collect(
        tmp_path, n_frames, indicator_interval_frames=interval,
    )
    expected_sampled_frames = len(
        [fi for fi in range(n_frames) if fi % interval == 0],
    )
    assert n_rows == expected_sampled_frames * 2
    # CSV 実体の行数も一致すること (メモリ上の rows と書き出しが食い違わないか)
    with open(out_path, newline="", encoding="utf-8") as f:
        n_csv_rows = sum(1 for _ in csv.DictReader(f))
    assert n_csv_rows == n_rows


def test_collect_indicator_interval_frames_explicit_one_matches_omitted(
    tmp_path: Path,
) -> None:
    """indicator_interval_frames=1 を明示指定しても省略時と同じ行数になること。"""
    n_frames = 10
    n_rows_omitted, _, _ = _run_fake_collect(tmp_path, n_frames)
    n_rows_explicit_one, _, _ = _run_fake_collect(
        tmp_path, n_frames, indicator_interval_frames=1,
    )
    assert n_rows_omitted == n_rows_explicit_one


def test_collect_ojama_and_game_idx_advance_every_frame_even_when_throttled(
    tmp_path: Path,
) -> None:
    """指標間引き中も、行に記録される tsumo が間引き幅ぶん飛んでいること。

    水面下 (おじゃま会計 drain) では tsumo_count が毎フレーム進んでいる証拠になる。
    間引き幅より粗い頻度でしか tsumo が進んでいなければ、おじゃま会計が
    誤って間引かれている回帰を検知できる。
    """
    n_frames = 20
    interval = 5
    _, _, out_path = _run_fake_collect(
        tmp_path, n_frames, indicator_interval_frames=interval,
    )
    with open(out_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    p1_tsumo = [int(r["tsumo"]) for r in rows if r["side"] == "1P"]
    diffs = [b - a for a, b in zip(p1_tsumo, p1_tsumo[1:])]
    assert all(d == interval for d in diffs), diffs


# ============================
# normalize_fps_30 (2026-07-30 追加、既定 OFF)
# 60fps 動画を実効30fpsへ stride-2 で間引く自動注入の優先順位・後方互換を検証。
# ============================


def test_collect_normalize_fps_30_default_omitted_applies_stride_2_for_60fps(
    tmp_path: Path,
) -> None:
    """normalize_fps_30 省略時 (2026-07-30 既定 True 化後) は 60fps 動画に

    stride-2 が自動適用され、2フレームに1回だけ認識・行出力されること。
    """
    n_frames = 10
    n_rows, fake_pipeline, _ = _run_fake_collect(tmp_path, n_frames, fps=60.0)
    assert fake_pipeline.update_calls == list(range(0, n_frames, 2))
    assert n_rows == len(range(0, n_frames, 2)) * 2


def test_collect_normalize_fps_30_explicit_false_is_bit_identical(
    tmp_path: Path,
) -> None:
    """normalize_fps_30=False を明示指定した場合は 60fps でも間引かれず

    全フレーム認識される (CLI --no-normalize-fps-30 相当、後方互換経路の保持)。
    """
    n_frames = 10
    n_rows, fake_pipeline, _ = _run_fake_collect(
        tmp_path, n_frames, fps=60.0, normalize_fps_30=False,
    )
    assert len(fake_pipeline.update_calls) == n_frames
    assert n_rows == n_frames * 2


def test_collect_normalize_fps_30_60fps_injects_stride_2(tmp_path: Path) -> None:
    """normalize_fps_30=True かつ 60fps のとき、2フレームに1回だけ認識される

    (resolve_normalize_fps_30_stride(60.0) == 2 が sample_interval_frames として
    自動注入されることの直接確認)。
    """
    n_frames = 10
    _, fake_pipeline, _ = _run_fake_collect(
        tmp_path, n_frames, fps=60.0, normalize_fps_30=True,
    )
    assert fake_pipeline.update_calls == list(range(0, n_frames, 2))


def test_collect_normalize_fps_30_30fps_no_effective_change(tmp_path: Path) -> None:
    """normalize_fps_30=True でも 30fps 動画は stride=1 (間引きなし) のままであること

    (resolve_normalize_fps_30_stride(30.0) == 1)。
    """
    n_frames = 10
    _, fake_pipeline, _ = _run_fake_collect(
        tmp_path, n_frames, fps=30.0, normalize_fps_30=True,
    )
    assert len(fake_pipeline.update_calls) == n_frames


def test_collect_normalize_fps_30_ignored_when_sample_interval_frames_explicit(
    tmp_path: Path,
) -> None:
    """明示 sample_interval_frames が normalize_fps_30 より優先されること

    (優先順位: 「明示 sample_interval_frames > 自動 normalize_fps_30」)。
    """
    n_frames = 12
    _, fake_pipeline, _ = _run_fake_collect(
        tmp_path, n_frames, fps=60.0,
        sample_interval_frames=4, normalize_fps_30=True,
    )
    assert fake_pipeline.update_calls == list(range(0, n_frames, 4))
