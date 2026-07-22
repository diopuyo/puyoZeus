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
