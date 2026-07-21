"""proto_net_threat_v2.py のユニットテスト。

重い動画認識は使わず、合成盤面 + NpzRecord 相当データで
配置列挙・フォールバック判定・NpzRecord 拡張の後方互換のみを検証する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_RED,
    Board,
)
from src.chain import ChainSimulator


def _import_v2():
    """proto_net_threat_v2 モジュールをインポートして返す。"""
    import scripts.proto_net_threat_v2 as mod
    return mod


def _make_board(color: int = COLOR_RED, row_count: int = 2) -> Board:
    """下段 row_count 行を color で埋めた合成盤面を返す。"""
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for col in range(BOARD_COLS):
        for row_offset in range(row_count):
            g[BOARD_ROWS - 1 - row_offset][col] = color
    return Board.from_list(g)


def _make_empty_board() -> Board:
    """全セルが EMPTY (0) の盤面を返す。"""
    return Board.from_list([[0] * BOARD_COLS for _ in range(BOARD_ROWS)])


# ============================
# _is_valid_next_pair
# ============================

class TestIsValidNextPair:
    """_is_valid_next_pair のフォールバック判定を検証する。"""

    def test_none_is_invalid(self) -> None:
        mod = _import_v2()
        assert mod._is_valid_next_pair(None) is False

    def test_valid_colors_1_to_5(self) -> None:
        mod = _import_v2()
        for a in range(1, 6):
            for b in range(1, 6):
                assert mod._is_valid_next_pair((a, b)) is True

    def test_unknown_sentinel_minus1_invalid(self) -> None:
        mod = _import_v2()
        assert mod._is_valid_next_pair((-1, -1)) is False

    def test_ojama_color_9_invalid(self) -> None:
        """NextDetector 誤検出で観測された color=9 (おじゃま) は無効扱い。"""
        mod = _import_v2()
        assert mod._is_valid_next_pair((9, 1)) is False

    def test_partial_invalid_rejects_whole_pair(self) -> None:
        mod = _import_v2()
        assert mod._is_valid_next_pair((1, -1)) is False


# ============================
# _next_pair_status (opp_next_known 内訳集計用)
# ============================

class TestNextPairStatus:
    """_next_pair_status の absent/misdetect/valid 分類を検証する。"""

    def test_valid_pair(self) -> None:
        mod = _import_v2()
        assert mod._next_pair_status((1, 2)) == mod.NEXT_STATUS_VALID

    def test_absent_pair_both_sentinel(self) -> None:
        """両方 -1 (未検出/未収集) は absent 扱い。"""
        mod = _import_v2()
        assert mod._next_pair_status((-1, -1)) == mod.NEXT_STATUS_ABSENT

    def test_misdetect_pair_ojama_color(self) -> None:
        """色9等 (-1 でも 1-5 でもない) は misdetect 扱い。"""
        mod = _import_v2()
        assert mod._next_pair_status((9, 1)) == mod.NEXT_STATUS_MISDETECT


# ============================
# _enumerate_pair_placements
# ============================

class TestEnumeratePairPlacements:
    """22通り配置列挙のロジックを検証する。"""

    def test_empty_board_yields_22_placements(self) -> None:
        """空盤面なら縦12+横10=22通り全てが有効。"""
        mod = _import_v2()
        board = _make_empty_board()
        placements = mod._enumerate_pair_placements(board, COLOR_RED, COLOR_BLUE)
        assert len(placements) == 22

    def test_vertical_placements_count(self) -> None:
        """縦置きは6列×2色順=12通り。"""
        mod = _import_v2()
        board = _make_empty_board()
        out = mod._vertical_placements(board, COLOR_RED, COLOR_BLUE)
        assert len(out) == 12

    def test_horizontal_placements_count(self) -> None:
        """横置きは5組×2色順=10通り。"""
        mod = _import_v2()
        board = _make_empty_board()
        out = mod._horizontal_placements(board, COLOR_RED, COLOR_BLUE)
        assert len(out) == 10

    def test_full_column_excluded_from_vertical(self) -> None:
        """列が満杯 (残り1段以下) なら縦置き候補から除外される。"""
        mod = _import_v2()
        # 全列 BOARD_ROWS-1 段まで埋める (残り1段のみ、縦2段は積めない)
        board = _make_board(COLOR_RED, row_count=BOARD_ROWS - 1)
        out = mod._vertical_placements(board, COLOR_RED, COLOR_BLUE)
        assert len(out) == 0

    def test_placement_does_not_mutate_original_board(self) -> None:
        """配置列挙が元盤面を破壊しないこと (stateless 原則)。"""
        mod = _import_v2()
        board = _make_empty_board()
        original_bytes = board._grid.tobytes()
        mod._enumerate_pair_placements(board, COLOR_RED, COLOR_BLUE)
        assert board._grid.tobytes() == original_bytes

    def test_full_board_yields_zero_placements(self) -> None:
        """全列満杯なら配置候補が0件になること。"""
        mod = _import_v2()
        board = _make_board(COLOR_RED, row_count=BOARD_ROWS)
        placements = mod._enumerate_pair_placements(board, COLOR_RED, COLOR_BLUE)
        assert len(placements) == 0


# ============================
# _predicted_counter_ojama_v2 フォールバック
# ============================

class TestPredictedCounterOjamaV2:
    """本命版のフォールバック挙動を検証する。"""

    def test_invalid_next_pair_falls_back_and_flags_false(self) -> None:
        """next_pair 無効時は used_real_next=False で v1 相当のフォールバックになる。"""
        mod = _import_v2()
        sim = ChainSimulator()
        board = _make_board(COLOR_RED, row_count=3)
        raw, used_real = mod._predicted_counter_ojama_v2(
            board, 0.0, k_hands=2, next_pair=None, dnext_pair=None, sim=sim,
        )
        assert used_real is False
        assert raw >= 0.0

    def test_valid_next_pair_flags_true(self) -> None:
        """next_pair 有効時は used_real_next=True になる。"""
        mod = _import_v2()
        sim = ChainSimulator()
        board = _make_board(COLOR_RED, row_count=3)
        raw, used_real = mod._predicted_counter_ojama_v2(
            board, 0.0, k_hands=1, next_pair=(COLOR_RED, COLOR_BLUE),
            dnext_pair=None, sim=sim,
        )
        assert used_real is True
        assert raw >= 0.0

    def test_full_board_returns_zero(self) -> None:
        """全列満杯盤面では相殺量0を返す (例外を出さない)。"""
        mod = _import_v2()
        sim = ChainSimulator()
        board = _make_board(COLOR_RED, row_count=BOARD_ROWS)
        raw, used_real = mod._predicted_counter_ojama_v2(
            board, 0.0, k_hands=2, next_pair=(COLOR_RED, COLOR_BLUE),
            dnext_pair=(COLOR_BLUE, COLOR_RED), sim=sim,
        )
        assert used_real is True
        assert raw == 0.0

    def test_ojama_color_next_pair_falls_back(self) -> None:
        """next1_a=9 (おじゃま、誤検出混入値) は無効なのでフォールバックすること。"""
        mod = _import_v2()
        sim = ChainSimulator()
        board = _make_board(COLOR_RED, row_count=3)
        raw, used_real = mod._predicted_counter_ojama_v2(
            board, 0.0, k_hands=1, next_pair=(9, COLOR_BLUE),
            dnext_pair=None, sim=sim,
        )
        assert used_real is False


# ============================
# _net_threat_v2
# ============================

class TestNetThreatV2:
    """net_threat_v2 の raw/norm 計算を検証する。"""

    def test_norm_clamped_0_1(self) -> None:
        mod = _import_v2()
        raw, norm = mod._net_threat_v2(ojama_sent=1000.0, predicted_counter=0.0)
        assert 0.0 <= norm <= 1.0
        raw2, norm2 = mod._net_threat_v2(ojama_sent=0.0, predicted_counter=1000.0)
        assert 0.0 <= norm2 <= 1.0

    def test_raw_is_difference(self) -> None:
        mod = _import_v2()
        raw, _ = mod._net_threat_v2(ojama_sent=30.0, predicted_counter=10.0)
        assert raw == 20.0


# ============================
# NpzRecord 拡張の後方互換
# ============================

class TestNpzRecordBackwardCompat:
    """NpzRecord への next 列追加が既存呼び出しを壊さないことを検証する。"""

    def test_positional_construction_without_next_fields(self) -> None:
        """次列を渡さない既存位置引数呼び出しがデフォルト値で動作すること。"""
        from scripts.label_exchange_outcome import NpzRecord
        rec = NpzRecord(
            "v29", "1P",
            np.array([1.0]), np.array([0]), np.zeros((1, BOARD_ROWS, BOARD_COLS)),
            np.array([1.0]), np.array([100]),
        )
        assert rec.next1_a.size == 0  # デフォルト空配列 (呼び出し側が使わない限り無害)

    def test_opp_next_at_reads_index(self) -> None:
        """_opp_next_at が指定 index の next/dnext ペアを正しく返すこと。"""
        from scripts.label_exchange_outcome import NpzRecord
        mod = _import_v2()
        rec = NpzRecord(
            "v29", "2P",
            np.array([0.0, 1.0]), np.array([0, 0]),
            np.zeros((2, BOARD_ROWS, BOARD_COLS)),
            np.array([0.0, 0.0]), np.array([100, 200]),
            next1_a=np.array([1, 2], dtype=np.int8),
            next1_b=np.array([3, 4], dtype=np.int8),
            dnext_a=np.array([5, -1], dtype=np.int8),
            dnext_b=np.array([1, -1], dtype=np.int8),
        )
        next_pair, dnext_pair = mod._opp_next_at(rec, 0)
        assert next_pair == (1, 3)
        assert dnext_pair == (5, 1)
        next_pair2, dnext_pair2 = mod._opp_next_at(rec, 1)
        assert dnext_pair2 == (-1, -1)

    def test_slice_record_propagates_next_fields(self) -> None:
        """_slice_record がマスク適用時に next 列も一緒にスライスすること。"""
        from scripts.label_exchange_outcome import NpzRecord
        mod = _import_v2()
        rec = NpzRecord(
            "v29", "1P",
            np.array([0.0, 1.0, 2.0]), np.array([0, 0, 1]),
            np.zeros((3, BOARD_ROWS, BOARD_COLS)),
            np.array([1.0, 1.0, 0.0]), np.array([100, 200, 300]),
            next1_a=np.array([1, 2, 3], dtype=np.int8),
            next1_b=np.array([1, 2, 3], dtype=np.int8),
            dnext_a=np.array([1, 2, 3], dtype=np.int8),
            dnext_b=np.array([1, 2, 3], dtype=np.int8),
        )
        mask = rec.game_idx == 0
        sliced = mod._slice_record(rec, mask)
        assert len(sliced.next1_a) == 2
        assert sliced.next1_a.tolist() == [1, 2]
