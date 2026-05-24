"""R-3 物理整合性チェックのテスト.

ぷよぷよルール (4+連結消去・重力・色種数 ≤5) で擬似ラベルを
cross-validate するモジュールの動作確認。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

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
from src.self_supervised.physical_consistency import (
    MAX_COLORS_IN_GAME,
    check_cell_color_settle_consistency,
    check_color_count,
    check_gravity_rule,
    check_no_pre_chain_4_plus_connection,
    filter_pseudo_labels_by_consistency,
)
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL,
    PseudoLabelSample,
)


# ============================
# helper
# ============================


def _empty_board() -> Board:
    return Board()


def _filled_floor_red(board: Board, row: int = 12) -> Board:
    """指定 row を全部 RED にする (4+ 連結が必ず生まれる)."""
    for c in range(BOARD_COLS):
        board.set(row, c, COLOR_RED)
    return board


# ============================
# check_color_count
# ============================


class TestCheckColorCount:
    def test_empty_board_zero_colors(self) -> None:
        ok, colors = check_color_count(_empty_board())
        assert ok is True
        assert colors == set()

    def test_5_colors_ok(self) -> None:
        b = Board()
        # 5 色を 1 つずつ別 row に置く
        b.set(12, 0, COLOR_RED)
        b.set(11, 0, COLOR_BLUE)
        b.set(10, 0, COLOR_GREEN)
        b.set(9, 0, COLOR_YELLOW)
        b.set(8, 0, COLOR_PURPLE)
        ok, colors = check_color_count(b)
        assert ok is True
        assert len(colors) == 5

    def test_5_colors_plus_ojama_ok(self) -> None:
        """おじゃまは色種数カウント対象外."""
        b = Board()
        for i, color in enumerate([
            COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE,
        ]):
            b.set(12 - i, 0, color)
        b.set(12, 1, COLOR_OJAMA)
        ok, colors = check_color_count(b)
        assert ok is True
        assert COLOR_OJAMA not in colors
        assert len(colors) == 5

    def test_unknown_excluded(self) -> None:
        b = Board()
        b.set(12, 0, COLOR_RED)
        b.set(11, 0, COLOR_UNKNOWN)
        ok, colors = check_color_count(b)
        assert ok is True
        assert COLOR_UNKNOWN not in colors
        assert colors == {COLOR_RED}

    def test_max_colors_constant(self) -> None:
        """ぷよぷよ通の最大同時色数."""
        assert MAX_COLORS_IN_GAME == 5


# ============================
# check_gravity_rule
# ============================


class TestCheckGravityRule:
    def test_empty_board_ok(self) -> None:
        ok, viol = check_gravity_rule(_empty_board())
        assert ok is True
        assert viol == []

    def test_floor_filled_ok(self) -> None:
        b = Board()
        for c in range(BOARD_COLS):
            b.set(12, c, COLOR_RED)
        ok, viol = check_gravity_rule(b)
        assert ok is True
        assert viol == []

    def test_stack_from_bottom_ok(self) -> None:
        """通常の積み (12, 11, 10) は重力 OK."""
        b = Board()
        b.set(12, 0, COLOR_RED)
        b.set(11, 0, COLOR_BLUE)
        b.set(10, 0, COLOR_GREEN)
        ok, _ = check_gravity_rule(b)
        assert ok is True

    def test_airborne_puyo_violation(self) -> None:
        """row=10 に puyo, row=11,12 が空 → 違反."""
        b = Board()
        b.set(10, 0, COLOR_RED)
        ok, viol = check_gravity_rule(b)
        assert ok is False
        assert (10, 0) in viol

    def test_unknown_does_not_break_continuity(self) -> None:
        """UNKNOWN セルは連続性を遮断しない (重力違反扱いせず)."""
        b = Board()
        b.set(12, 0, COLOR_RED)
        b.set(11, 0, COLOR_UNKNOWN)
        b.set(10, 0, COLOR_BLUE)
        ok, viol = check_gravity_rule(b)
        assert ok is True
        assert viol == []

    def test_multiple_columns_independent(self) -> None:
        b = Board()
        b.set(12, 0, COLOR_RED)
        b.set(10, 1, COLOR_BLUE)  # column 1 で 違反
        ok, viol = check_gravity_rule(b)
        assert ok is False
        assert (10, 1) in viol
        assert (12, 0) not in viol


# ============================
# check_no_pre_chain_4_plus_connection
# ============================


class TestCheckNoPreChain4Plus:
    def test_empty_board_ok(self) -> None:
        ok, viol = check_no_pre_chain_4_plus_connection(_empty_board())
        assert ok is True
        assert viol == []

    def test_3_connected_ok(self) -> None:
        """3 連結なら STABLE で残り得る."""
        b = Board()
        b.set(12, 0, COLOR_RED)
        b.set(12, 1, COLOR_RED)
        b.set(12, 2, COLOR_RED)
        ok, _ = check_no_pre_chain_4_plus_connection(b)
        assert ok is True

    def test_4_horizontal_violation(self) -> None:
        b = Board()
        b.set(12, 0, COLOR_RED)
        b.set(12, 1, COLOR_RED)
        b.set(12, 2, COLOR_RED)
        b.set(12, 3, COLOR_RED)
        ok, viol = check_no_pre_chain_4_plus_connection(b)
        assert ok is False
        assert len(viol) == 1
        assert viol[0]["color"] == COLOR_RED
        assert len(viol[0]["cells"]) == 4

    def test_4_vertical_violation(self) -> None:
        b = Board()
        for r in (12, 11, 10, 9):
            b.set(r, 0, COLOR_BLUE)
        ok, viol = check_no_pre_chain_4_plus_connection(b)
        assert ok is False
        assert viol[0]["color"] == COLOR_BLUE

    def test_4_l_shape_violation(self) -> None:
        """L 字 4 連結も検出."""
        b = Board()
        b.set(12, 0, COLOR_GREEN)
        b.set(12, 1, COLOR_GREEN)
        b.set(11, 0, COLOR_GREEN)
        b.set(10, 0, COLOR_GREEN)
        ok, viol = check_no_pre_chain_4_plus_connection(b)
        assert ok is False
        assert len(viol[0]["cells"]) == 4

    def test_ojama_4_plus_not_violation(self) -> None:
        """おじゃまは消去対象外なので 4+ 連結も合法."""
        b = Board()
        for c in range(4):
            b.set(12, c, COLOR_OJAMA)
        ok, viol = check_no_pre_chain_4_plus_connection(b)
        assert ok is True
        assert viol == []

    def test_separated_3s_ok(self) -> None:
        """同色でも分離していれば 3 連結ずつなので OK."""
        b = Board()
        # row=12 で c=0..2 RED, c=3 EMPTY, c=4..5 RED
        b.set(12, 0, COLOR_RED)
        b.set(12, 1, COLOR_RED)
        b.set(12, 2, COLOR_RED)
        b.set(12, 4, COLOR_RED)
        b.set(12, 5, COLOR_RED)
        ok, _ = check_no_pre_chain_4_plus_connection(b)
        assert ok is True


# ============================
# check_cell_color_settle_consistency
# ============================


class TestSettleConsistency:
    def test_valid_board_ok(self) -> None:
        b = Board()
        b.set(12, 0, COLOR_RED)
        b.set(12, 1, COLOR_BLUE)
        ok, reason = check_cell_color_settle_consistency(
            COLOR_RED, b, (12, 0),
        )
        assert ok is True
        assert reason == ""

    def test_color_violation_via_check_color_count(self) -> None:
        """6 色の Board は VALID_COLORS の都合で構築不能なため、
        check_color_count を MAX_COLORS_IN_GAME=5 が機能する点だけ確認."""
        # 5 色置き → OK
        b = Board()
        for i, color in enumerate([
            COLOR_RED, COLOR_BLUE, COLOR_GREEN,
            COLOR_YELLOW, COLOR_PURPLE,
        ]):
            b.set(12 - i, 0, color)
        ok, colors = check_color_count(b)
        assert ok is True
        assert len(colors) == 5
        # statebleed: ok と reason が空文字 (settle consistency は通る)
        ok2, reason = check_cell_color_settle_consistency(
            COLOR_RED, b, (12, 0),
        )
        assert ok2 is True
        assert reason == ""

    def test_gravity_violation_detected(self) -> None:
        b = Board()
        b.set(10, 0, COLOR_RED)  # 空中 puyo
        ok, reason = check_cell_color_settle_consistency(
            COLOR_RED, b, (10, 0),
        )
        assert ok is False
        assert "gravity" in reason

    def test_4plus_violation_detected(self) -> None:
        b = Board()
        for c in range(4):
            b.set(12, c, COLOR_RED)
        ok, reason = check_cell_color_settle_consistency(
            COLOR_RED, b, (12, 0),
        )
        assert ok is False
        assert "4plus" in reason


# ============================
# filter_pseudo_labels_by_consistency
# ============================


def _make_cell_sample(
    timestamp: float, side: str, row: int, col: int, color: int,
) -> PseudoLabelSample:
    """cell 擬似ラベル sample を生成."""
    return PseudoLabelSample(
        component=COMPONENT_CELL,
        timestamp=timestamp,
        input_data={
            "patch": np.zeros((4, 4, 3), dtype=np.uint8),
            "side": side,
            "row": row,
            "col": col,
        },
        label=color,
        confidence=0.9,
        metadata={"frame_idx": int(timestamp * 30)},
    )


class TestFilterByConsistency:
    def test_no_lookup_noop(self) -> None:
        samples = [
            _make_cell_sample(0.1, "1P", 12, 0, COLOR_RED),
            _make_cell_sample(0.2, "1P", 12, 1, COLOR_BLUE),
        ]
        out, stats = filter_pseudo_labels_by_consistency(samples, None)
        assert len(out) == 2
        assert stats["n_in"] == 2
        assert stats["n_out"] == 2
        assert stats["n_no_board"] == 2

    def test_all_valid_kept(self) -> None:
        b = Board()
        b.set(12, 0, COLOR_RED)
        b.set(12, 1, COLOR_BLUE)

        def lookup(_t: float, _s: str) -> Optional[Board]:
            return b

        samples = [
            _make_cell_sample(0.1, "1P", 12, 0, COLOR_RED),
            _make_cell_sample(0.2, "1P", 12, 1, COLOR_BLUE),
        ]
        out, stats = filter_pseudo_labels_by_consistency(samples, lookup)
        assert len(out) == 2
        assert stats["n_out"] == 2
        assert stats["n_color_violation"] == 0

    def test_4plus_violation_filtered(self) -> None:
        """4+ 連結を持つ盤面の sample は除外される."""
        b = Board()
        for c in range(4):
            b.set(12, c, COLOR_RED)

        def lookup(_t: float, _s: str) -> Optional[Board]:
            return b

        samples = [_make_cell_sample(0.1, "1P", 12, 0, COLOR_RED)]
        out, stats = filter_pseudo_labels_by_consistency(samples, lookup)
        assert len(out) == 0
        assert stats["n_4plus_violation"] == 1

    def test_gravity_violation_filtered(self) -> None:
        b = Board()
        b.set(10, 0, COLOR_RED)  # 空中

        def lookup(_t: float, _s: str) -> Optional[Board]:
            return b

        samples = [_make_cell_sample(0.1, "1P", 10, 0, COLOR_RED)]
        out, stats = filter_pseudo_labels_by_consistency(samples, lookup)
        assert len(out) == 0
        assert stats["n_gravity_violation"] == 1

    def test_no_board_counted(self) -> None:
        def lookup(_t: float, _s: str) -> Optional[Board]:
            return None

        samples = [_make_cell_sample(0.1, "1P", 12, 0, COLOR_RED)]
        out, stats = filter_pseudo_labels_by_consistency(samples, lookup)
        assert len(out) == 0
        assert stats["n_no_board"] == 1

    def test_mixed_samples(self) -> None:
        valid_b = Board()
        valid_b.set(12, 0, COLOR_RED)
        invalid_b = Board()
        for c in range(4):
            invalid_b.set(12, c, COLOR_BLUE)

        def lookup(t: float, _s: str) -> Optional[Board]:
            return valid_b if t < 1.0 else invalid_b

        samples = [
            _make_cell_sample(0.1, "1P", 12, 0, COLOR_RED),  # valid
            _make_cell_sample(0.2, "1P", 12, 1, COLOR_RED),  # valid (other cell)
            _make_cell_sample(2.0, "1P", 12, 0, COLOR_BLUE),  # invalid (4+)
        ]
        out, stats = filter_pseudo_labels_by_consistency(samples, lookup)
        assert stats["n_out"] == 2
        assert stats["n_4plus_violation"] == 1
        assert stats["n_in"] == 3

    def test_invalid_sample_skipped(self) -> None:
        """label=None や row/col 欠落は other に分類."""
        bad_sample = PseudoLabelSample(
            component=COMPONENT_CELL,
            timestamp=0.1,
            input_data={"patch": np.zeros((4, 4, 3), dtype=np.uint8)},
            label=None,
            confidence=0.9,
            metadata={},
        )

        def lookup(_t: float, _s: str) -> Optional[Board]:
            return Board()

        out, stats = filter_pseudo_labels_by_consistency([bad_sample], lookup)
        assert len(out) == 0
        assert stats["n_other"] == 1


# ============================
# Edge case
# ============================


class TestEdgeCases:
    def test_full_unknown_board_color_count_ok(self) -> None:
        b = Board()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                b.set(r, c, COLOR_UNKNOWN)
        ok, colors = check_color_count(b)
        assert ok is True
        assert colors == set()

    def test_full_unknown_gravity_ok(self) -> None:
        b = Board()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                b.set(r, c, COLOR_UNKNOWN)
        ok, _ = check_gravity_rule(b)
        assert ok is True

    def test_full_unknown_4plus_ok(self) -> None:
        b = Board()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                b.set(r, c, COLOR_UNKNOWN)
        ok, _ = check_no_pre_chain_4_plus_connection(b)
        assert ok is True

    def test_filter_empty_samples(self) -> None:
        out, stats = filter_pseudo_labels_by_consistency([], lambda t, s: Board())
        assert out == []
        assert stats["n_in"] == 0
        assert stats["n_out"] == 0
