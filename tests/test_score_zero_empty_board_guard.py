"""score_zero 空盤面ガード (_apply_score_zero_empty_board_guard) のユニットテスト.

η (2026-05-25) 実装: confirmed_board が None または空盤面のとき、
score_zero_both を無効化 (= 試合外と誤判定しない) することを検証する。

テスト設計方針:
- self-contained: 実フレーム / 動画不要
- mock 最小限: Board オブジェクトを直接生成
- 5 test: None / 空盤面 / 1 cell / 18 cells / backwards compat
"""
from __future__ import annotations

import numpy as np
import pytest

from src.board import Board, COLOR_RED, COLOR_EMPTY
from src.recognition_pipeline import (
    _apply_score_zero_empty_board_guard,
    _SCORE_ZERO_EMPTY_BOARD_MIN_PUYOS,
    RecognitionPipeline,
)


# ============================
# ヘルパー
# ============================

def _make_board_with_puyos(count: int) -> Board:
    """指定数のぷよ (赤色) を左上から詰めた Board を生成する."""
    b = Board()
    placed = 0
    for r in range(13):
        for c in range(6):
            if placed >= count:
                return b
            b.set(r, c, COLOR_RED)
            placed += 1
    return b


# ============================
# test 1: confirmed_board が None のとき score_zero_both を無効化
# ============================

def test_none_board_disables_score_zero() -> None:
    """confirmed_board が両側 None → score_zero_both を False に戻す."""
    result = _apply_score_zero_empty_board_guard(
        score_zero_both=True,
        p1_confirmed=None,
        p2_confirmed=None,
    )
    assert result is False, (
        "両側 confirmed=None のとき score_zero_both=True は空盤面ガードで False になるべき"
    )


# ============================
# test 2: confirmed_board が 0 cells (空盤面) のとき score_zero_both を無効化
# ============================

def test_empty_board_disables_score_zero() -> None:
    """confirmed_board が両側 0 cells → score_zero_both を False に戻す."""
    empty1 = Board()  # 全 EMPTY
    empty2 = Board()
    assert empty1.count_puyos() == 0
    result = _apply_score_zero_empty_board_guard(
        score_zero_both=True,
        p1_confirmed=empty1,
        p2_confirmed=empty2,
    )
    assert result is False, (
        "両側 confirmed=空盤面のとき score_zero_both=True は空盤面ガードで False になるべき"
    )


# ============================
# test 3: confirmed_board に 1 cell あれば score_zero_both を無効化
# (1 < MIN_PUYOS=2 のため、まだ「認識できていない」扱い → ガード発火)
# ============================

def test_single_cell_board_disables_score_zero() -> None:
    """1 cell のみ → MIN_PUYOS=2 未満 → 空盤面ガードで False."""
    b1 = _make_board_with_puyos(1)
    assert b1.count_puyos() == 1
    result = _apply_score_zero_empty_board_guard(
        score_zero_both=True,
        p1_confirmed=b1,
        p2_confirmed=None,
    )
    assert result is False, (
        "1 cell (< MIN_PUYOS) のとき score_zero_both は空盤面ガードで False になるべき"
    )


# ============================
# test 4: confirmed_board に 18 cells (十分なぷよ) があれば既存挙動で False
# (元ロジック: count >= MIN_PUYOS で試合中確定)
# ============================

def test_sufficient_puyos_disables_score_zero() -> None:
    """18 cells → count >= MIN_PUYOS → 試合中確定 → score_zero_both=False."""
    b1 = _make_board_with_puyos(18)
    b2 = _make_board_with_puyos(18)
    assert b1.count_puyos() == 18
    result = _apply_score_zero_empty_board_guard(
        score_zero_both=True,
        p1_confirmed=b1,
        p2_confirmed=b2,
    )
    assert result is False, (
        "十分なぷよがあれば試合中確定で score_zero_both=False になるべき"
    )


# ============================
# test 5: score_zero_both=False 入力はそのまま False を返す (backwards compat)
# ============================

def test_score_zero_false_passthrough() -> None:
    """score_zero_both=False の入力は confirmed に関わらず False のまま."""
    # None でも空盤面でも、元が False なら変わらない
    assert _apply_score_zero_empty_board_guard(
        score_zero_both=False,
        p1_confirmed=None,
        p2_confirmed=None,
    ) is False
    assert _apply_score_zero_empty_board_guard(
        score_zero_both=False,
        p1_confirmed=_make_board_with_puyos(18),
        p2_confirmed=_make_board_with_puyos(18),
    ) is False


# ============================
# test 6: RecognitionPipeline クラスに定数が存在し、
#          モジュール定数と一致することを確認 (regression check)
# ============================

def test_pipeline_constant_matches_module_constant() -> None:
    """RecognitionPipeline.SCORE_ZERO_EMPTY_BOARD_MIN_PUYOS が
    モジュールレベル定数と一致する (= ロジックとクラス定数の乖離防止)."""
    assert RecognitionPipeline.SCORE_ZERO_EMPTY_BOARD_MIN_PUYOS == (
        _SCORE_ZERO_EMPTY_BOARD_MIN_PUYOS
    ), (
        "RecognitionPipeline.SCORE_ZERO_EMPTY_BOARD_MIN_PUYOS と "
        "_SCORE_ZERO_EMPTY_BOARD_MIN_PUYOS が不一致 → どちらかを修正せよ"
    )
