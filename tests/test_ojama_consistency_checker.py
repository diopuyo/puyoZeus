"""src/ojama_consistency_checker.py のテスト (Phase O)。"""
from __future__ import annotations

import pytest

from src.ojama_consistency_checker import (
    METHOD_AGREED,
    METHOD_FALLBACK_SCORE,
    METHOD_FALLBACK_VISUAL,
    METHOD_NONE,
    METHOD_SCORE_DELTA,
    ConsistencyResult,
    OjamaConsistencyChecker,
)


def test_both_none_returns_none() -> None:
    c = OjamaConsistencyChecker()
    r = c.cross_check(score_delta_ojama=None, visual_icons=None)
    assert r.final_ojama is None
    assert r.method_used == METHOD_NONE
    assert r.confidence == 0.0


def test_score_only_uses_high_confidence() -> None:
    """score 単独 = 視覚版なし → score 採用 + high confidence。"""
    c = OjamaConsistencyChecker()
    r = c.cross_check(score_delta_ojama=120, visual_icons=None)
    assert r.final_ojama == 120
    assert r.method_used == METHOD_FALLBACK_SCORE
    assert r.confidence >= 0.85


def test_visual_only_uses_medium_confidence() -> None:
    """視覚単独 → 中信頼度。"""
    c = OjamaConsistencyChecker()
    r = c.cross_check(
        score_delta_ojama=None,
        visual_icons=[("rock", 3)],
    )
    assert r.final_ojama == 90
    assert r.method_used == METHOD_FALLBACK_VISUAL
    assert 0.4 <= r.confidence <= 0.6


def test_agreed_high_confidence() -> None:
    """両方ほぼ一致 → AGREED + high。"""
    c = OjamaConsistencyChecker()
    # score 90 → アイコン分解 = rock 3 個
    # 視覚版 rock 3 個 → 一致
    r = c.cross_check(
        score_delta_ojama=90,
        visual_icons=[("rock", 3)],
    )
    assert r.method_used == METHOD_AGREED
    assert r.final_ojama == 90
    assert r.confidence >= 0.85
    assert r.agreement == 0


def test_agreed_with_overflow_truncation() -> None:
    """端数表示落ちで視覚版が score より少なくても許容差内なら AGREED。"""
    c = OjamaConsistencyChecker()
    # score 100 → アイコン分解 = rock 3 + large 1 = 96 (端数 4 個表示落ち)
    # 視覚版が rock 3 + large 1 を見えれば 96 個分、差 0
    r = c.cross_check(
        score_delta_ojama=100,
        visual_icons=[("rock", 3), ("large", 1)],
    )
    assert r.method_used == METHOD_AGREED
    assert r.final_ojama == 100  # score を採用


def test_disagreement_lowers_confidence() -> None:
    """大きく食い違う → score 採用するが confidence 低。"""
    c = OjamaConsistencyChecker()
    # score 100、視覚 rock 6 (= 180 個) → 大きく違う
    r = c.cross_check(
        score_delta_ojama=100,
        visual_icons=[("rock", 6)],
    )
    assert r.method_used == METHOD_SCORE_DELTA
    assert r.final_ojama == 100
    assert r.confidence < 0.5


def test_zero_score_zero_visual_agreed() -> None:
    """0/0 ojama → AGREED。"""
    c = OjamaConsistencyChecker()
    r = c.cross_check(score_delta_ojama=0, visual_icons=[])
    assert r.method_used == METHOD_AGREED
    assert r.final_ojama == 0


def test_negative_score_clamped_to_zero() -> None:
    """負の score (OCR ノイズ) は 0 にクランプ。"""
    c = OjamaConsistencyChecker()
    r = c.cross_check(score_delta_ojama=-50, visual_icons=None)
    assert r.final_ojama == 0


def test_result_dataclass_fields() -> None:
    """ConsistencyResult が必要なフィールドを持つ。"""
    r = ConsistencyResult(
        final_ojama=10, confidence=0.9, method_used=METHOD_AGREED,
        score_delta_ojama=10, visual_ojama=10, agreement=0,
    )
    assert r.final_ojama == 10
    assert r.confidence == 0.9
    assert r.method_used == METHOD_AGREED


def test_score_with_partial_agreement_displayed() -> None:
    """score 200、視覚 rock 6 個 (180) の差は許容差内で AGREED。"""
    c = OjamaConsistencyChecker()
    # score 200 → アイコン分解 = rock 6 + large 3 + small 2 = 200 (制限内に収まる)
    # 視覚版が rock 6 だけ = 180、差 20 → 許容差 6 を超える → 不一致
    r = c.cross_check(
        score_delta_ojama=200,
        visual_icons=[("rock", 6)],
    )
    # score icons total と visual の差
    # icons_to_ojama [(rock,6)] = 180、score_ojama=200、diff=20
    # icon分解(200): rock 6 (180) + large 3 (18) + small 2 (2) = 200 (6アイコン)
    # 視覚 vs icon分解(score) = 180 vs 200、差 20 > 6 → 不一致
    assert r.method_used == METHOD_SCORE_DELTA  # 不一致で score 採用


def test_agreed_within_tolerance() -> None:
    """許容差 6 以内の差は AGREED。"""
    c = OjamaConsistencyChecker(diff_tolerance=6)
    # score 36 → icons = rock 1 + large 1 = 36 (full count)
    # 視覚版 rock 1 + large 1 = 36 (full count)、差 0
    r = c.cross_check(
        score_delta_ojama=36,
        visual_icons=[("rock", 1), ("large", 1)],
    )
    assert r.method_used == METHOD_AGREED
