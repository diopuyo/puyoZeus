"""ScoreValidator のテスト."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from src.board_state_machine import BoardState
from src.self_supervised.score_validator import (
    HIGH_CONFIDENCE,
    SCORE_AGREE_MIN,
    SCORE_HISTORY_WINDOW,
    SCORE_LENIENT_AGREE_MIN,
    SCORE_LENIENT_CONFIDENCE,
    ScoreValidator,
)


# ============================
# 共通モック
# ============================


@dataclass
class _MockSide:
    state: BoardState = BoardState.STABLE
    score: int | None = 0
    confirmed_board: Any = None


@dataclass
class _MockResult:
    is_match_active: bool = True
    p1: _MockSide = None
    p2: _MockSide = None


def _make_frame_1080p(fill: int = 0) -> np.ndarray:
    """1080p BGR フレームを生成 (single value で埋める)."""
    return np.full((1080, 1920, 3), fill, dtype=np.uint8)


def _make_result(score_1p: int, score_2p: int = 0) -> _MockResult:
    return _MockResult(
        is_match_active=True,
        p1=_MockSide(state=BoardState.STABLE, score=score_1p),
        p2=_MockSide(state=BoardState.STABLE, score=score_2p),
    )


# ============================
# 基本動作
# ============================


def test_validator_init_invalid_window():
    """history_window が小さすぎると ValueError."""
    with pytest.raises(ValueError):
        ScoreValidator(history_window=2)


def test_validator_init_invalid_agree_min():
    """agree_min が範囲外なら ValueError."""
    with pytest.raises(ValueError):
        ScoreValidator(history_window=7, agree_min=1)
    with pytest.raises(ValueError):
        ScoreValidator(history_window=7, agree_min=10)


def test_validator_skips_when_match_inactive():
    """is_match_active=False では何も emit しない."""
    v = ScoreValidator()
    res = _MockResult(is_match_active=False)
    res.p1 = _MockSide()
    res.p2 = _MockSide()
    for i in range(10):
        v.update(i, i * 0.1, res, _make_frame_1080p())
    assert v.collect() == []


def test_validator_skips_with_no_frame():
    """frame_bgr=None では何も emit しない."""
    v = ScoreValidator()
    for i in range(10):
        v.update(i, i * 0.1, _make_result(score_1p=12345678), None)
    assert v.collect() == []


def test_validator_emits_when_window_agrees():
    """同一 score を agree_min frame 観測すれば擬似ラベルが emit される.

    Phase I 改良後は lenient と HIGH 両方の emit が混じるため、
    confidence は HIGH_CONFIDENCE 以上 (= HIGH または lenient_confidence) を確認。
    """
    v = ScoreValidator()
    frame = _make_frame_1080p()
    # 同じ score を SCORE_HISTORY_WINDOW frame 投入
    score = 12345678
    for i in range(SCORE_HISTORY_WINDOW):
        v.update(i, i * 0.1, _make_result(score_1p=score), frame)
    samples = v.collect()
    # 各桁 (1P) で agree、emit される
    assert len(samples) >= 8
    # 少なくとも 1 件は HIGH 信頼で emit
    assert any(s.confidence == HIGH_CONFIDENCE for s in samples)
    # 各桁 label が score の対応桁 (HIGH のみで取り直し)
    expected_digits = [int(c) for c in f"{score:08d}"]
    high_1p = [
        s for s in samples
        if s.metadata.get("side") == "1P"
        and s.confidence == HIGH_CONFIDENCE
    ]
    by_pos = {s.metadata["digit_pos"]: s.label for s in high_1p}
    for pos in range(8):
        assert by_pos[pos] == expected_digits[pos]


def test_validator_no_emit_on_score_decrease():
    """score が decrease (単調性違反) を含む window では low score 桁は emit されない."""
    v = ScoreValidator()
    frame = _make_frame_1080p()
    # 高い score → 低い score (= 単調性違反)
    # まず高 score window で emit
    for i in range(SCORE_HISTORY_WINDOW):
        v.update(i, i * 0.1, _make_result(score_1p=99000000), frame)
    v.collect()
    # 続いて低 score window
    for i in range(SCORE_HISTORY_WINDOW):
        v.update(
            i + SCORE_HISTORY_WINDOW, (i + SCORE_HISTORY_WINDOW) * 0.1,
            _make_result(score_1p=10000000), frame,
        )
    samples_2 = v.collect()
    # 低 score (10000000) を表す digit (= 1 が pos 0、0 が pos 1..) は emit されない
    # 高 score (99000000) の各桁ラベル {9, 9, 0, 0, ...} 範囲外の "1" は emit せず
    low_pos_zero_emits = [
        s for s in samples_2
        if s.metadata.get("digit_pos") == 0 and s.label == 1
    ]
    assert len(low_pos_zero_emits) == 0


def test_validator_chain_consistency_match():
    """emit_chain_consistency は一致時 high confidence emit."""
    v = ScoreValidator()
    matched = v.emit_chain_consistency(
        side="1P", before_score=1000, after_score=1500,
        expected_delta=500, t_sec=10.0,
    )
    assert matched is True
    samples = v.collect()
    assert len(samples) == 1
    assert samples[0].confidence == HIGH_CONFIDENCE
    assert samples[0].metadata["match"] is True


def test_validator_chain_consistency_mismatch():
    """emit_chain_consistency は不一致時 medium confidence で emit."""
    v = ScoreValidator()
    matched = v.emit_chain_consistency(
        side="1P", before_score=1000, after_score=1200,
        expected_delta=500, t_sec=10.0,
    )
    assert matched is False
    samples = v.collect()
    assert len(samples) == 1
    assert samples[0].metadata["match"] is False


def test_validator_reset_clears_state():
    """reset で内部 state がクリアされる."""
    v = ScoreValidator()
    frame = _make_frame_1080p()
    for i in range(SCORE_HISTORY_WINDOW):
        v.update(i, i * 0.1, _make_result(score_1p=12345678), frame)
    v.collect()
    # reset 後は新規に emit できる
    v.reset()
    for i in range(SCORE_HISTORY_WINDOW):
        v.update(i + 100, (i + 100) * 0.1, _make_result(score_1p=11111111), frame)
    samples = v.collect()
    assert len(samples) > 0


def test_validator_collect_clears_buffer():
    """collect 後に buffer が空になる."""
    v = ScoreValidator()
    frame = _make_frame_1080p()
    for i in range(SCORE_HISTORY_WINDOW):
        v.update(i, i * 0.1, _make_result(score_1p=12345678), frame)
    first = v.collect()
    assert len(first) > 0
    second = v.collect()
    assert second == []


def test_validator_handles_1080p_only():
    """non-1080p フレームではパッチ切出さず、emit せず."""
    v = ScoreValidator()
    small_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(SCORE_HISTORY_WINDOW):
        v.update(i, i * 0.1, _make_result(score_1p=12345678), small_frame)
    samples = v.collect()
    assert samples == []


# ============================
# Phase I 改良: lenient emit
# ============================


def test_validator_lenient_emit_with_3_frames():
    """lenient モード ON で 3 frame 一致でも MEDIUM 信頼で emit."""
    v = ScoreValidator(enable_lenient_emit=True)
    frame = _make_frame_1080p()
    score = 12345678
    # SCORE_LENIENT_AGREE_MIN frame だけ投入 (= 3)
    for i in range(SCORE_LENIENT_AGREE_MIN):
        v.update(i, i * 0.1, _make_result(score_1p=score), frame)
    samples = v.collect()
    # 各桁 1P/2P (2P=score=0 で全桁 0 一致) で emit される
    assert len(samples) >= 8
    # lenient emit は SCORE_LENIENT_CONFIDENCE
    lenient_samples = [
        s for s in samples
        if s.metadata.get("source") == "window_agreement_lenient"
    ]
    assert len(lenient_samples) >= 1
    assert all(
        s.confidence == SCORE_LENIENT_CONFIDENCE for s in lenient_samples
    )


def test_validator_lenient_disabled_strict_only():
    """lenient OFF では agree_min 未満では emit しない."""
    v = ScoreValidator(enable_lenient_emit=False)
    frame = _make_frame_1080p()
    score = 12345678
    # 3 frame 投入 (agree_min=5 未満)
    for i in range(SCORE_LENIENT_AGREE_MIN):
        v.update(i, i * 0.1, _make_result(score_1p=score), frame)
    samples = v.collect()
    assert samples == []


def test_validator_high_confidence_overrides_lenient():
    """5 frame 一致なら HIGH 信頼で emit (lenient ではなく)."""
    v = ScoreValidator()
    frame = _make_frame_1080p()
    score = 22222222
    for i in range(SCORE_AGREE_MIN):
        v.update(i, i * 0.1, _make_result(score_1p=score), frame)
    samples = v.collect()
    # HIGH source
    high = [
        s for s in samples
        if s.metadata.get("source") == "window_agreement"
    ]
    assert len(high) >= 8
    assert all(s.confidence == HIGH_CONFIDENCE for s in high)


def test_validator_lenient_agree_min_validation():
    """lenient_agree_min の範囲外は ValueError."""
    with pytest.raises(ValueError):
        ScoreValidator(lenient_agree_min=1)
    with pytest.raises(ValueError):
        ScoreValidator(lenient_agree_min=10)


def test_validator_lenient_no_double_emit_per_pos():
    """同一 (frame_idx, side, pos) は lenient で 1 度 emit したら HIGH 後追いしない."""
    v = ScoreValidator()
    frame = _make_frame_1080p()
    score = 11111111
    # 3 frame 投入: lenient emit が走る
    for i in range(SCORE_LENIENT_AGREE_MIN):
        v.update(i, i * 0.1, _make_result(score_1p=score), frame)
    first = v.collect()
    # さらに 4 frame 追加 (合計 7 frame): 同じ center frame は再 emit されない
    for i in range(SCORE_LENIENT_AGREE_MIN, SCORE_HISTORY_WINDOW):
        v.update(i, i * 0.1, _make_result(score_1p=score), frame)
    second = v.collect()
    # 1 度 emit した center frame の pos キーは second でも emit せず
    # (center frame が変わると別 emit はあり得る)
    first_keys = {
        (s.metadata["frame_idx"], s.metadata["side"], s.metadata["digit_pos"])
        for s in first
    }
    second_keys = {
        (s.metadata["frame_idx"], s.metadata["side"], s.metadata["digit_pos"])
        for s in second
    }
    assert first_keys.isdisjoint(second_keys)


def test_validator_lenient_skips_monotonic_violation():
    """lenient モードでも単調性違反 frame では emit しない (旧仕様互換)."""
    v = ScoreValidator()
    frame = _make_frame_1080p()
    # まず高 score 7 frame 投入で last_confirmed を設定
    for i in range(SCORE_HISTORY_WINDOW):
        v.update(i, i * 0.1, _make_result(score_1p=99000000), frame)
    v.collect()
    # 続いて低 score 3 frame (lenient threshold) 投入 → 単調性違反で emit せず
    base = SCORE_HISTORY_WINDOW * 2
    for i in range(SCORE_LENIENT_AGREE_MIN):
        v.update(base + i, (base + i) * 0.1,
                 _make_result(score_1p=10000000), frame)
    samples = v.collect()
    # 低 score (= 1) を pos 0 で emit していたら fail
    bad = [
        s for s in samples
        if s.metadata.get("digit_pos") == 0 and s.label == 1
    ]
    assert len(bad) == 0
