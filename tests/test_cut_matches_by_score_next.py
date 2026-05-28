"""
scripts/cut_matches_by_score_next.py のユニットテスト。

テスト方針:
    - 実動画不要: numpy で合成した疑似フレームで境界検出ロジックを検証
    - ScoreZeroDetector / WinPanelDetector はモックで置き換える
    - _handle_searching / _check_match_end / _hash_distance を直接テスト
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.cut_matches_by_score_next import (
    NEXT_HASH_HAMMING_THRESHOLD,
    NEXT_HASH_SIZE,
    NEXT_ROI_1P,
    NEXT_ROI_2P,
    MatchBoundary,
    _compute_next_hash,
    _finalize_boundary,
    _handle_searching,
    _hash_distance,
    _resize_to_1080p,
)


# ============================
# ヘルパ
# ============================


def _make_blank_frame(h: int = 1080, w: int = 1920) -> np.ndarray:
    """指定解像度のゼロフレームを生成する。"""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_colorful_next_roi(frame: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """NEXT ROI に指定色を塗ったフレームを返す (in-place 変更なし)。"""
    f = frame.copy()
    for y1, y2, x1, x2 in (NEXT_ROI_1P, NEXT_ROI_2P):
        f[y1:y2, x1:x2] = color
    return f


def _make_score_zero_result(both_zero: bool):
    """ScoreZeroResult の最小モック。"""
    r = MagicMock()
    r.both_zero = both_zero
    return r



# ============================
# _resize_to_1080p
# ============================


class TestResizeTo1080p:
    def test_1080p_unchanged(self):
        """1080p フレームはそのまま返る。"""
        frame = _make_blank_frame(1080, 1920)
        result = _resize_to_1080p(frame)
        assert result.shape == (1080, 1920, 3)

    def test_720p_upscaled(self):
        """720p フレームは 1080p にリサイズされる。"""
        frame = _make_blank_frame(720, 1280)
        result = _resize_to_1080p(frame)
        assert result.shape == (1080, 1920, 3)


# ============================
# _compute_next_hash / _hash_distance
# ============================


class TestNextHashAndDistance:
    def test_identical_frames_distance_zero(self):
        """同一フレームのハッシュ距離は 0 になる。"""
        frame = _make_colorful_next_roi(_make_blank_frame(), (100, 150, 200))
        h1 = _compute_next_hash(frame)
        h2 = _compute_next_hash(frame.copy())
        dist = _hash_distance(h1, h2)
        assert dist == pytest.approx(0.0, abs=1.0)

    def test_different_colors_distance_large(self):
        """全く異なる色の NEXT ROI はハッシュ距離が閾値を超える。"""
        frame_red = _make_colorful_next_roi(_make_blank_frame(), (0, 0, 200))
        frame_green = _make_colorful_next_roi(_make_blank_frame(), (0, 200, 0))
        h1 = _compute_next_hash(frame_red)
        h2 = _compute_next_hash(frame_green)
        dist = _hash_distance(h1, h2)
        assert dist > NEXT_HASH_HAMMING_THRESHOLD

    def test_hash_length(self):
        """ハッシュの長さは NEXT_HASH_SIZE^2 * 2 (1P+2P)。"""
        frame = _make_blank_frame()
        h = _compute_next_hash(frame)
        expected_len = NEXT_HASH_SIZE * NEXT_HASH_SIZE * 2
        assert len(h) == expected_len

    def test_minor_noise_below_threshold(self):
        """ごく小さいノイズはハッシュ距離が閾値未満に収まる。"""
        frame = _make_colorful_next_roi(_make_blank_frame(), (128, 128, 128))
        noisy = frame.copy()
        # NEXT ROI 外に微小ノイズを加える (ROI 外)
        noisy[0:10, 0:10] = (255, 255, 255)
        h1 = _compute_next_hash(frame)
        h2 = _compute_next_hash(noisy)
        dist = _hash_distance(h1, h2)
        # ROI 外のノイズなので距離はゼロのはず
        assert dist == pytest.approx(0.0, abs=1.0)


# ============================
# _handle_searching
# ============================


class TestHandleSearching:
    def _call(
        self,
        score_zero: bool,
        next_changed: bool,
        confirm_count: int = 2,
        current_start_confirm: int = 0,
    ) -> tuple[str, int]:
        """_handle_searching を呼び出してフェーズとカウントを返す。"""
        frame = _make_blank_frame()
        cur_hash = _make_colorful_next_roi(
            frame, (200, 100, 50) if next_changed else (0, 0, 0),
        )
        prev_hash_base = _make_blank_frame()

        cur_h = _compute_next_hash(
            _make_colorful_next_roi(frame, (200, 100, 50) if next_changed else (0, 0, 0))
        )
        prev_h = _compute_next_hash(prev_hash_base)

        zero_result = _make_score_zero_result(score_zero)
        phase, count, _ = _handle_searching(
            t=10.0,
            frame=frame,
            zero_result=zero_result,
            cur_hash=cur_h,
            prev_hash=prev_h if next_changed else cur_h,
            start_confirm_count=current_start_confirm,
            confirm_count=confirm_count,
        )
        return phase, count

    def test_no_signal_stays_searching(self):
        """score_zero=False の場合は searching のまま。"""
        phase, count = self._call(score_zero=False, next_changed=False)
        assert phase == "searching"
        assert count == 0

    def test_score_zero_but_no_next_change_stays_searching(self):
        """score=0 でもネクスト変化なしなら searching のまま。

        = 「score=0 ずっと続く区間で next が変化しない場合に偽陽性出さない」 ケース
        """
        phase, count = self._call(score_zero=True, next_changed=False)
        assert phase == "searching"
        assert count == 0

    def test_score_zero_and_next_changed_increments_count(self):
        """score=0 + next 変化で確定カウントが増加する。"""
        phase, count = self._call(
            score_zero=True, next_changed=True,
            confirm_count=3, current_start_confirm=0,
        )
        assert phase == "searching"
        assert count == 1

    def test_score_zero_and_next_changed_reaches_confirm(self):
        """確定カウントが閾値に達すると in_match に遷移する。"""
        phase, count = self._call(
            score_zero=True, next_changed=True,
            confirm_count=2, current_start_confirm=1,
        )
        assert phase == "in_match"

    def test_no_prev_hash_no_transition(self):
        """prev_hash=None の場合は next_changed 判定不能 → searching のまま。"""
        frame = _make_blank_frame()
        cur_h = _compute_next_hash(frame)
        zero_result = _make_score_zero_result(True)
        phase, count, _ = _handle_searching(
            t=0.0,
            frame=frame,
            zero_result=zero_result,
            cur_hash=cur_h,
            prev_hash=None,  # 先頭フレーム相当
            start_confirm_count=0,
            confirm_count=2,
        )
        assert phase == "searching"
        assert count == 0


# ============================
# _finalize_boundary
# ============================


class TestFinalizeBoundary:
    def test_short_match_skipped(self):
        """極端に短い試合はスキップされる (偽陽性除去)。"""
        boundaries: list[MatchBoundary] = []
        # start_triggers: prev=100s, new=103s  → 間隔 3s < buffer_sec*2=10s
        start_triggers = [100.0, 103.0]
        _finalize_boundary(
            boundaries=boundaries,
            start_triggers=start_triggers,
            buffer_sec=5.0,
            duration_sec=200.0,
            new_trigger_sec=103.0,
        )
        # 短すぎるのでスキップ
        assert len(boundaries) == 0

    def test_normal_match_finalized(self):
        """十分な長さの試合は境界として確定される。"""
        boundaries: list[MatchBoundary] = []
        start_triggers = [50.0, 120.0]
        _finalize_boundary(
            boundaries=boundaries,
            start_triggers=start_triggers,
            buffer_sec=5.0,
            duration_sec=300.0,
            new_trigger_sec=120.0,
        )
        assert len(boundaries) == 1
        b = boundaries[0]
        # clip_start = prev_trigger - buffer = 50 - 5 = 45
        assert b.clip_start_sec == pytest.approx(45.0)
        # clip_end = new_trigger - buffer = 120 - 5 = 115
        assert b.clip_end_sec == pytest.approx(115.0)
        assert b.trigger_sec == pytest.approx(50.0)

    def test_clip_start_clamped_at_zero(self):
        """動画先頭付近でバッファが 0 未満にならない。"""
        boundaries: list[MatchBoundary] = []
        start_triggers = [3.0, 80.0]
        _finalize_boundary(
            boundaries=boundaries,
            start_triggers=start_triggers,
            buffer_sec=5.0,
            duration_sec=200.0,
            new_trigger_sec=80.0,
        )
        assert len(boundaries) == 1
        # clip_start = max(0, 3-5) = 0
        assert boundaries[0].clip_start_sec == pytest.approx(0.0)


# ============================
# MatchBoundary
# ============================


class TestMatchBoundary:
    def test_fields(self):
        """MatchBoundary のフィールドが正しく設定される。"""
        b = MatchBoundary(
            match_index=0,
            clip_start_sec=5.0,
            clip_end_sec=65.0,
            trigger_sec=10.0,
            end_trigger_sec=60.0,
        )
        assert b.match_index == 0
        assert b.clip_start_sec == pytest.approx(5.0)
        assert b.clip_end_sec == pytest.approx(65.0)
        assert b.trigger_sec == pytest.approx(10.0)
        assert b.end_trigger_sec == pytest.approx(60.0)

    def test_buffer_applied(self):
        """clip_start_sec は trigger_sec - buffer_sec に相当することを確認。"""
        buffer_sec = 5.0
        trigger_sec = 12.0
        expected_start = trigger_sec - buffer_sec
        b = MatchBoundary(
            match_index=0,
            clip_start_sec=expected_start,
            clip_end_sec=70.0,
            trigger_sec=trigger_sec,
            end_trigger_sec=65.0,
        )
        assert b.clip_start_sec == pytest.approx(7.0)
