"""HybridClassifier の override prob 境界挙動 + default 値テスト (cycle 71)."""
from __future__ import annotations

import numpy as np
import pytest

from src.board import (
    COLOR_BLUE, COLOR_EMPTY, COLOR_RED, COLOR_YELLOW,
)
from src.hybrid_classifier import (
    DEFAULT_CNN_OVERRIDE_PROB, HybridClassifier, _correct_red_yellow,
)
from src.image_reader import ColorClassifier, RED_GREEN_DIFF_FOR_RED
from src.patch_classifier import CLASS_INDEX_TO_COLOR, NUM_CLASSES


class _StubCnn:
    """テスト用の CNN スタブ. 指定した固定確率を返す."""

    def __init__(self, fixed_probs: np.ndarray) -> None:
        self._probs = fixed_probs.astype(np.float32)

    def predict_proba(self, bgr_patch: np.ndarray) -> np.ndarray:
        return self._probs.copy()


class TestHybridClassifierDefault:
    """default 値の確認 (= cycle 71 で 0.75 → 0.70)."""

    def test_default_override_prob_is_70(self):
        assert DEFAULT_CNN_OVERRIDE_PROB == pytest.approx(0.70)

    def test_constructor_uses_default(self):
        clf = HybridClassifier(use_ui_mask=False)
        assert clf._cnn_override_prob == pytest.approx(0.70)

    def test_custom_override_prob_accepted(self):
        clf = HybridClassifier(use_ui_mask=False, cnn_override_prob=0.50)
        assert clf._cnn_override_prob == pytest.approx(0.50)


class TestHybridClassifierOverrideBoundary:
    """CNN 採用閾値の境界挙動テスト (= 0.70 で CNN 採用 / HSV fallback)."""

    def _make_red_patch(self, size: int = 20) -> np.ndarray:
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        patch[:, :] = (0, 0, 200)  # BGR red
        return patch

    def _make_probs(
        self, target_color: int, prob: float,
    ) -> np.ndarray:
        """target_color に prob、 他に残りを均等配分."""
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        for i, c in enumerate(CLASS_INDEX_TO_COLOR):
            if c == target_color:
                probs[i] = prob
        leftover = (1.0 - prob) / max(1, NUM_CLASSES - 1)
        for i in range(NUM_CLASSES):
            if probs[i] == 0:
                probs[i] = leftover
        return probs

    def test_cnn_high_conf_overrides_hsv(self):
        """CNN が prob=0.85 で BLUE を返す + HSV は RED → CNN 採用で BLUE."""
        cnn = _StubCnn(self._make_probs(COLOR_BLUE, 0.85))
        hsv = ColorClassifier()
        clf = HybridClassifier(
            hsv_classifier=hsv, cnn_classifier=cnn,
            cnn_override_prob=0.70, use_ui_mask=False,
        )
        result = clf.classify(self._make_red_patch())
        assert result == COLOR_BLUE

    def test_cnn_low_conf_falls_back_to_hsv(self):
        """CNN が prob=0.55 で BLUE を返す (= 閾値 0.70 未満) + HSV RED → HSV 採用で RED."""
        cnn = _StubCnn(self._make_probs(COLOR_BLUE, 0.55))
        hsv = ColorClassifier()
        clf = HybridClassifier(
            hsv_classifier=hsv, cnn_classifier=cnn,
            cnn_override_prob=0.70, use_ui_mask=False,
        )
        result = clf.classify(self._make_red_patch())
        assert result == COLOR_RED

    def test_cnn_low_conf_agrees_with_hsv(self):
        """CNN 低確信度 (0.55) で RED + HSV RED → 一致なので RED."""
        cnn = _StubCnn(self._make_probs(COLOR_RED, 0.55))
        hsv = ColorClassifier()
        clf = HybridClassifier(
            hsv_classifier=hsv, cnn_classifier=cnn,
            cnn_override_prob=0.70, use_ui_mask=False,
        )
        result = clf.classify(self._make_red_patch())
        assert result == COLOR_RED

    def test_cnn_just_above_threshold_overrides(self):
        """CNN prob=0.71 (= 閾値 0.70 をわずかに超え) → CNN 採用."""
        cnn = _StubCnn(self._make_probs(COLOR_BLUE, 0.71))
        hsv = ColorClassifier()
        clf = HybridClassifier(
            hsv_classifier=hsv, cnn_classifier=cnn,
            cnn_override_prob=0.70, use_ui_mask=False,
        )
        result = clf.classify(self._make_red_patch())
        assert result == COLOR_BLUE


# ===== 修正②: 赤/黄補正 _correct_red_yellow テスト =====

class TestCorrectRedYellow:
    """_correct_red_yellow のユニットテスト。"""

    def _make_yellow_bgr_patch(self, size: int = 16) -> np.ndarray:
        """黄ぷよを模擬: R≈G >> B (BGRで B=0, G=200, R=200 → R-G差=0)。"""
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        patch[:, :, 0] = 0    # B
        patch[:, :, 1] = 200  # G
        patch[:, :, 2] = 200  # R
        return patch

    def _make_red_bgr_patch(self, size: int = 16) -> np.ndarray:
        """赤ぷよを模擬: R >> G (BGRで B=0, G=50, R=200 → R-G差=150 >= 80)。"""
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        patch[:, :, 0] = 0    # B
        patch[:, :, 1] = 50   # G
        patch[:, :, 2] = 200  # R
        return patch

    def test_yellow_patch_cnn_red_corrected_to_yellow(self):
        """CNN が RED と判定した黄ぷよパッチは YELLOW に補正される。"""
        yellow_patch = self._make_yellow_bgr_patch()
        result = _correct_red_yellow(COLOR_RED, yellow_patch)
        assert result == COLOR_YELLOW, (
            f"黄ぷよパッチが補正されず RED={result} のまま。"
        )

    def test_red_patch_cnn_red_stays_red(self):
        """CNN が RED と判定した本物の赤ぷよパッチは RED のまま。"""
        red_patch = self._make_red_bgr_patch()
        result = _correct_red_yellow(COLOR_RED, red_patch)
        assert result == COLOR_RED, (
            f"赤ぷよパッチが YELLOW に誤補正された: {result}。"
        )

    def test_non_red_color_unchanged(self):
        """RED 以外の色コードは補正されない (backwards compat)。"""
        patch = self._make_yellow_bgr_patch()
        assert _correct_red_yellow(COLOR_BLUE, patch) == COLOR_BLUE
        assert _correct_red_yellow(COLOR_YELLOW, patch) == COLOR_YELLOW
        assert _correct_red_yellow(COLOR_EMPTY, patch) == COLOR_EMPTY

    def test_empty_patch_red_stays_red(self):
        """空パッチ (size=0) で RED が渡されたとき補正せず RED を返す (クラッシュしない)。"""
        empty_patch = np.zeros((0, 0, 3), dtype=np.uint8)
        result = _correct_red_yellow(COLOR_RED, empty_patch)
        assert result == COLOR_RED

    def test_rg_diff_exactly_at_threshold(self):
        """R-G差 = RED_GREEN_DIFF_FOR_RED (= 80) ちょうどなら赤と判定される。"""
        patch = np.zeros((8, 8, 3), dtype=np.uint8)
        patch[:, :, 1] = 100   # G
        patch[:, :, 2] = 100 + RED_GREEN_DIFF_FOR_RED  # R = G + threshold
        result = _correct_red_yellow(COLOR_RED, patch)
        assert result == COLOR_RED

    def test_rg_diff_one_below_threshold(self):
        """R-G差 = RED_GREEN_DIFF_FOR_RED - 1 (= 79) なら YELLOW に補正される。"""
        patch = np.zeros((8, 8, 3), dtype=np.uint8)
        patch[:, :, 1] = 100   # G
        patch[:, :, 2] = 100 + RED_GREEN_DIFF_FOR_RED - 1  # R = G + threshold - 1
        result = _correct_red_yellow(COLOR_RED, patch)
        assert result == COLOR_YELLOW


class TestHybridClassifierRedYellowCorrection:
    """HybridClassifier の CNN 高確信経路における赤/黄補正の結合テスト。"""

    def _make_probs(self, target_color: int, prob: float) -> np.ndarray:
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        for i, c in enumerate(CLASS_INDEX_TO_COLOR):
            if c == target_color:
                probs[i] = prob
        leftover = (1.0 - prob) / max(1, NUM_CLASSES - 1)
        for i in range(NUM_CLASSES):
            if probs[i] == 0:
                probs[i] = leftover
        return probs

    def _make_yellow_bgr_patch(self, size: int = 16) -> np.ndarray:
        patch = np.zeros((size, size, 3), dtype=np.uint8)
        patch[:, :, 1] = 200  # G
        patch[:, :, 2] = 200  # R (R-G=0 < 80)
        return patch

    def test_cnn_high_conf_red_yellow_patch_corrected(self):
        """CNN 高確信度 RED + 黄ぷよパッチ → YELLOW に補正されること。

        cnn_override_prob=0.90 は変更しない (CYCLE_FINDINGS.md 確定ルール)。
        """
        cnn = _StubCnn(self._make_probs(COLOR_RED, 0.95))
        clf = HybridClassifier(
            cnn_classifier=cnn,
            cnn_override_prob=0.90,
            use_ui_mask=False,
        )
        yellow_patch = self._make_yellow_bgr_patch()
        result = clf.classify(yellow_patch)
        assert result == COLOR_YELLOW, (
            f"CNN 高確信 RED + 黄パッチ → {result}。YELLOW 補正が機能していない。"
        )

    def test_cnn_high_conf_red_red_patch_stays_red(self):
        """CNN 高確信度 RED + 本物赤ぷよパッチ → RED のまま (補正で壊れない)。"""
        cnn = _StubCnn(self._make_probs(COLOR_RED, 0.95))
        clf = HybridClassifier(
            cnn_classifier=cnn,
            cnn_override_prob=0.90,
            use_ui_mask=False,
        )
        red_patch = np.zeros((16, 16, 3), dtype=np.uint8)
        red_patch[:, :, 1] = 50   # G
        red_patch[:, :, 2] = 200  # R (R-G=150 >= 80)
        result = clf.classify(red_patch)
        assert result == COLOR_RED

    def test_cnn_override_prob_unchanged_at_090(self):
        """cnn_override_prob=0.90 がデフォルトとして使用できる (確定ルール不変確認)。"""
        clf = HybridClassifier(cnn_override_prob=0.90, use_ui_mask=False)
        assert clf._cnn_override_prob == pytest.approx(0.90)
