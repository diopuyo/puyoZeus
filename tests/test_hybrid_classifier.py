"""HybridClassifier の override prob 境界挙動 + default 値テスト (cycle 71)."""
from __future__ import annotations

import numpy as np
import pytest

from src.board import (
    COLOR_BLUE, COLOR_EMPTY, COLOR_RED, COLOR_YELLOW,
)
from src.hybrid_classifier import (
    DEFAULT_CNN_OVERRIDE_PROB, HybridClassifier,
)
from src.image_reader import ColorClassifier
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
