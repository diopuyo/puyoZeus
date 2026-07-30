"""HybridClassifier の override prob 境界挙動 + default 値テスト (cycle 71)."""
from __future__ import annotations

import numpy as np
import pytest

from src.board import (
    COLOR_BLUE, COLOR_EMPTY, COLOR_RED,
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

    def predict_proba_batch(self, bgr_patches: list[np.ndarray]) -> np.ndarray:
        """classify_batch テスト用: 全 patch に同じ固定確率を返す."""
        return np.tile(self._probs[None, :], (len(bgr_patches), 1))


class _CountingUiMatcher:
    """is_ui 呼出回数を数える偽 UiMaskMatcher (案B の呼出削減検証用)。"""

    def __init__(self, is_ui_result: bool = True) -> None:
        self.call_count: int = 0
        self._is_ui_result = is_ui_result

    def is_ui(self, bgr_patch: np.ndarray) -> bool:
        self.call_count += 1
        return self._is_ui_result


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


class TestUiMaskCellsRestriction:
    """案B (2026-07-30): ui_mask_cells + cell_positions によるセル限定テスト。

    is_ui() の呼出自体を対象セル以外で省略できることと、既定 (None) では
    bit-identical (従来通り全セル判定) であることを確認する。
    """

    def _make_patches(self, n: int, size: int = 8) -> list[np.ndarray]:
        return [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(n)]

    def test_default_none_checks_all_cells_bit_identical(self):
        """ui_mask_cells=None (既定) では cell_positions を渡しても全セル判定
        (= 呼出回数が patch 数と一致、従来挙動と bit-identical)。
        """
        counting_matcher = _CountingUiMatcher(is_ui_result=False)
        clf = HybridClassifier(use_ui_mask=False)  # cnn=None (HSV-only 経路)
        clf._ui_matcher = counting_matcher  # type: ignore[assignment]
        patches = self._make_patches(6)
        positions = [(0, 0), (1, 2), (2, 2), (1, 1), (1, 3), (0, 2)]
        colors_a = clf.classify_batch(patches, cell_positions=positions)
        assert counting_matcher.call_count == 6
        # positions を渡さない旧来呼出でも同じ結果 (bit-identical)
        counting_matcher.call_count = 0
        colors_b = clf.classify_batch(patches)
        assert counting_matcher.call_count == 6
        assert colors_a == colors_b

    def test_restricted_cells_skip_is_ui_outside_target(self):
        """ui_mask_cells 指定時、対象セル以外は is_ui 呼出を省略する。"""
        counting_matcher = _CountingUiMatcher(is_ui_result=True)
        clf = HybridClassifier(
            use_ui_mask=False, ui_mask_cells=frozenset({(1, 2)}),
        )
        clf._ui_matcher = counting_matcher  # type: ignore[assignment]
        patches = self._make_patches(6)
        # (1, 2) が1つだけ含まれる位置リスト (751セル→約12回/フレームの縮小版)
        positions = [(0, 0), (1, 2), (2, 2), (1, 1), (1, 3), (0, 2)]
        colors = clf.classify_batch(patches, cell_positions=positions)
        assert counting_matcher.call_count == 1
        # 対象セル (index=1) のみ is_ui=True (matcher固定) → EMPTY 化。
        # それ以外は is_ui 呼出自体を省略し常時 False 扱い (= HSV 分類結果のまま)。
        from src.board import COLOR_EMPTY
        assert colors[1] == COLOR_EMPTY
        expected_non_target = clf._hsv.classify(patches[0])
        for i in (0, 2, 3, 4, 5):
            assert colors[i] == expected_non_target

    def test_restricted_cells_without_cell_positions_falls_back_to_full_check(self):
        """ui_mask_cells 指定でも cell_positions 未指定なら全セル判定にフォールバック
        (両方揃わないと制限が効かない = 安全側デフォルト)。
        """
        counting_matcher = _CountingUiMatcher(is_ui_result=False)
        clf = HybridClassifier(
            use_ui_mask=False, ui_mask_cells=frozenset({(1, 2)}),
        )
        clf._ui_matcher = counting_matcher  # type: ignore[assignment]
        patches = self._make_patches(6)
        clf.classify_batch(patches)  # cell_positions 未指定
        assert counting_matcher.call_count == 6

    def test_length_mismatch_falls_back_to_full_check(self):
        """cell_positions の長さが bgr_patches と不一致なら安全側 (全セル判定)。"""
        counting_matcher = _CountingUiMatcher(is_ui_result=False)
        clf = HybridClassifier(
            use_ui_mask=False, ui_mask_cells=frozenset({(1, 2)}),
        )
        clf._ui_matcher = counting_matcher  # type: ignore[assignment]
        patches = self._make_patches(6)
        clf.classify_batch(patches, cell_positions=[(1, 2)])  # 長さ不一致
        assert counting_matcher.call_count == 6

    def test_call_reduction_matches_751_to_12_scale(self):
        """UI_MASK_TARGET_CELLS 実定数を使い、751セル相当→対象1セルのみ
        呼出されることを確認する (速度改善の主張の単体検証)。
        """
        from src.ui_mask import UI_MASK_TARGET_CELLS
        counting_matcher = _CountingUiMatcher(is_ui_result=False)
        clf = HybridClassifier(
            use_ui_mask=False, ui_mask_cells=UI_MASK_TARGET_CELLS,
        )
        clf._ui_matcher = counting_matcher  # type: ignore[assignment]
        # 6列 x 13行 (HIDDEN_ROWS込み) x 2P分 = 156 セル相当の縮小版として
        # 全board相当 (12*6 + 1*6)*2 = 156 を模した位置リストを生成
        positions = [
            (row, col) for row in range(13) for col in range(6)
        ] * 2  # 1P/2P で 156*2=312 (実運用の751はCNNパッチ複数解像度等を含む)
        patches = self._make_patches(len(positions))
        clf.classify_batch(patches, cell_positions=positions)
        # 位置リスト中 (1, 2) の出現回数だけ呼ばれる
        expected_calls = sum(1 for p in positions if p in UI_MASK_TARGET_CELLS)
        assert counting_matcher.call_count == expected_calls
        assert counting_matcher.call_count < len(positions)
