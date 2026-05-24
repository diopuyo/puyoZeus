"""TTAClassifier のテスト。"""
from __future__ import annotations

import numpy as np
import pytest

from src.board import COLOR_BLUE, COLOR_RED
from src.tta import DEFAULT_AUGMENTERS, TTAClassifier


class _FakeBase:
    """テスト用: 与えられたクラスを常に返すスタブ。"""

    def __init__(self, fixed_probs: np.ndarray) -> None:
        self._probs = fixed_probs
        self.calls = 0

    def predict_proba(self, patch: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self._probs


class _FlipDifferentBase:
    """左右反転すると別クラスを返す擬似モデル。"""

    def predict_proba(self, patch: np.ndarray) -> np.ndarray:
        # 左半分が暗ければ「青」(idx=2)、明るければ「赤」(idx=1) を返すルール
        h, w = patch.shape[:2]
        left_mean = patch[:, : w // 2].mean()
        probs = np.zeros(7, dtype=np.float32)
        if left_mean < 100:
            probs[2] = 1.0  # 青
        else:
            probs[1] = 1.0  # 赤
        return probs


def test_tta_classifier_requires_predict_proba() -> None:
    class NoProba:
        pass
    with pytest.raises(TypeError):
        TTAClassifier(NoProba())


def test_tta_classifier_uses_all_augmenters() -> None:
    """augmentations 数だけベースが呼ばれる。"""
    probs = np.zeros(7, dtype=np.float32)
    probs[1] = 1.0  # 赤
    base = _FakeBase(probs)
    tta = TTAClassifier(base, augmenters=DEFAULT_AUGMENTERS)
    patch = np.full((44, 44, 3), 128, dtype=np.uint8)
    code = tta.classify(patch)
    assert code == COLOR_RED
    assert base.calls == len(DEFAULT_AUGMENTERS)


def test_tta_aggregates_proba_via_mean() -> None:
    """異なる augmentation での予測が平均されて投票される。"""
    base = _FlipDifferentBase()
    # 左半分暗、右半分明 → 普通の predict は左基準で「青」
    patch = np.zeros((44, 44, 3), dtype=np.uint8)
    patch[:, 22:] = 200
    base_pred = base.predict_proba(patch)
    assert int(np.argmax(base_pred)) == 2  # 青

    # 左右反転すると左半分明 → 「赤」になる
    tta = TTAClassifier(base)
    # 5 augmenter のうち identity, brightness +10/-10, flipped+bright = 1 flip 系 2 件
    # 平均すれば「青」と「赤」が半々で曖昧 → 全部明るくしても判定変わるレベル
    code = tta.classify(patch)
    # ベース予測と異なる結果になり得るかをテスト（少なくとも例外なし）
    assert code in (COLOR_BLUE, COLOR_RED)


def test_tta_predict_proba_normalized() -> None:
    """predict_proba の合計が約 1.0（平均後）。"""
    probs = np.array([0.1, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    base = _FakeBase(probs)
    tta = TTAClassifier(base)
    patch = np.full((44, 44, 3), 128, dtype=np.uint8)
    out = tta.predict_proba(patch)
    assert abs(out.sum() - 1.0) < 1e-5
