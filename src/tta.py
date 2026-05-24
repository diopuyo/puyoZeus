"""
Test-Time Augmentation (TTA) ラッパー。

推論時に複数のバリエーション（左右反転、明度シフト等）でモデルを呼び出し、
クラス確率の平均で投票する。学習を変えずに推論精度のみ底上げを狙う。

ぷよぷよパッチは「左右対称」だが「上下非対称」（目玉が上）なので:
    - 左右反転 (horizontal flip): 安全
    - 上下反転 (vertical flip): 適用しない
    - 90度回転: 上下情報が変わるので適用しない
    - 明度 ±N: 照明・ハロー差を吸収
    - 軽いガウシアンノイズ: 過学習耐性

使い方:
    from src.patch_classifier import CnnPatchClassifier
    from src.tta import TTAClassifier

    base = CnnPatchClassifier.load("models/cnn_global_best.pt")
    tta = TTAClassifier(base)
    color_code = tta.classify(patch)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from src.patch_classifier import CLASS_INDEX_TO_COLOR

# 1 枚あたりの augmentation 適用関数
Augmenter = Callable[[np.ndarray], np.ndarray]


def _identity(patch: np.ndarray) -> np.ndarray:
    """無加工。"""
    return patch


def _flip_horizontal(patch: np.ndarray) -> np.ndarray:
    """左右反転。"""
    return cv2.flip(patch, 1)


def _brightness(delta: int) -> Augmenter:
    """画素値を ±delta シフト（クリップ）。"""
    def _aug(patch: np.ndarray) -> np.ndarray:
        out = patch.astype(np.int32) + delta
        return np.clip(out, 0, 255).astype(np.uint8)
    return _aug


# デフォルトの augmentation セット（5 視点）
DEFAULT_AUGMENTERS: tuple[Augmenter, ...] = (
    _identity,
    _flip_horizontal,
    _brightness(+10),
    _brightness(-10),
    # flip + brightness 軽い組合せ
    lambda p: _brightness(+8)(_flip_horizontal(p)),
)


class TTAClassifier:
    """
    任意の `predict_proba` を持つ分類器を TTA でラップする。

    要件:
        base.predict_proba(patch) -> np.ndarray (shape=(NUM_CLASSES,))

    投票:
        各 augmentation でのクラス確率を取得し、平均 → argmax をクラスに採用。
    """

    def __init__(
        self,
        base,
        augmenters: tuple[Augmenter, ...] = DEFAULT_AUGMENTERS,
    ) -> None:
        if not hasattr(base, "predict_proba"):
            raise TypeError(
                "TTA に渡す base 分類器は predict_proba(patch)->np.ndarray を実装する必要がある"
            )
        self._base = base
        self._augmenters = augmenters

    def classify(self, bgr_patch: np.ndarray) -> int:
        probs = self._aggregate_proba(bgr_patch)
        idx = int(np.argmax(probs))
        return CLASS_INDEX_TO_COLOR[idx]

    def predict_proba(self, bgr_patch: np.ndarray) -> np.ndarray:
        """augmentation 平均後のクラス確率を返す。"""
        return self._aggregate_proba(bgr_patch)

    def _aggregate_proba(self, bgr_patch: np.ndarray) -> np.ndarray:
        votes: list[np.ndarray] = []
        for aug in self._augmenters:
            try:
                augmented = aug(bgr_patch)
            except Exception:
                continue
            probs = self._base.predict_proba(augmented)
            votes.append(probs)
        if not votes:
            # フォールバック: ベースのみ
            return self._base.predict_proba(bgr_patch)
        return np.mean(np.stack(votes, axis=0), axis=0)
