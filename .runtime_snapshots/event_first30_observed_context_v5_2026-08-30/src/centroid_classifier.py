"""W9-E: セル平均色 (centroid) 分類器。

ユーザー提案: 「範囲内のセルの色情報の平均値で色を見る、シンプル実装で評価高い気がする」

戦略:
    1. 全 review 済セル (ground truth ラベル付き) を集める
    2. 各セルの BGR + HSV ピクセル平均 (= 6 次元特徴) を計算
    3. クラスごとの centroid (=平均特徴) を保存
    4. 新パッチを分類するときは centroid との距離 (L2 in 6D) 最小のクラスを採用

メリット:
    - 学習データの特性をそのまま反映 (CNN のような暗黙学習でない)
    - 1 動画分の review でも即時更新可能
    - 推論超高速 (O(クラス数) の距離計算のみ)

デメリット:
    - 形状情報を捨てる (色が似ていれば同じと判定)
    - エフェクト被り、影、半透明には弱い

CNN とのアンサンブル (両方の予測が一致したら確定、違うときは各 confidence で判断)
での運用も検討。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.board import (
    COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_YELLOW,
)


SUPPORTED_COLORS: tuple[int, ...] = (
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)


def _patch_to_features(bgr_patch: np.ndarray) -> np.ndarray:
    """patch を 6 次元特徴 (BGR 平均 + HSV 平均) に変換。

    HSV の hue は周期的なので円周平均 (sin/cos) を使う方が厳密だが、
    まずは単純平均でシンプル実装を優先。
    """
    if bgr_patch.size == 0:
        return np.zeros(6, dtype=np.float32)
    # 中心の 80% だけ採用 (背景ピクセルを減らす)
    h, w = bgr_patch.shape[:2]
    crop = bgr_patch[
        int(h * 0.1):int(h * 0.9),
        int(w * 0.1):int(w * 0.9),
    ]
    if crop.size == 0:
        crop = bgr_patch
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bgr_mean = crop.reshape(-1, 3).mean(axis=0).astype(np.float32)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)
    return np.concatenate([bgr_mean, hsv_mean])  # (6,)


@dataclass
class CentroidClassifier:
    """クラスごとの平均色 centroid + L2 最近傍分類器。"""

    centroids: dict[int, np.ndarray] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    # 標準化用 (各特徴の標準偏差) — None なら標準化なし
    feature_std: np.ndarray | None = None

    def fit(
        self,
        patches: Sequence[np.ndarray],
        labels: Sequence[int],
        normalize: bool = True,
    ) -> None:
        """patches/labels から centroid を学習。

        normalize=True なら BGR/HSV のスケール差を吸収するため特徴を std で
        正規化してから centroid を計算 (距離測定時も同じ std を使う)。
        """
        if len(patches) != len(labels):
            raise ValueError("patches/labels の長さ不一致")
        feats = np.stack([_patch_to_features(p) for p in patches])
        if normalize:
            std = feats.std(axis=0)
            std[std < 1e-6] = 1.0
            self.feature_std = std.astype(np.float32)
            feats = feats / self.feature_std
        else:
            self.feature_std = None

        labels_arr = np.array(labels, dtype=np.int32)
        self.centroids = {}
        self.counts = {}
        for c in SUPPORTED_COLORS:
            mask = labels_arr == c
            n = int(mask.sum())
            if n == 0:
                continue
            self.centroids[c] = feats[mask].mean(axis=0).astype(np.float32)
            self.counts[c] = n

    def _featurize(self, bgr_patch: np.ndarray) -> np.ndarray:
        f = _patch_to_features(bgr_patch)
        if self.feature_std is not None:
            f = f / self.feature_std
        return f

    def classify(self, bgr_patch: np.ndarray) -> int:
        """最近傍 centroid のクラスを返す。"""
        if not self.centroids:
            return COLOR_EMPTY
        f = self._featurize(bgr_patch)
        best_c = COLOR_EMPTY
        best_d = float("inf")
        for c, ct in self.centroids.items():
            d = float(np.linalg.norm(f - ct))
            if d < best_d:
                best_d = d
                best_c = c
        return int(best_c)

    def classify_with_distance(
        self, bgr_patch: np.ndarray,
    ) -> tuple[int, float]:
        f = self._featurize(bgr_patch)
        best_c = COLOR_EMPTY
        best_d = float("inf")
        for c, ct in self.centroids.items():
            d = float(np.linalg.norm(f - ct))
            if d < best_d:
                best_d = d
                best_c = c
        return int(best_c), best_d

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "feature_std": self.feature_std,
            "counts": self.counts,
            **{f"c{c}": ct for c, ct in self.centroids.items()},
        }
        np.savez(path, **{k: v for k, v in d.items() if v is not None})

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.feature_std = (
            np.array(data["feature_std"]) if "feature_std" in data.files
            and data["feature_std"].shape != ()
            else None
        )
        self.centroids = {}
        for c in SUPPORTED_COLORS:
            key = f"c{c}"
            if key in data.files:
                self.centroids[c] = np.array(data[key])
        self.counts = (
            data["counts"].item() if "counts" in data.files else {}
        )


__all__ = [
    "CentroidClassifier",
    "SUPPORTED_COLORS",
]
