"""W10-A: Multi-classifier ensemble (v7 CNN + centroid + bg-em)。

各 classifier の failure mode が違うため、多数決 (重み付き) で精度向上を狙う。

戦略:
    1. BgEmptyDetector が EM 確定 (距離 < strict_threshold) なら即 EM 採用
       → 試合前の正確な EM パターンが残っているセル用 (最強の制約)
    2. CNN v7 と centroid の予測を取得
    3. 両者が一致 → そのまま採用 (最も信頼できる)
    4. 不一致 → centroid の最近距離 vs CNN の confidence で判断
    5. それでも怪しければ COLOR_UNKNOWN
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bg_empty_detector import BgEmptyDetector
from src.board import COLOR_EMPTY, COLOR_UNKNOWN
from src.centroid_classifier import CentroidClassifier
from src.patch_classifier import CnnPatchClassifier


@dataclass
class EnsembleClassifier:
    """v7 + centroid + bg-em の多数決アンサンブル。"""

    cnn: CnnPatchClassifier
    centroid: CentroidClassifier
    bg: BgEmptyDetector | None = None

    # Strict EM 確定閾値 (BG マッチが非常に近い)
    bg_strict_threshold: float = 12.0

    # Centroid との距離が近すぎ判定 (信頼度)
    centroid_confidence_threshold: float = 30.0

    def classify(
        self, bgr_patch: np.ndarray,
        side: str | None = None,
        row: int | None = None,
        col: int | None = None,
    ) -> int:
        """戦略: CNN v7 を default、極めて高信頼の override のみ許す。

        - BG strict EM (距離 < bg_strict_threshold): 試合前 EMPTY に酷似 → EM
        - その他のケースは全て CNN を信頼 (centroid / BG 中距離は採用しない、
          以前の試行で v7 の正解を多数上書きして劣化したため)
        """
        if self.bg is not None and side is not None:
            d_bg = self.bg.distance(side, row, col, bgr_patch)
            if d_bg < self.bg_strict_threshold:
                return COLOR_EMPTY
        return self.cnn.classify(bgr_patch)

    def classify_with_breakdown(
        self, bgr_patch: np.ndarray,
        side: str | None = None,
        row: int | None = None,
        col: int | None = None,
    ) -> dict:
        """各 sub-classifier の出力を debug 用に併記。"""
        cnn_pred = self.cnn.classify(bgr_patch)
        cen_pred, cen_dist = self.centroid.classify_with_distance(bgr_patch)
        bg_d = float("inf")
        if self.bg is not None and side is not None:
            bg_d = self.bg.distance(side, row, col, bgr_patch)
        final = self.classify(bgr_patch, side, row, col)
        return {
            "final": final,
            "cnn": cnn_pred,
            "centroid": cen_pred,
            "centroid_dist": cen_dist,
            "bg_dist": bg_d,
        }


__all__ = ["EnsembleClassifier"]
