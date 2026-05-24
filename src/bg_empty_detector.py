"""W9-F: 試合前 (盤面 EMPTY) のセル平均色を記憶し、新パッチが類似していれば EM 判定。

ユーザー要望: 「背景色については試合前の盤面の情報を記憶し、その情報とマッチしていれば
empty とみなせます」

既存の BG fingerprint (背景打消し) は dot pattern 全体を記憶しているが、ここでは
各セル独立の平均色で「emptyらしさ」を判定する独立レイヤー。

使い方:
    bg = BgEmptyDetector()
    bg.calibrate_from_frame(empty_frame)  # 試合開始前の empty 盤面で 1 度
    is_em = bg.is_empty("1P", row, col, current_patch)  # 各フレームで判定
    # または
    color = bg.classify_or_none("1P", row, col, current_patch)
    # → COLOR_EMPTY なら EM、None なら他の分類器に委譲
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, HIDDEN_ROWS,
)
from src.image_reader import (
    BoardRegion, DEFAULT_P1_REGION, DEFAULT_P2_REGION,
)


def _patch_features(bgr_patch: np.ndarray) -> np.ndarray:
    """patch → 6 次元特徴 (BGR 中央 80% 平均 + HSV 中央 80% 平均)。"""
    if bgr_patch.size == 0:
        return np.zeros(6, dtype=np.float32)
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
    return np.concatenate([bgr_mean, hsv_mean])


def _extract_patch(
    frame: np.ndarray, region: BoardRegion, row: int, col: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return frame[y1:y2, x1:x2]


@dataclass
class BgEmptyDetector:
    """試合前 EMPTY 盤面の各セル平均色を記憶、L2 距離で EM 判定。"""

    # (side, row, col) -> 6 次元特徴
    bg_features: dict[tuple[str, int, int], np.ndarray] = field(
        default_factory=dict,
    )
    # マッチング閾値 (L2 距離)。試合前後で BGR/HSV 各 ±10 程度の変動を許容
    threshold: float = 18.0
    # キャリブレーション平均化用フレーム数
    n_calibration: int = 0

    def reset(self) -> None:
        self.bg_features = {}
        self.n_calibration = 0

    def calibrate_from_frame(
        self,
        frame: np.ndarray,
        p1_region: BoardRegion = DEFAULT_P1_REGION,
        p2_region: BoardRegion = DEFAULT_P2_REGION,
        accumulate: bool = False,
    ) -> None:
        """1 フレーム (試合開始前で全セル EMPTY) からキャリブレーション。

        accumulate=True なら既存特徴と移動平均で併合 (複数フレームで安定化)。
        """
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        new_features: dict[tuple[str, int, int], np.ndarray] = {}
        for side, region in (("1P", p1_region), ("2P", p2_region)):
            for vrow in range(12):  # visible rows のみ (隠し段は通常 EMPTY)
                row = vrow + HIDDEN_ROWS
                for col in range(BOARD_COLS):
                    patch = _extract_patch(frame, region, row, col)
                    if patch.size == 0:
                        continue
                    new_features[(side, vrow, col)] = _patch_features(patch)

        if accumulate and self.bg_features:
            n = self.n_calibration
            for k, v in new_features.items():
                if k in self.bg_features:
                    self.bg_features[k] = (
                        self.bg_features[k] * n + v
                    ) / (n + 1)
                else:
                    self.bg_features[k] = v
            self.n_calibration = n + 1
        else:
            self.bg_features = new_features
            self.n_calibration = 1

    def calibrate_from_video(
        self,
        cap,
        anchor_sec: float,
        offsets_sec: tuple[float, ...] = (-0.4, -0.2, 0.0),
        p1_region: BoardRegion = DEFAULT_P1_REGION,
        p2_region: BoardRegion = DEFAULT_P2_REGION,
    ) -> int:
        """試合開始秒の前後 N フレームを平均化してキャリブレーション。"""
        self.reset()
        n_used = 0
        for off in offsets_sec:
            t = max(0.0, anchor_sec + off)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            self.calibrate_from_frame(
                frame, p1_region, p2_region, accumulate=(n_used > 0),
            )
            n_used += 1
        return n_used

    def is_empty(
        self, side: str, row: int, col: int, bgr_patch: np.ndarray,
    ) -> bool:
        key = (side, row, col)
        if key not in self.bg_features:
            return False
        feat = _patch_features(bgr_patch)
        d = float(np.linalg.norm(feat - self.bg_features[key]))
        return d < self.threshold

    def distance(
        self, side: str, row: int, col: int, bgr_patch: np.ndarray,
    ) -> float:
        key = (side, row, col)
        if key not in self.bg_features:
            return float("inf")
        feat = _patch_features(bgr_patch)
        return float(np.linalg.norm(feat - self.bg_features[key]))

    def classify_or_none(
        self, side: str, row: int, col: int, bgr_patch: np.ndarray,
    ) -> int | None:
        """EM と判定できれば COLOR_EMPTY、それ以外は None (上位委譲)。"""
        if self.is_empty(side, row, col, bgr_patch):
            return COLOR_EMPTY
        return None


__all__ = [
    "BgEmptyDetector",
]
