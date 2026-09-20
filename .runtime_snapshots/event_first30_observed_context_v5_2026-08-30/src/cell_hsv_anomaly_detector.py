"""Cell HSV mean anomaly 検出 (Z-3J' 改良版)。

Z-3J の pHash 版は puyo の自然変動 (照明、回転) で過剰補正、全 18 動画悪化。
本モジュールは HSV (S, V) mean の Euclidean 距離で anomaly 検出する。

設計思想:
    - pHash は spatial pattern を見るため puyo の微小回転で大きく変わる
    - HSV (S, V) mean は色の濃さ・明るさ → 連鎖アニメ・エフェクトでは
      大きく変動するが、puyo の自然変動では小さい
    - threshold は実測 (sample mean ± 2σ) で調整

連鎖中 (is_chain=True) は対象外、ChainSimulator 予測を尊重。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    HIDDEN_ROWS,
    Board,
)

# threshold 設計: puyo 自然変動の S/V 差は ~10、連鎖アニメは 50+
SV_DISTANCE_THRESHOLD: float = 35.0
WINDOW: int = 3


def _cell_hsv_mean(
    frame: np.ndarray, region, row: int, col: int,
) -> tuple[float, float] | None:
    """cell 中央 50% の (S mean, V mean) を抽出。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])), float(np.mean(hsv[:, :, 2]))


@dataclass
class CellHsvAnomalyDetector:
    """cell の HSV (S, V) mean 距離で anomaly 検出。"""
    threshold: float = SV_DISTANCE_THRESHOLD
    window: int = WINDOW
    history: dict[
        tuple[str, int, int], "deque[tuple[float, float, int]]",
    ] = field(default_factory=dict)

    def reset(self) -> None:
        self.history = {}

    def _get_deque(
        self, key: tuple[str, int, int],
    ) -> "deque[tuple[float, float, int]]":
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window)
        return self.history[key]

    def refine(
        self,
        frame: np.ndarray,
        region,
        board: Board,
        side: str,
        is_chain: bool = False,
    ) -> tuple[Board, np.ndarray]:
        """frame と board から HSV anomaly 検出 + 補正。

        Returns:
            (refined_board, anomaly_mask shape=(12, BOARD_COLS) bool)
        """
        out = board.copy()
        anomaly_mask = np.zeros((12, BOARD_COLS), dtype=bool)
        if is_chain:
            return out, anomaly_mask
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                key = (side, vrow, col)
                hsv_pair = _cell_hsv_mean(frame, region, row, col)
                if hsv_pair is None:
                    continue
                s_mean, v_mean = hsv_pair
                cur_color = int(out.get(row, col))
                dq = self._get_deque(key)
                # 直近 stable と比較 (window 蓄積後)
                if len(dq) >= self.window:
                    distances = [
                        ((s_mean - s_old) ** 2 + (v_mean - v_old) ** 2) ** 0.5
                        for s_old, v_old, _ in dq
                    ]
                    min_dist = min(distances)
                    if min_dist > self.threshold:
                        # 直近最頻 color に戻す
                        color_counts: dict[int, int] = {}
                        for _, _, c in dq:
                            color_counts[c] = color_counts.get(c, 0) + 1
                        stable_color, _ = max(
                            color_counts.items(), key=lambda kv: kv[1],
                        )
                        if stable_color != cur_color:
                            out.set(row, col, stable_color)
                            anomaly_mask[vrow, col] = True
                # anomaly でなければ history 更新
                if not anomaly_mask[vrow, col]:
                    dq.append((s_mean, v_mean, cur_color))
        return out, anomaly_mask


__all__ = [
    "CellHsvAnomalyDetector",
    "SV_DISTANCE_THRESHOLD",
    "WINDOW",
]
