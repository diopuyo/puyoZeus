"""Cell HSV 標準偏差ベースの安定度追跡 (試行 G)。

CellAnomalyDetector とは異なり、各 cell の HSV mean 時系列の **標準偏差** を
計算し、低偏差 cell は「安定している」 = 信頼度高、高偏差 cell は「不安定」
= 認識結果を直前 stable に戻すか UNKNOWN 化する。

設計:
    - 各 cell の HSV (H, S, V) mean を window で蓄積
    - 標準偏差を計算: σ_max = max(σ_H, σ_S, σ_V)
    - σ_max > THRESHOLD → 不安定 → 直前 stable color に戻す
    - σ_max ≤ THRESHOLD → 安定 → 現フレーム認識結果を lock

CellAnomalyDetector (距離ベース、急変検出) と相補。
時系列の「振動度」で判定するため、puyo の自然変動 (低振動) を許容しつつ
連続的な異常 (高振動) を検出する。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, COLOR_EMPTY, COLOR_UNKNOWN, HIDDEN_ROWS, Board,
)


# 標準偏差閾値: 各 channel の σ がこれ未満なら安定
SIGMA_THRESHOLD: float = 18.0
WINDOW: int = 5


def _cell_hsv(
    frame: np.ndarray, region, row: int, col: int,
) -> tuple[float, float, float] | None:
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
    return (
        float(np.median(hsv[:, :, 0])),
        float(np.median(hsv[:, :, 1])),
        float(np.median(hsv[:, :, 2])),
    )


@dataclass
class CellStabilityTracker:
    """cell の HSV 時系列の標準偏差で安定度判定。"""
    threshold: float = SIGMA_THRESHOLD
    window: int = WINDOW
    history: dict[
        tuple[str, int, int],
        "deque[tuple[float, float, float, int]]",
    ] = field(default_factory=dict)

    def reset(self) -> None:
        self.history = {}

    def _get_deque(
        self, key: tuple[str, int, int],
    ) -> "deque[tuple[float, float, float, int]]":
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
        """frame と region から cell の HSV を抽出、標準偏差で不安定判定。

        Returns:
            (refined_board, instability_mask shape=(12, BOARD_COLS) bool)
        """
        out = board.copy()
        unstable_mask = np.zeros((12, BOARD_COLS), dtype=bool)
        if is_chain:
            return out, unstable_mask
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                key = (side, vrow, col)
                hsv = _cell_hsv(frame, region, row, col)
                if hsv is None:
                    continue
                h, s, v = hsv
                cur_color = int(out.get(row, col))
                dq = self._get_deque(key)
                dq.append((h, s, v, cur_color))
                if len(dq) < self.window:
                    continue
                # 標準偏差を計算 (window の最後 N frame)
                hs = np.array([t[0] for t in dq])
                ss = np.array([t[1] for t in dq])
                vs = np.array([t[2] for t in dq])
                sigma_max = max(
                    float(np.std(hs)),
                    float(np.std(ss)),
                    float(np.std(vs)),
                )
                if sigma_max > self.threshold:
                    # 不安定: 履歴の最頻 color に戻す
                    color_counts: dict[int, int] = {}
                    for _, _, _, c in dq:
                        color_counts[c] = color_counts.get(c, 0) + 1
                    stable_color, _ = max(
                        color_counts.items(), key=lambda kv: kv[1],
                    )
                    if stable_color != cur_color:
                        out.set(row, col, stable_color)
                        unstable_mask[vrow, col] = True
        return out, unstable_mask


__all__ = [
    "CellStabilityTracker",
    "SIGMA_THRESHOLD",
    "WINDOW",
]
