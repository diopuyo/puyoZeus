"""W12-C: HSV 彩度極低 cell を強制 EM。

戦略:
    パッチ中心 80% の HSV S 平均が閾値 (default 30) 未満なら強制 EM。
    Puyo (RED/BLUE/GRN/YEL/PUR) は全て高彩度。低彩度は EM (黒背景) または
    OJAMA (灰色) のみ。OJAMA は別 rule で判定済なので、低彩度→ EM が安全。

bg_empty_detector との違い:
    - bg_empty_detector: 試合前 BG パターンとの類似度 (per-cell L2 距離)
    - low_sat_em_gate: 単純に saturation 平均で判定 (per-cell 不要)
    - こちらの方がシンプルで video 横断で安定 (bg fingerprint 不要)

期待効果:
    CNN が EM cell を BLUE/PUR/GRN と誤検出する hallucination を抑制。
    背景の dot pattern は saturation 中庸 (40-80) なので閾値 30 では引っかからない
    (誤上書き防止)。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, COLOR_EMPTY, COLOR_OJAMA,
    COLOR_UNKNOWN, HIDDEN_ROWS, Board,
)
from src.image_reader import (
    BoardRegion, DEFAULT_P1_REGION, DEFAULT_P2_REGION,
)


def _saturation_mean(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    h, w = bgr.shape[:2]
    crop = bgr[
        int(h * 0.1):int(h * 0.9),
        int(w * 0.1):int(w * 0.9),
    ] if bgr.size else bgr
    if crop.size == 0:
        crop = bgr
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


@dataclass
class LowSatEmGate:
    """HSV S 平均が閾値未満なら EM 強制。"""

    threshold: float = 30.0
    n_overrides: int = 0

    def reset_stats(self) -> None:
        self.n_overrides = 0

    def refine(
        self, frame: np.ndarray, side: str, board: Board,
    ) -> Board:
        """1 side の board: 非 EM cell で saturation 低なら EM 上書き。"""
        region = (
            DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
        )
        out = board.copy()
        h, w = frame.shape[:2]
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                color = int(out.get(row, col))
                if color in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA):
                    continue
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                s = _saturation_mean(patch)
                if s < self.threshold:
                    out.set(row, col, COLOR_EMPTY)
                    self.n_overrides += 1
        return out


__all__ = ["LowSatEmGate"]
