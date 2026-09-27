"""E16専用: ばたんきゅーを観測した試合では新しい発火を受け付けない。"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from src.match_end_detector import DEFAULT_NCC_THRESHOLD, SEARCH_P1, SEARCH_P2

CONFIRM_FRAMES = 2
SIDES = ("1P", "2P")
TEMPLATE = Path("models/ui_templates/match_end_batan.png")


class ObservedDeathDetector:
    """既存ばたんきゅーテンプレートを両側へ適用し、連続観測だけを公開する。"""
    def __init__(self) -> None:
        self.template = cv2.imread(str(TEMPLATE), cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise FileNotFoundError(TEMPLATE)
        self.streak = [0, 0]
        self.scores = [0., 0.]

    def update(self, frame: np.ndarray) -> tuple[str, ...]:
        """検出閾値は既存値のまま、勝敗ラベルや未来フレームを参照しない。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        confirmed = []
        for idx, (x, y, width, height) in enumerate((SEARCH_P1, SEARCH_P2)):
            roi = gray[y:y+height, x:x+width]
            score = float(cv2.minMaxLoc(cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED))[1])
            self.scores[idx] = score
            self.streak[idx] = self.streak[idx]+1 if score >= DEFAULT_NCC_THRESHOLD else 0
            if self.streak[idx] >= CONFIRM_FRAMES:
                confirmed.append(SIDES[idx])
        return tuple(confirmed)
