"""
画面中央下部の「数値★ WIN ★数値」表示（勝敗パネル）の検出と数値読取。

原理:
    - "★ WIN ★" 部分はフォント・色とも固定。テンプレートマッチで存在判定。
    - 存在するフレームは「試合セクション」（対戦動画の一部）。
    - 左右の数値（勝ち数）の変化で個別の試合が終わったことを検出。

判定ロジック:
    - パネル未検出 → 試合外（イントロ、インタビュー、ブラウザ等）
    - パネル検出 + 数値 (L, R) 同じ → 同一試合継続 or 試合準備中
    - パネル検出 + 数値変化 → 直前に 1 試合終了

テンプレ:
    models/ui_templates/win_panel/star_win_star.png  (200×40 相当)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# テンプレート / 数値 ROI の固定座標（1920×1080 前提）
PANEL_Y_RANGE: tuple[int, int] = (960, 1020)  # 縦方向（少し広げて許容）
PANEL_X_RANGE: tuple[int, int] = (800, 1120)  # パネル全体
NUMBER_LEFT_X: tuple[int, int] = (810, 870)   # 左側数値
NUMBER_RIGHT_X: tuple[int, int] = (1050, 1110)  # 右側数値
NUMBER_Y: tuple[int, int] = (965, 1010)

# マッチ閾値
PANEL_NCC_THRESHOLD: float = 0.70

DEFAULT_TEMPLATE: Path = Path("models/ui_templates/win_panel/star_win_star.png")


@dataclass(frozen=True)
class WinPanelResult:
    """勝敗パネル検出結果。"""
    present: bool
    score: float           # テンプレートマッチスコア（最大 NCC）
    digit_left_roi: np.ndarray | None = None   # 数値領域（OCR 前提）
    digit_right_roi: np.ndarray | None = None


class WinPanelDetector:
    """勝敗パネル `数値★WIN★数値` をテンプレートマッチで検出する。"""

    def __init__(
        self,
        template: np.ndarray,
        threshold: float = PANEL_NCC_THRESHOLD,
    ) -> None:
        # グレースケールで保持
        if template.ndim == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self._template = template
        self._threshold = threshold

    @classmethod
    def load_default(
        cls,
        template_path: Path = DEFAULT_TEMPLATE,
        threshold: float = PANEL_NCC_THRESHOLD,
    ) -> "WinPanelDetector":
        img = cv2.imread(str(template_path))
        if img is None:
            raise FileNotFoundError(f"テンプレートが存在しない: {template_path}")
        return cls(template=img, threshold=threshold)

    def detect(self, frame: np.ndarray) -> WinPanelResult:
        if frame is None or frame.ndim != 3:
            return WinPanelResult(present=False, score=0.0)
        h, w = frame.shape[:2]
        if (h, w) != (1080, 1920):
            # 必要なら呼び出し側でリサイズしておく想定
            return WinPanelResult(present=False, score=0.0)

        y1, y2 = PANEL_Y_RANGE
        x1, x2 = PANEL_X_RANGE
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return WinPanelResult(present=False, score=0.0)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        th, tw = self._template.shape[:2]
        if roi_gray.shape[0] < th or roi_gray.shape[1] < tw:
            return WinPanelResult(present=False, score=0.0)

        result = cv2.matchTemplate(roi_gray, self._template, cv2.TM_CCOEFF_NORMED)
        score = float(result.max())
        present = score >= self._threshold

        digit_left = frame[NUMBER_Y[0]:NUMBER_Y[1], NUMBER_LEFT_X[0]:NUMBER_LEFT_X[1]]
        digit_right = frame[NUMBER_Y[0]:NUMBER_Y[1], NUMBER_RIGHT_X[0]:NUMBER_RIGHT_X[1]]

        return WinPanelResult(
            present=present,
            score=score,
            digit_left_roi=digit_left if present else None,
            digit_right_roi=digit_right if present else None,
        )
