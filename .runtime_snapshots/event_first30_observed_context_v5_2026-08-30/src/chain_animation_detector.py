"""W14-C: 画面全体の輝度/色相変動から連鎖アニメーションフレームを検出。

戦略:
    前フレームと現フレームを比較し、盤面領域全体の平均ピクセル差分が大きければ
    「連鎖アニメーション中」と判定。検出時は当該フレームの board 出力を
    前フレームの安定 board で置換 (CNN の不安定出力を回避)。

score_eraser との違い:
    - score_eraser: ScoreOcr の数値変化に依存 (OCR 失敗時動作せず)
    - chain_anim: 視覚的フラッシュ/モーションで判定 (OCR 不要、応答性高)

両方併用すれば二重チェックになる。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.board import Board


# 平均輝度差分がこの値超で「アニメ中」と判定
LUMA_DIFF_THRESHOLD: float = 18.0
# 検出後に board 保留する最大フレーム数 (フォールバック)
MAX_HOLD_FRAMES: int = 3


@dataclass
class ChainAnimationDetector:
    """画面全体の motion で連鎖アニメ frame を検出。"""

    luma_threshold: float = LUMA_DIFF_THRESHOLD
    max_hold_frames: int = MAX_HOLD_FRAMES

    prev_gray: dict[str, np.ndarray | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )
    last_stable_board: dict[str, Board | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )
    hold_remaining: dict[str, int] = field(
        default_factory=lambda: {"1P": 0, "2P": 0}
    )
    n_holds: int = 0

    def reset(self) -> None:
        self.prev_gray = {"1P": None, "2P": None}
        self.last_stable_board = {"1P": None, "2P": None}
        self.hold_remaining = {"1P": 0, "2P": 0}
        self.n_holds = 0

    def is_animating(
        self, side: str, field_crop: np.ndarray,
    ) -> bool:
        """field 全体の輝度差分から連鎖アニメ判定。"""
        if field_crop is None or field_crop.size == 0:
            return False
        gray = cv2.cvtColor(field_crop, cv2.COLOR_BGR2GRAY)
        prev = self.prev_gray.get(side)
        animating = False
        if prev is not None and prev.shape == gray.shape:
            diff = np.abs(
                gray.astype(np.int16) - prev.astype(np.int16)
            ).mean()
            if diff > self.luma_threshold:
                animating = True
        self.prev_gray[side] = gray
        return animating

    def refine(
        self, side: str, field_crop: np.ndarray, board: Board,
    ) -> Board:
        """field crop と board を受けて、連鎖アニメ中なら前 stable で置換。"""
        animating = self.is_animating(side, field_crop)
        if animating:
            self.hold_remaining[side] = self.max_hold_frames
            self.n_holds += 1
            if self.last_stable_board.get(side) is not None:
                # 前 stable board を返す
                return self.last_stable_board[side].copy()
            # 前回が無ければ board そのまま (初回)
            return board

        # ホールド残あれば board を維持
        if self.hold_remaining.get(side, 0) > 0:
            self.hold_remaining[side] -= 1
            self.n_holds += 1
            if self.last_stable_board.get(side) is not None:
                return self.last_stable_board[side].copy()
            return board

        # 安定中: 現 board を stable として記録
        self.last_stable_board[side] = board.copy()
        return board


__all__ = [
    "ChainAnimationDetector",
    "LUMA_DIFF_THRESHOLD",
    "MAX_HOLD_FRAMES",
]
