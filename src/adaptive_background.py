"""継続的背景フィンガープリント学習 (Phase T v2 サイクル T-v2-C)。

試合開始時の 1 フレームのみで取得した BackgroundFingerprint は、連鎖アニメ・
キャラクター動き・スコア表示の変化に追随できない。これが背景色干渉の原因。

本モジュールは「現フレームの HSV が既存背景に近いセルだけ」移動平均で
背景値を更新することで、空セルが続いている所の背景は徐々に変化に追随し、
ぷよが置かれているセルは背景値を固定する (距離大で更新をスキップする)。

利用例:
    base = BackgroundFingerprint.capture(frame, x, y, w, h)
    adaptive = AdaptiveBackgroundFingerprint(base)
    # 各フレームで...
    adaptive.update(frame, region_x, region_y, region_w, region_h)
    # 推論時には adaptive.to_fingerprint() で BackgroundFingerprint 互換オブジェクトを取得
"""
from __future__ import annotations

import cv2
import numpy as np

from src.background_fingerprint import (
    BackgroundFingerprint,
    CellFingerprint,
    H_WEIGHT,
    S_WEIGHT,
    V_WEIGHT,
)
from src.board import BOARD_COLS, VISIBLE_ROWS

# ぷよあり/なしを推定する HSV 距離閾値:
# 距離 < UPDATE_DISTANCE_MAX のセルだけ背景更新 (空セル候補)
DEFAULT_UPDATE_DISTANCE_MAX: float = 35.0

# 移動平均の学習率 (1 フレーム分の重み)。低いほど安定、高いほど追随性高
DEFAULT_LEARNING_RATE: float = 0.05

# サンプリング比率 (cell width/height のうち中央何 % をサンプル)
DEFAULT_SAMPLE_RATIO: float = 0.6


def _hsv_distance(
    h1: float, s1: float, v1: float,
    h2: float, s2: float, v2: float,
) -> float:
    """重み付き HSV 距離 (H は循環)。"""
    dh = abs(h1 - h2)
    dh = min(dh, 180.0 - dh)
    return H_WEIGHT * dh + S_WEIGHT * abs(s1 - s2) + V_WEIGHT * abs(v1 - v2)


class AdaptiveBackgroundFingerprint:
    """継続的に背景値を更新する FP。"""

    def __init__(
        self,
        base: BackgroundFingerprint,
        update_distance_max: float = DEFAULT_UPDATE_DISTANCE_MAX,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    ) -> None:
        # 浮動小数で保持し、移動平均を高精度に
        self._cells: list[list[list[float]]] = [
            [[float(c.h), float(c.s), float(c.v)] for c in row]
            for row in base.cells
        ]
        self._update_dist_max = float(update_distance_max)
        self._lr = float(learning_rate)
        self._sample_ratio = float(sample_ratio)

    def cell_at(self, row_visible: int, col: int) -> CellFingerprint:
        """可視行 + 列でセル取得 (BackgroundFingerprint 互換 API)。"""
        if not (0 <= row_visible < VISIBLE_ROWS and 0 <= col < BOARD_COLS):
            return CellFingerprint(0, 0, 0)
        h, s, v = self._cells[row_visible][col]
        return CellFingerprint(int(round(h)), int(round(s)), int(round(v)))

    @property
    def cells(self) -> tuple[tuple[CellFingerprint, ...], ...]:
        """BackgroundFingerprint.cells と同形式で返す (互換 API)。"""
        return tuple(
            tuple(
                CellFingerprint(int(round(h)), int(round(s)), int(round(v)))
                for h, s, v in row
            )
            for row in self._cells
        )

    def to_fingerprint(self) -> BackgroundFingerprint:
        """現在の状態を不変 BackgroundFingerprint に変換。"""
        return BackgroundFingerprint(cells=self.cells)

    def update(
        self,
        frame: np.ndarray,
        region_x: int,
        region_y: int,
        region_w: int,
        region_h: int,
    ) -> int:
        """フレームから各セル HSV を取得し、距離が小さいセルだけ背景を更新。

        Returns:
            更新されたセル数 (デバッグ用)。
        """
        if frame is None or frame.ndim != 3:
            return 0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cell_w = region_w / BOARD_COLS
        cell_h = region_h / VISIBLE_ROWS
        updated = 0
        for r in range(VISIBLE_ROWS):
            for c in range(BOARD_COLS):
                cx = int(region_x + (c + 0.5) * cell_w)
                cy = int(region_y + (r + 0.5) * cell_h)
                half_w = max(1, int(cell_w * self._sample_ratio / 2))
                half_h = max(1, int(cell_h * self._sample_ratio / 2))
                x1 = max(0, cx - half_w)
                y1 = max(0, cy - half_h)
                x2 = min(hsv.shape[1], cx + half_w)
                y2 = min(hsv.shape[0], cy + half_h)
                patch = hsv[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                h_med = float(np.median(patch[:, :, 0]))
                s_med = float(np.median(patch[:, :, 1]))
                v_med = float(np.median(patch[:, :, 2]))
                # 既存背景値との距離 (近ければ空セル候補)
                bh, bs, bv = self._cells[r][c]
                dist = _hsv_distance(h_med, s_med, v_med, bh, bs, bv)
                if dist > self._update_dist_max:
                    # ぷよが置かれている可能性が高い → 更新スキップ
                    continue
                # 移動平均で背景を緩やかに更新
                lr = self._lr
                self._cells[r][c] = [
                    bh * (1.0 - lr) + h_med * lr,
                    bs * (1.0 - lr) + s_med * lr,
                    bv * (1.0 - lr) + v_med * lr,
                ]
                updated += 1
        return updated


__all__ = [
    "AdaptiveBackgroundFingerprint",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_SAMPLE_RATIO",
    "DEFAULT_UPDATE_DISTANCE_MAX",
]
