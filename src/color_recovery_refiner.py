"""W11-A: CNN が EM 判定したセルから「ぷよ取り逃し」を平均色マッチで復活。

背景:
    W10 の BG-EM/ScorePhysics は「色 → EM」方向の補正で、本物の puyo を EM に
    潰してしまった。
    本クラスは **逆方向**: 「EM → 色」のみ補正、CNN が見逃した puyo を平均色で
    復活。false negative (ぷよがあるのに EM 判定) のみを直す。

戦略 (W11-A 改訂版):
    既存 HSV ColorClassifier は厳格な rule (色相+彩度+明度) を持つが、
    HybridClassifier は CNN 高確信時に HSV を無視する。本クラスは CNN が EM
    判定したセルでだけ HSV を再評価し、HSV が明確に色判定すれば復活させる。

    1. CNN が EM のセルのみが補正対象
    2. HSV ColorClassifier (既存 rule) で再分類
    3. HSV が EM 以外を返したらそれを採用
    4. (オプション) CentroidClassifier の確認も追加可能

これにより:
    - false negative 減少 (見逃したぷよを HSV で救う)
    - false positive は増えない (色付き判定は CNN を信頼)
    - HSV rule は誤検出が少ない厳格な rule なので安全
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, COLOR_EMPTY, COLOR_OJAMA,
    COLOR_UNKNOWN, HIDDEN_ROWS, Board,
)
from src.centroid_classifier import CentroidClassifier
from src.image_reader import (
    BoardRegion, DEFAULT_P1_REGION, DEFAULT_P2_REGION,
)


# 中心 80% を採用 (背景影響低減、bg_empty_detector と同じ)
def _patch_inner(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    return bgr[
        int(h * 0.1):int(h * 0.9),
        int(w * 0.1):int(w * 0.9),
    ] if bgr.size else bgr


def _saturation_mean(bgr: np.ndarray) -> float:
    """中心 80% パッチの HSV S 平均。"""
    if bgr.size == 0:
        return 0.0
    inner = _patch_inner(bgr)
    if inner.size == 0:
        inner = bgr
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


@dataclass
class ColorRecoveryRefiner:
    """EM→色 方向のみの補正。色付き cell には触らない。

    HSV ColorClassifier (既存 rule) を主、CentroidClassifier を補助に。
    両方が同じ色 (EM 以外) を出すときのみ復活 (二重確認で安全)。
    """

    centroid: CentroidClassifier | None = None
    hsv_classifier: object | None = None  # ColorClassifier
    # saturation がこの値未満なら絶対 EM とみなしリカバリ skip
    min_saturation: float = 80.0
    # centroid との L2 距離がこの値以下で確認採用 (推定値)
    max_centroid_distance: float = 18.0
    # require centroid agreement: HSV と centroid が一致しないと復活しない
    require_centroid_agree: bool = True
    # 統計
    n_recovered: int = 0

    def __post_init__(self) -> None:
        if self.hsv_classifier is None:
            from src.image_reader import ColorClassifier
            self.hsv_classifier = ColorClassifier()

    def reset_stats(self) -> None:
        self.n_recovered = 0

    def refine_cell(self, cnn_color: int, bgr_patch: np.ndarray) -> int:
        """1 cell 補正: CNN が EM のときだけ recovery を試みる。"""
        if cnn_color != COLOR_EMPTY:
            return cnn_color
        if bgr_patch is None or bgr_patch.size == 0:
            return cnn_color
        # 低彩度 = 真の EM (背景 dot pattern など)
        s = _saturation_mean(bgr_patch)
        if s < self.min_saturation:
            return cnn_color
        # HSV rule で再分類 (既存厳格 rule)
        try:
            hsv_color = int(self.hsv_classifier.classify(bgr_patch))
        except Exception:
            return cnn_color
        if hsv_color in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA):
            return cnn_color
        # centroid と一致しない場合は採用しない (二重確認)
        if self.require_centroid_agree and self.centroid is not None:
            try:
                cen_color, cen_dist = self.centroid.classify_with_distance(
                    bgr_patch,
                )
            except Exception:
                return cnn_color
            if cen_color != hsv_color:
                return cnn_color
            if cen_dist > self.max_centroid_distance:
                return cnn_color
        self.n_recovered += 1
        return hsv_color

    def refine_board(
        self, frame: np.ndarray, side: str, board: Board,
    ) -> Board:
        """1 side の board 全 cell を refine。"""
        region = (
            DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
        )
        out = board.copy()
        h, w = frame.shape[:2]
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                cur = int(out.get(row, col))
                if cur != COLOR_EMPTY:
                    continue
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))
                patch = frame[y1:y2, x1:x2]
                new_color = self.refine_cell(cur, patch)
                if new_color != cur:
                    out.set(row, col, new_color)
        return out


__all__ = ["ColorRecoveryRefiner"]
