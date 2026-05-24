"""ROI 自動キャリブレーション (Z-3K / D)。

未知動画 (別大会、別 UI 配置) で 1080p ハードコード ROI が
ずれている場合への対応。試合開始 frame の UI マーカー
(NEXT 枠の暗い縁、フィールド外周) を検出して ROI offset を auto 算出。

設計:
    - フィールド領域は暗背景 + 規則的なグリッド線
    - 1P フィールドは画面左寄り (x ~ 280-660 1080p)
    - 2P フィールドは画面右寄り (x ~ 1260-1640 1080p)
    - エッジ検出 + 垂直線群の中央座標で ROI 中心を推定

シンプル実装:
    1. frame をグレースケール化、Canny エッジ
    2. 各 region 期待位置の周辺 (±50px) で垂直線群を Hough で検出
    3. 検出された垂直線の中央 ↔ DEFAULT region 中央のオフセットを出力
    4. オフセットが過大 (>30px) なら未検出扱いで default 維持
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 検出許容範囲 (default ROI 中央から ±N px の探索)
SEARCH_RADIUS: int = 40
# 検出失敗判定 (offset がこれ以上なら信頼できない、default 維持)
MAX_VALID_OFFSET: int = 30
# Canny 閾値
CANNY_LOW: int = 50
CANNY_HIGH: int = 150
# Hough 直線検出 minLineLength
HOUGH_MIN_LINE_LENGTH: int = 200
HOUGH_MAX_LINE_GAP: int = 10


@dataclass(frozen=True)
class RoiCalibration:
    """1P/2P それぞれの ROI offset (dx, dy)。"""
    p1_offset: tuple[int, int]
    p2_offset: tuple[int, int]
    confidence: float  # 0..1、検出信頼度


def detect_roi_offsets(
    frame: np.ndarray,
    default_p1_x: int = 282,
    default_p1_y: int = 160,
    default_p2_x: int = 1258,
    default_p2_y: int = 160,
    region_w: int = 384,
    region_h: int = 720,
) -> RoiCalibration:
    """frame から 1P/2P フィールド ROI のオフセットを検出。

    Args:
        frame: 1080p BGR 画像 (試合開始の安定 frame 推奨)
        default_*_*: ImageReader.DEFAULT_*_REGION の値
        region_w/h: フィールド幅/高さ

    Returns:
        RoiCalibration: P1/P2 それぞれの offset と信頼度
    """
    if frame.shape[:2] != (1080, 1920):
        return RoiCalibration((0, 0), (0, 0), 0.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    # 1P / 2P それぞれの探索領域
    p1_offset = _detect_field_offset(
        edges, default_p1_x, default_p1_y, region_w, region_h,
    )
    p2_offset = _detect_field_offset(
        edges, default_p2_x, default_p2_y, region_w, region_h,
    )
    # 信頼度: offset の絶対値が小さいほど高い
    p1_conf = 1.0 - min(1.0, max(abs(p1_offset[0]), abs(p1_offset[1])) / 30.0)
    p2_conf = 1.0 - min(1.0, max(abs(p2_offset[0]), abs(p2_offset[1])) / 30.0)
    confidence = (p1_conf + p2_conf) / 2.0
    return RoiCalibration(
        p1_offset=p1_offset,
        p2_offset=p2_offset,
        confidence=confidence,
    )


def _detect_field_offset(
    edges: np.ndarray,
    default_x: int,
    default_y: int,
    region_w: int,
    region_h: int,
) -> tuple[int, int]:
    """フィールド領域の左端・上端を Hough 直線検出で特定し offset を算出。

    Args:
        edges: Canny エッジ画像
        default_x/y: デフォルト ROI 左上
        region_w/h: 領域幅/高さ

    Returns:
        (dx, dy): default からの offset (-MAX..+MAX)。検出失敗で (0, 0)。
    """
    h, w = edges.shape[:2]
    # 探索領域 (default 中央から SEARCH_RADIUS の周辺)
    search_x0 = max(0, default_x - SEARCH_RADIUS)
    search_x1 = min(w, default_x + region_w + SEARCH_RADIUS)
    search_y0 = max(0, default_y - SEARCH_RADIUS)
    search_y1 = min(h, default_y + region_h + SEARCH_RADIUS)
    if search_x1 <= search_x0 or search_y1 <= search_y0:
        return (0, 0)
    roi_edges = edges[search_y0:search_y1, search_x0:search_x1]
    # HoughLinesP で水平/垂直線を検出
    lines = cv2.HoughLinesP(
        roi_edges, 1, np.pi / 180, threshold=80,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    if lines is None:
        return (0, 0)
    # 垂直線 (x 座標が一定) と水平線 (y 座標が一定) を分類
    vertical_xs: list[int] = []
    horizontal_ys: list[int] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > dx * 5 and dy > 100:
            vertical_xs.append((x1 + x2) // 2 + search_x0)
        elif dx > dy * 5 and dx > 100:
            horizontal_ys.append((y1 + y2) // 2 + search_y0)
    # 左端 (default_x に最も近い垂直線) を検出
    if not vertical_xs or not horizontal_ys:
        return (0, 0)
    # default_x に最も近い vertical line
    closest_x = min(vertical_xs, key=lambda x: abs(x - default_x))
    closest_y = min(horizontal_ys, key=lambda y: abs(y - default_y))
    dx = closest_x - default_x
    dy = closest_y - default_y
    # 異常値 (探索範囲外) は default 維持
    if abs(dx) > MAX_VALID_OFFSET or abs(dy) > MAX_VALID_OFFSET:
        return (0, 0)
    return (int(dx), int(dy))


__all__ = [
    "RoiCalibration",
    "detect_roi_offsets",
    "SEARCH_RADIUS",
    "MAX_VALID_OFFSET",
]
