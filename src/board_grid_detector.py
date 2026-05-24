"""
盤面グリッド自動検出モジュール (R-10 対応)

固定 ROI (`BoardRegion`) は 1920×1080 想定で破綻するため、
Canny + Hough Line + 交点クラスタリングで 6 列 × 13 行のグリッド四隅を
自動推定する。

参考手法 (出典のみ参照):
    - daniel-bandstra/watchGo : Canny + Hough Line で碁盤検出
    - match-3 系 OpenCV Q&A   : Adaptive thresholding によるグリッド線抽出
    - chess board recognition  : corner detection + perspective transform

設計方針:
    - 既存 ROI 経路 (image_reader.DEFAULT_P1_REGION 等) は破壊しない。
    - opt-in: 呼出側が必要に応じて `BoardGridCache.detect_with_cache` を呼ぶ。
    - 失敗時は `None` を返し、呼出側で固定 ROI fallback。
    - 1 動画 1 回検出 → キャッシュ。毎 frame 検出は不要。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, VISIBLE_ROWS
from src.image_reader import BoardRegion, DEFAULT_P1_REGION, DEFAULT_P2_REGION

# ============================
# 定数定義
# ============================

# Canny edge 閾値 (low, high)
CANNY_LOW: int = 50
CANNY_HIGH: int = 150

# Hough Line 検出パラメータ
HOUGH_RHO: float = 1.0
HOUGH_THETA: float = np.pi / 180.0
HOUGH_THRESHOLD: int = 80
HOUGH_MIN_LINE_LENGTH: int = 60
HOUGH_MAX_LINE_GAP: int = 10

# 水平・垂直判定の角度許容 (rad)
ANGLE_TOLERANCE: float = np.pi / 36.0  # 5 度

# Line clustering 距離閾値 (px) — 同じ行/列の line とみなす最大距離
LINE_CLUSTER_DISTANCE: int = 15

# Aspect 比検証 (盤面 6 列 × 12 可視行 → cell が square 想定)
EXPECTED_CELL_ASPECT: float = 1.0
CELL_ASPECT_TOLERANCE: float = 0.4

# 信頼度の最小閾値 (これ未満は失敗扱い)
MIN_CONFIDENCE: float = 0.3

# グリッド検出に必要な水平/垂直 line 数の下限
MIN_HORIZONTAL_LINES: int = 4
MIN_VERTICAL_LINES: int = 3


# ============================
# データクラス
# ============================


@dataclass
class GridDetection:
    """
    検出された 6 × 13 (可視 12 行) のグリッド四隅。

    座標系は frame ピクセル。隠し段 (row 0) は画面外のため、
    bottom_left/right は可視 12 行下端、top_left/right は可視領域上端
    (= row 1 の上端) を指す。
    """

    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    confidence: float


# ============================
# 内部ヘルパ
# ============================


def _segments_to_lines(
    segments: np.ndarray,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    """
    HoughLinesP の出力を水平 / 垂直に分類して返す。

    Returns:
        (horizontal_lines, vertical_lines) の tuple。各 line は (x1,y1,x2,y2)。
    """
    horizontals: list[tuple[float, float, float, float]] = []
    verticals: list[tuple[float, float, float, float]] = []
    for seg in segments:
        x1, y1, x2, y2 = seg[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        angle = np.arctan2(abs(dy), abs(dx))
        if angle < ANGLE_TOLERANCE:
            horizontals.append((float(x1), float(y1), float(x2), float(y2)))
        elif angle > (np.pi / 2.0 - ANGLE_TOLERANCE):
            verticals.append((float(x1), float(y1), float(x2), float(y2)))
    return horizontals, verticals


def _cluster_positions(
    positions: list[float],
    distance: int = LINE_CLUSTER_DISTANCE,
) -> list[float]:
    """
    1D 座標列を距離 `distance` 以内でクラスタリングし、
    各クラスタ平均値を昇順で返す (簡易 single-link)。
    """
    if not positions:
        return []
    sorted_pos = sorted(positions)
    clusters: list[list[float]] = [[sorted_pos[0]]]
    for p in sorted_pos[1:]:
        if p - clusters[-1][-1] <= distance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [float(np.mean(c)) for c in clusters]


def _line_axis_position(
    line: tuple[float, float, float, float], horizontal: bool
) -> float:
    """水平 line なら y 平均、垂直 line なら x 平均を返す。"""
    x1, y1, x2, y2 = line
    return (y1 + y2) / 2.0 if horizontal else (x1 + x2) / 2.0


def _bounding_quad_from_lines(
    h_positions: list[float], v_positions: list[float]
) -> Optional[tuple[float, float, float, float]]:
    """
    水平 / 垂直 line 位置から bounding box (x_min, y_min, x_max, y_max) を返す。

    検出失敗時 (line 数不足) None。
    """
    if len(h_positions) < MIN_HORIZONTAL_LINES or len(v_positions) < MIN_VERTICAL_LINES:
        return None
    return (
        float(min(v_positions)),
        float(min(h_positions)),
        float(max(v_positions)),
        float(max(h_positions)),
    )


def _confidence_from_box(
    box: tuple[float, float, float, float],
    h_count: int,
    v_count: int,
) -> float:
    """
    検出 box から信頼度を計算 (cell aspect が想定通りなら高い)。

    - cell aspect (cell_height / cell_width) が 1.0 に近いほど高信頼
    - h/v line 数が多いほど高信頼
    """
    x_min, y_min, x_max, y_max = box
    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)
    cell_w = width / BOARD_COLS
    cell_h = height / VISIBLE_ROWS
    aspect = cell_h / cell_w if cell_w > 0 else 0.0
    aspect_score = max(
        0.0,
        1.0 - abs(aspect - EXPECTED_CELL_ASPECT) / CELL_ASPECT_TOLERANCE,
    )
    line_score = min(1.0, (h_count + v_count) / float(BOARD_ROWS + BOARD_COLS))
    return float(0.6 * aspect_score + 0.4 * line_score)


# ============================
# 公開 API
# ============================


class BoardGridDetector:
    """
    盤面グリッド自動検出器。

    `detect(frame_bgr)` で 1 frame からグリッド四隅を推定。
    内部で Canny → HoughLinesP → 水平/垂直 clustering → 外接矩形抽出。
    """

    def __init__(
        self,
        canny_low: int = CANNY_LOW,
        canny_high: int = CANNY_HIGH,
        cluster_distance: int = LINE_CLUSTER_DISTANCE,
    ) -> None:
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._cluster_distance = cluster_distance

    def detect(self, frame_bgr: np.ndarray) -> Optional[GridDetection]:
        """
        frame からグリッド四隅を検出。失敗時 None。

        Args:
            frame_bgr: BGR 画像 (H, W, 3)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # 単一色フレームなら edge が皆無 → None
        if int(gray.std()) < 3:
            return None
        edges = cv2.Canny(gray, self._canny_low, self._canny_high)
        segments = cv2.HoughLinesP(
            edges,
            HOUGH_RHO,
            HOUGH_THETA,
            HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )
        if segments is None:
            return None
        h_lines, v_lines = _segments_to_lines(segments)
        h_positions = _cluster_positions(
            [_line_axis_position(line, True) for line in h_lines],
            self._cluster_distance,
        )
        v_positions = _cluster_positions(
            [_line_axis_position(line, False) for line in v_lines],
            self._cluster_distance,
        )
        box = _bounding_quad_from_lines(h_positions, v_positions)
        if box is None:
            return None
        confidence = _confidence_from_box(box, len(h_positions), len(v_positions))
        if confidence < MIN_CONFIDENCE:
            return None
        x_min, y_min, x_max, y_max = box
        return GridDetection(
            top_left=(x_min, y_min),
            top_right=(x_max, y_min),
            bottom_left=(x_min, y_max),
            bottom_right=(x_max, y_max),
            confidence=confidence,
        )


def cells_from_grid(grid: GridDetection) -> np.ndarray:
    """
    検出 grid から各 cell の bbox 配列を生成。

    Returns:
        shape (BOARD_ROWS, BOARD_COLS, 4) の int array。
        各要素は (x_min, y_min, x_max, y_max)。
        row 0 (隠し段) は可視領域上端の上方に推定座標を置く
        (画面外で実データ無いことが多いが座標系は連続)。
    """
    x_min, y_min = grid.top_left
    x_max, y_max = grid.bottom_right
    visible_height = max(1.0, y_max - y_min)
    cell_w = (x_max - x_min) / float(BOARD_COLS)
    cell_h = visible_height / float(VISIBLE_ROWS)
    out = np.zeros((BOARD_ROWS, BOARD_COLS, 4), dtype=np.int32)
    for row in range(BOARD_ROWS):
        # 隠し段 row 0 は可視領域上端の 1 セル分上方へ
        visible_row = row - (BOARD_ROWS - VISIBLE_ROWS)
        cy_top = y_min + visible_row * cell_h
        cy_bot = cy_top + cell_h
        for col in range(BOARD_COLS):
            cx_left = x_min + col * cell_w
            cx_right = cx_left + cell_w
            out[row, col] = [
                int(round(cx_left)),
                int(round(cy_top)),
                int(round(cx_right)),
                int(round(cy_bot)),
            ]
    return out


def grid_to_board_region(grid: Optional[GridDetection]) -> Optional[BoardRegion]:
    """
    `GridDetection` を既存 `BoardRegion` に変換 (互換 adapter)。

    grid is None → None (呼出側で fallback)
    """
    if grid is None:
        return None
    x_min, y_min = grid.top_left
    x_max, y_max = grid.bottom_right
    width = max(1, int(round(x_max - x_min)))
    height = max(1, int(round(y_max - y_min)))
    return BoardRegion(
        x=int(round(x_min)),
        y=int(round(y_min)),
        width=width,
        height=height,
    )


def grid_or_default_region(
    grid: Optional[GridDetection], player: int = 1
) -> BoardRegion:
    """
    grid 検出があればそれを `BoardRegion` 化、無ければ player に応じた
    default を返す (1=P1, 2=P2)。
    """
    region = grid_to_board_region(grid)
    if region is not None:
        return region
    return DEFAULT_P1_REGION if player == 1 else DEFAULT_P2_REGION


class BoardGridCache:
    """
    per-video キャッシュ付きグリッド検出器。

    `detect_with_cache(video_id, frame)` で同じ video_id は 1 回のみ実検出。
    成功した検出 (None でない) のみキャッシュ。失敗時は再試行可能。
    """

    def __init__(self, detector: Optional[BoardGridDetector] = None) -> None:
        self._detector = detector or BoardGridDetector()
        self._cache: dict[str, GridDetection] = {}

    def detect_with_cache(
        self, video_id: str, frame_bgr: np.ndarray
    ) -> Optional[GridDetection]:
        """
        video_id ごとに 1 回検出してキャッシュ。
        既ヒットなら frame は使わず cache を返す。
        """
        cached = self._cache.get(video_id)
        if cached is not None:
            return cached
        detected = self._detector.detect(frame_bgr)
        if detected is not None:
            self._cache[video_id] = detected
        return detected

    def clear(self) -> None:
        """キャッシュ全消去 (主にテスト用)。"""
        self._cache.clear()

    def has(self, video_id: str) -> bool:
        """指定 video_id がキャッシュされているか。"""
        return video_id in self._cache
