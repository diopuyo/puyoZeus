"""ぷよぷよ盤面の ROI 自動キャリブレーション (Phase U-1)。

試合中フレームから 1P / 2P の盤面外枠 (黒枠) を検出し、BoardRegion を
動的生成する。720p / 1080p / 配信解像度のばらつきに頑健。

検出ロジック:
    1. グレースケール化 + Canny エッジ
    2. HoughLinesP で縦/横の直線群を抽出
    3. 画面中央 (x=screen_w/2) で 1P 側 / 2P 側に分離
    4. 各側で「縦線の左右端」「横線の上下端」を採って矩形候補を確定
    5. 矩形の縦横比が想定 (1:1.85, 6×12 セル + 黒枠) と一致するか検証
    6. 妥当ならその矩形を BoardRegion として返す

利用例:
    p1_region, p2_region = detect_board_rois(frame)
    if p1_region is None:
        # キャリブ失敗 → デフォルト ROI を使う
        ...
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.image_reader import BoardRegion

# 想定する盤面アスペクト比 (横 : 縦) = 6 : 12 + 余白 ≒ 1 : 1.85
EXPECTED_ASPECT_MIN: float = 1.65
EXPECTED_ASPECT_MAX: float = 2.10

# 盤面サイズの妥当性チェック (px)
MIN_BOARD_WIDTH: int = 200
MAX_BOARD_WIDTH: int = 600
MIN_BOARD_HEIGHT: int = 400
MAX_BOARD_HEIGHT: int = 1000

# Canny / Hough パラメータ (盤面の比較的薄い枠も拾えるよう緩めに)
CANNY_LOW: int = 30
CANNY_HIGH: int = 100
HOUGH_THRESHOLD: int = 40
HOUGH_MIN_LINE_LEN: int = 30
HOUGH_MAX_LINE_GAP: int = 20


@dataclass(frozen=True)
class CalibrationResult:
    """ROI キャリブレーション結果。"""

    p1_region: BoardRegion | None
    p2_region: BoardRegion | None
    p1_score: float = 0.0
    p2_score: float = 0.0
    reason: str = ""


def _is_valid_board_rect(x: int, y: int, w: int, h: int) -> tuple[bool, str]:
    """矩形候補が盤面として妥当かチェック。"""
    if not (MIN_BOARD_WIDTH <= w <= MAX_BOARD_WIDTH):
        return False, f"width {w} out of range"
    if not (MIN_BOARD_HEIGHT <= h <= MAX_BOARD_HEIGHT):
        return False, f"height {h} out of range"
    aspect = h / float(w) if w > 0 else 0.0
    if not (EXPECTED_ASPECT_MIN <= aspect <= EXPECTED_ASPECT_MAX):
        return False, f"aspect {aspect:.2f} out of range"
    return True, "ok"


def _detect_lines(gray: np.ndarray) -> np.ndarray | None:
    """Canny + HoughLinesP で直線群を取得。"""
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    return lines


def _split_lines(
    lines: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """縦線と横線に分離する。"""
    verticals: list[tuple[int, int, int, int]] = []
    horizontals: list[tuple[int, int, int, int]] = []
    if lines is None:
        return verticals, horizontals
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line[0]]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > dx * 3:
            verticals.append((x1, y1, x2, y2))
        elif dx > dy * 3:
            horizontals.append((x1, y1, x2, y2))
    return verticals, horizontals


def _fit_rect_one_side(
    verticals: list[tuple[int, int, int, int]],
    horizontals: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """片側の縦/横線群から最大矩形をフィット (左端,上端,右端,下端)。"""
    if len(verticals) < 2 or len(horizontals) < 2:
        return None
    # 縦線の x 座標
    v_xs = sorted({(l[0] + l[2]) // 2 for l in verticals})
    h_ys = sorted({(l[1] + l[3]) // 2 for l in horizontals})
    if len(v_xs) < 2 or len(h_ys) < 2:
        return None
    return v_xs[0], h_ys[0], v_xs[-1], h_ys[-1]


def detect_board_rois(
    frame: np.ndarray,
    expected_height: int = 1080,
    expected_width: int = 1920,
) -> CalibrationResult:
    """フレームから 1P / 2P の盤面 ROI を自動検出。

    Args:
        frame: BGR フレーム (1080p 想定、それ以外は内部でリサイズ)。
        expected_height/width: リサイズ先解像度。

    Returns:
        CalibrationResult: p1_region / p2_region (検出失敗なら None)。
    """
    if frame is None or frame.ndim != 3:
        return CalibrationResult(None, None, reason="invalid frame")
    if frame.shape[:2] != (expected_height, expected_width):
        frame = cv2.resize(
            frame, (expected_width, expected_height),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lines = _detect_lines(gray)
    verticals, horizontals = _split_lines(lines if lines is not None else None)

    mid_x = expected_width // 2
    # 1P 側 (画面左半分の縦/横線)
    v_left = [
        l for l in verticals if (l[0] + l[2]) / 2 < mid_x
    ]
    h_left = [
        l for l in horizontals
        if (l[0] + l[2]) / 2 < mid_x
        and abs(l[2] - l[0]) < expected_width // 2
    ]
    rect_p1 = _fit_rect_one_side(v_left, h_left)

    v_right = [
        l for l in verticals if (l[0] + l[2]) / 2 >= mid_x
    ]
    h_right = [
        l for l in horizontals
        if (l[0] + l[2]) / 2 >= mid_x
        and abs(l[2] - l[0]) < expected_width // 2
    ]
    rect_p2 = _fit_rect_one_side(v_right, h_right)

    p1_region = None
    p1_score = 0.0
    if rect_p1 is not None:
        x1, y1, x2, y2 = rect_p1
        w, h = x2 - x1, y2 - y1
        ok, reason1 = _is_valid_board_rect(x1, y1, w, h)
        if ok:
            p1_region = BoardRegion(x=x1, y=y1, width=w, height=h)
            p1_score = 1.0

    p2_region = None
    p2_score = 0.0
    if rect_p2 is not None:
        x1, y1, x2, y2 = rect_p2
        w, h = x2 - x1, y2 - y1
        ok, reason2 = _is_valid_board_rect(x1, y1, w, h)
        if ok:
            p2_region = BoardRegion(x=x1, y=y1, width=w, height=h)
            p2_score = 1.0

    return CalibrationResult(
        p1_region=p1_region,
        p2_region=p2_region,
        p1_score=p1_score,
        p2_score=p2_score,
        reason="ok",
    )


__all__ = [
    "CalibrationResult",
    "EXPECTED_ASPECT_MAX",
    "EXPECTED_ASPECT_MIN",
    "MAX_BOARD_HEIGHT",
    "MAX_BOARD_WIDTH",
    "MIN_BOARD_HEIGHT",
    "MIN_BOARD_WIDTH",
    "detect_board_rois",
]
