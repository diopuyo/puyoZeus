"""BoardRegion 自動 calibration (cycle 71r、 案 A、 2026-05-13).

動画ごとに盤面領域の x/y 座標が微妙にずれる問題への対策.
起動時の試合中 frame で BoardRegion 周辺 ±N px の grid search を実施、
認識 cell 数が最大の座標を採用する.

使い方:
    from src.auto_calibration import auto_calibrate_board_regions
    p1_adj, p2_adj = auto_calibrate_board_regions(
        frame_1080p, DEFAULT_P1_REGION, DEFAULT_P2_REGION, classifier,
    )
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, HIDDEN_ROWS,
)
from src.image_reader import BoardRegion, ColorClassifier


# grid search 範囲 (= ±N px、 step px ごと)
DEFAULT_X_RANGE: int = 8
DEFAULT_Y_RANGE: int = 8
DEFAULT_STEP: int = 2


def _score_region(
    frame_1080p: np.ndarray, region: BoardRegion,
    classifier: ColorClassifier,
) -> int:
    """region で frame の cell を認識して、 puyo 認識 cell 数を返す.

    多数の cell が puyo として認識される (= EMPTY/UNKNOWN 以外) ほど、
    region が「正しい位置」 にある shoulder. ただし試合中盤の安定 frame では
    puyo 多めなので、 適切な指標になる.
    """
    count = 0
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame_1080p.shape[1], int(x2))
            y2 = min(frame_1080p.shape[0], int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            patch = frame_1080p[y1:y2, x1:x2]
            color = classifier.classify(patch)
            if color not in (COLOR_EMPTY, COLOR_UNKNOWN):
                count += 1
    return count


def auto_calibrate_region(
    frame_1080p: np.ndarray, base_region: BoardRegion,
    classifier: ColorClassifier,
    x_range: int = DEFAULT_X_RANGE,
    y_range: int = DEFAULT_Y_RANGE,
    step: int = DEFAULT_STEP,
    min_score_gain_ratio: float = 1.1,
) -> BoardRegion:
    """base_region 周辺の grid search で最適 region を見つける.

    起動時の試合中 frame (= puyo 多めの STABLE frame) で実行する想定.

    cycle 71t (2026-05-13 副作用対策): min_score_gain_ratio を導入.
    grid search の候補 score が base の min_score_gain_ratio 倍以上なら採用、
    そうでなければ base 維持. これで「**誤認識で cell 数増える座標**」 を
    候補として採用してしまう副作用 (= 背景誤認増加) を抑止.

    Args:
        frame_1080p: 1920x1080 BGR frame (= 試合中の代表 frame).
        base_region: 既存の BoardRegion (= DEFAULT_P1_REGION 等).
        classifier: ColorClassifier (= HSV 系で OK).
        x_range / y_range / step: grid search 範囲とステップ.
        min_score_gain_ratio: 候補採用の最低 score 比率 (= base 比).

    Returns:
        最適化された BoardRegion. 大きい改善が無ければ base_region.
    """
    best_region = base_region
    base_score = _score_region(frame_1080p, base_region, classifier)
    best_score = base_score
    for dy in range(-y_range, y_range + 1, step):
        for dx in range(-x_range, x_range + 1, step):
            if dx == 0 and dy == 0:
                continue
            candidate = replace(
                base_region,
                x=base_region.x + dx,
                y=base_region.y + dy,
            )
            score = _score_region(frame_1080p, candidate, classifier)
            if score > best_score:
                best_score = score
                best_region = candidate
    # cycle 71t 副作用対策: 大きい改善 (= ratio 以上) なしなら base 維持
    if base_score == 0 or best_score < base_score * min_score_gain_ratio:
        return base_region
    return best_region


def auto_calibrate_board_regions(
    frame_1080p: np.ndarray,
    base_p1: BoardRegion,
    base_p2: BoardRegion,
    classifier: ColorClassifier,
) -> tuple[BoardRegion, BoardRegion]:
    """1P と 2P の region を独立に auto calibrate."""
    p1_adj = auto_calibrate_region(frame_1080p, base_p1, classifier)
    p2_adj = auto_calibrate_region(frame_1080p, base_p2, classifier)
    return p1_adj, p2_adj
