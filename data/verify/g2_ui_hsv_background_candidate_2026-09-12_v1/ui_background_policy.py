"""既存UIテンプレの横位置一致だけでHSV赤を背景へ正規化する私有候補。"""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from src.ui_mask import DEFAULT_NCC_THRESHOLD, NORM_SIZE, UI_MASK_TARGET_CELLS

RED, EMPTY = 1, 0
UI_CELL = (1, 2)


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('ui_background:' + reason)


def search_image(image: np.ndarray, region: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    require(image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8, 'image')
    cx, cy = region.cell_center(*UI_CELL)
    width, height = int(region.cell_width), int(region.cell_height)
    left, top = cx - width // 2, cy - height // 2
    require(left >= 0 and top >= 0 and left + width <= image.shape[1]
            and top + height <= image.shape[0], 'cell_bounds')
    x1, y1, x2, y2 = region.cell_sample_rect(*UI_CELL)
    nx, ny = round(width * NORM_SIZE[0] / (x2 - x1)), round(height * NORM_SIZE[1] / (y2 - y1))
    require(nx >= NORM_SIZE[0] and ny >= NORM_SIZE[1], 'search_geometry')
    patch = image[top:top + height, left:left + width]
    gray = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (nx, ny), interpolation=cv2.INTER_AREA)
    return gray, (width, nx, x1 - left, ny, y1 - top, height)


def horizontal_match(image: np.ndarray, region: Any, matcher: Any) -> dict:
    require(UI_CELL in UI_MASK_TARGET_CELLS and bool(matcher._templates), 'templates_or_target')
    require(matcher._threshold == DEFAULT_NCC_THRESHOLD, 'original_threshold')
    gray, (width, nx, sample_left, ny, sample_top, height) = search_image(image, region)
    center = round(sample_top * ny / height)
    candidates = []
    for name, template in matcher._templates.items():
        require(template.shape == (NORM_SIZE[1], NORM_SIZE[0]), 'template_shape')
        scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        require(0 <= center < scores.shape[0], 'vertical_sample_origin')
        _, score, _, (dx, _) = cv2.minMaxLoc(scores[center:center + 1])
        require(np.isfinite(score), 'finite_score')
        candidates.append((score, name, dx))
    score, name, dx = max(candidates)
    return dict(score=score, template=name, threshold=float(matcher._threshold),
                horizontal_offset=dx * width / nx - sample_left,
                vertical_offset=center * height / ny - sample_top,
                is_ui=score >= matcher._threshold)


def normalize(board: Any, image: np.ndarray, region: Any, matcher: Any, *, stable: bool) -> tuple[Any, dict]:
    report = dict(changed=False, semantic_UI_evidence=True,
                  independent_sensor_agreement=False, quality_gate_clear=False)
    if stable is not True or board is None:
        return board, report | dict(reason='not_STABLE_or_missing_HSV')
    if board.get(*UI_CELL) != RED:
        return board, report | dict(reason='not_red')
    evidence = horizontal_match(image, region, matcher)
    if not evidence['is_ui']:
        return board, report | dict(reason='no_UI_match', evidence=evidence)
    copied = board.copy()
    require(copied is not board, 'copy_identity')
    copied.set(*UI_CELL, EMPTY)
    return copied, report | dict(changed=True, reason='red_UI_background', evidence=evidence)
