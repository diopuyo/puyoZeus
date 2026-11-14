"""×印の中央が発光した場合と、ぷよ・一様な背景の区別を検証する。"""
from pathlib import Path

import cv2
import joblib
import numpy as np
import pytest

from src.image_free_ui_mask import ImageFreeUiMaskMatcher
from src.image_reader import ImageReader, DEFAULT_P1_REGION
from src.board import Board

SIZE = (60, 64)
CENTER_SLICE = (slice(15, 45), slice(16, 48))


class EmptyEstimator:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), '')


class ConfidenceEstimator:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), 0.68)


def matcher(tmp_path: Path) -> ImageFreeUiMaskMatcher:
    path = tmp_path / 'mask.joblib'
    joblib.dump(dict(schema=1, labels=EmptyEstimator(), confidence=ConfidenceEstimator()), path)
    return ImageFreeUiMaskMatcher(path)


def synthetic_shape(cross: bool) -> np.ndarray:
    image = np.zeros((*SIZE, 3), dtype=np.uint8)
    color = (0, 0, 220)
    if cross:
        cv2.line(image, (15, 13), (48, 46), color, 20)
        cv2.line(image, (48, 13), (15, 46), color, 20)
    else:
        cv2.circle(image, (32, 30), 27, color, -1)
    return image


@pytest.mark.parametrize('halo', [False, True])
def test_outer_shape_recovers_cross(tmp_path: Path, halo: bool) -> None:
    reader = matcher(tmp_path)
    context = synthetic_shape(True)
    if halo:
        context = cv2.addWeighted(context, 0.4, np.full_like(context, (0, 220, 220)), 0.6, 0)
    patch = context[CENTER_SLICE]
    assert not reader.match(patch).is_ui
    assert reader.match(patch, context=context).is_ui


@pytest.mark.parametrize('kind', ['circle', 'flat', 'empty'])
def test_context_does_not_erase_non_cross(tmp_path: Path, kind: str) -> None:
    reader = matcher(tmp_path)
    context = synthetic_shape(False) if kind == 'circle' else np.zeros((*SIZE, 3), dtype=np.uint8)
    if kind == 'flat':
        context[:] = (0, 220, 240)
    assert not reader.match(context[CENTER_SLICE], context=context).is_ui


def test_reader_applies_context_only_at_ui_cell(tmp_path: Path) -> None:
    reader = ImageReader.__new__(ImageReader)
    reader._ui_matcher = matcher(tmp_path)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    x1, y1, x2, y2 = DEFAULT_P1_REGION.cell_sample_rect(1, 2)
    frame[y1-15:y2+15, x1-16:x2+16] = synthetic_shape(True)
    board = Board()
    board.set(1, 2, 1)
    board.set(1, 3, 1)
    reader._apply_image_free_ui_context(frame, DEFAULT_P1_REGION, board)
    assert board.get(1, 2) == 0
    assert board.get(1, 3) == 1


def test_reader_with_legacy_matcher_leaves_board_unchanged() -> None:
    reader = ImageReader.__new__(ImageReader)
    reader._ui_matcher = None
    board = Board()
    board.set(1, 2, 1)
    reader._apply_image_free_ui_context(np.zeros((1, 1, 3), dtype=np.uint8), DEFAULT_P1_REGION, board)
    assert board.get(1, 2) == 1
