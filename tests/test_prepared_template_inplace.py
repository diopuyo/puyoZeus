"""PreparedTemplate.scores の in-place 化は元実装 (scores_reference) と bit-identical であること。"""
from __future__ import annotations

import numpy as np
import pytest

from src.prepared_template import PreparedImage, PreparedTemplate

ROI_SHAPES = [(120, 200), (400, 720), (97, 131)]
TEMPLATE_SHAPES = [(20, 30), (60, 200), (7, 9)]


def _image(rng: np.random.Generator, shape: tuple[int, int], flat_block: bool) -> np.ndarray:
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    if flat_block:  # 分散 0 の平坦領域 (variance<=floor の分岐と denominator==0 を通す)
        image[10:60, 20:90] = 200
    return image


@pytest.mark.parametrize('roi', ROI_SHAPES)
@pytest.mark.parametrize('template_shape', TEMPLATE_SHAPES)
@pytest.mark.parametrize('flat_block', [False, True])
def test_scores_bit_identical_to_reference(roi, template_shape, flat_block) -> None:
    rng = np.random.default_rng(hash((roi, template_shape, flat_block)) % 2**32)
    image = _image(rng, roi, flat_block)
    template = np.ascontiguousarray(image[5:5+template_shape[0], 8:8+template_shape[1]])
    prepared = PreparedTemplate(template)
    new = prepared.scores(image, PreparedImage(image))
    old = prepared.scores_reference(image, PreparedImage(image))
    assert new.dtype == old.dtype == np.float32 and new.shape == old.shape
    assert np.array_equal(new, old)
    assert np.array_equal(np.isnan(new), np.isnan(old))


def test_flat_template_returns_ones_like_reference() -> None:
    image = _image(np.random.default_rng(1), (120, 200), False)
    prepared = PreparedTemplate(np.full((10, 12), 77, dtype=np.uint8))
    assert np.array_equal(prepared.scores(image), prepared.scores_reference(image))


def test_integral_tables_are_not_modified_by_scores() -> None:
    """窓和の in-place 化が、フレーム内で共有する積分画像 (次のテンプレートも使う) を壊さない。"""
    image = _image(np.random.default_rng(2), (150, 260), True)
    prepared_image = PreparedImage(image)
    template = PreparedTemplate(np.ascontiguousarray(image[3:40, 5:60]))
    before = [table.copy() for table in prepared_image.integrals]
    template.scores(image, prepared_image)
    assert all(np.array_equal(a, b) for a, b in zip(before, prepared_image.integrals))


def test_peak_decision_unchanged() -> None:
    image = _image(np.random.default_rng(3), (200, 300), False)
    template = np.ascontiguousarray(image[50:90, 100:180])
    prepared = PreparedTemplate(template)
    peak = prepared.peak(image, 0.55)
    assert peak[1] == (100, 50) and peak[0] > 0.99
