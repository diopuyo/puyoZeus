"""計測器の人工対照。映像の物理進行ラベルではない。"""
from pathlib import Path

import numpy as np
import pytest

import observables as O

SYNTHETIC_SIZE = 128
TEST_ROI = (30, 95, 30, 95)
SHIFT_X, SHIFT_Y = 3, -4
NUMERICAL_TOLERANCE = 0.2


def texture() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (SYNTHETIC_SIZE, SYNTHETIC_SIZE), dtype=np.uint8)


def test_scaled_roi_preserves_original_overlap() -> None:
    top = O.scaled_roi((162, 237, 710, 785))
    bottom = O.scaled_roi((222, 297, 710, 785))
    assert top == (108, 158, 473, 524)
    assert bottom == (148, 198, 473, 524)
    assert O.inside(np.asarray([500, 150]), top)
    assert O.inside(np.asarray([500, 150]), bottom)


def test_same_synthetic_image_has_zero_motion() -> None:
    O.configure()
    before = texture()
    points = O.starting_points(before, TEST_ROI)
    records = O.trace_points(before, before.copy(), points)
    summary = O.summarize(records, TEST_ROI, 2)
    assert summary['valid_count'] > 0
    assert summary['dx_px']['mean'] == 0.0
    assert summary['dy_px']['mean'] == 0.0
    assert summary['fb_distance_px']['mean'] == 0.0
    assert summary['retained_fraction_of_valid'] == 1.0


def test_known_synthetic_translation_direction() -> None:
    O.configure()
    before, after = texture(), np.zeros((SYNTHETIC_SIZE, SYNTHETIC_SIZE), dtype=np.uint8)
    after[:SHIFT_Y, SHIFT_X:] = before[-SHIFT_Y:, :-SHIFT_X]
    records = O.trace_points(before, after, O.starting_points(before, TEST_ROI))
    summary = O.summarize(records, TEST_ROI, 2)
    assert summary['valid_count'] > 0
    assert summary['dx_px']['quantiles']['0.5'] == pytest.approx(SHIFT_X, abs=NUMERICAL_TOLERANCE)
    assert summary['dy_px']['quantiles']['0.5'] == pytest.approx(SHIFT_Y, abs=NUMERICAL_TOLERANCE)
    assert summary['dy_px_per_source_frame']['quantiles']['0.5'] == pytest.approx(
        SHIFT_Y / 2, abs=NUMERICAL_TOLERANCE)


def test_featureless_image_is_unavailable_not_static_certificate() -> None:
    before = np.zeros((SYNTHETIC_SIZE, SYNTHETIC_SIZE), dtype=np.uint8)
    points = O.starting_points(before, TEST_ROI)
    summary = O.summarize(O.trace_points(before, before, points), TEST_ROI, 2)
    assert summary['seed_count'] == summary['valid_count'] == 0
    assert summary['retained_fraction_of_valid'] is None
    assert summary['dy_px']['mean'] is None


def test_changed_input_hash_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / 'synthetic.bin'
    path.write_bytes(b'synthetic changed')
    with pytest.raises(ValueError, match='input_sha'):
        O.checked_read(path, O.sha(b'synthetic original'))
