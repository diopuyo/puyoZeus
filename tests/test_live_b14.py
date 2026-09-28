"""B14の背景統計・NCC高速化と全通知同値検査の回帰。"""
from __future__ import annotations

from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.background_fingerprint import (
    CellPatchFingerprint, PATCH_NCC_STD_MIN, PATCH_NCC_UNIFORM_FALLBACK,
    _compute_ncc_prepared,
)
from src.prepared_template import PreparedTemplate, NCC_DECISION_MARGIN
from src.ui_mask import UiMaskMatcher, NORM_SIZE


@pytest.mark.parametrize('shape', [(1, 1, 3), (16, 16, 3), (23, 24, 3)])
@pytest.mark.parametrize('value', [0, 4, 5, 128, 255])
def test_background_median_is_exact_and_reused(shape: tuple, value: int) -> None:
    background = CellPatchFingerprint(np.full(shape, value, dtype=np.float32))
    expected = float(np.median(background.patch_hsv[:, :, 2]))
    assert background.v_median == expected
    with patch('numpy.median', side_effect=AssertionError('再計算は禁止')):
        assert background.v_median == expected


def previous_ncc(a: np.ndarray, b: np.ndarray, dot: float, std: float) -> float:
    values = a.ravel().astype(np.float64)
    if values.std() < PATCH_NCC_STD_MIN or std < PATCH_NCC_STD_MIN:
        return PATCH_NCC_UNIFORM_FALLBACK
    centered = values-values.mean()
    denominator = float(np.sqrt(float(np.dot(centered, centered))*dot))
    if denominator == 0:
        return PATCH_NCC_UNIFORM_FALLBACK
    result = float(np.dot(centered, b)/denominator)
    return result if not np.isnan(result) else PATCH_NCC_UNIFORM_FALLBACK


@pytest.mark.parametrize('size', [1, 3, 12, 768, 1728])
@pytest.mark.parametrize('scale', [0, 1e-7, 1e-6, 1, 255])
def test_prepared_ncc_preserves_float_result(size: int, scale: float) -> None:
    rng = np.random.default_rng(size)
    for _ in range(20):
        a, raw = (rng.normal(size=size)*scale for _ in range(2))
        centered = raw-raw.mean()
        dot, std = float(np.dot(centered, centered)), float(raw.std())
        assert _compute_ncc_prepared(a, centered, dot, std) == previous_ncc(a, centered, dot, std)


@pytest.mark.parametrize('shape', [(20, 30), (60, 80), (111, 147)])
def test_prepared_template_matches_reference(shape: tuple) -> None:
    rng = np.random.default_rng(sum(shape))
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    template = image[3:12, 5:17].copy()
    matcher = PreparedTemplate(template)
    reference = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    assert np.max(abs(matcher.scores(image)-reference)) < NCC_DECISION_MARGIN
    _, maximum, _, position = cv2.minMaxLoc(reference)
    assert matcher.peak(image, 0.55) == (maximum, position)


def test_prepared_template_falls_back_near_threshold() -> None:
    matcher = PreparedTemplate(np.ones((2, 2), dtype=np.uint8))
    image = np.zeros((4, 4), dtype=np.uint8)
    approximate = np.full((3, 3), 0.55-NCC_DECISION_MARGIN/2, dtype=np.float32)
    with patch.object(matcher, 'scores', return_value=approximate):
        with patch('cv2.matchTemplate', return_value=np.ones((3, 3), np.float32)) as original:
            assert matcher.peak(image, 0.55)[0] == 1
            original.assert_called_once()


@pytest.mark.parametrize('seed', range(10))
def test_ui_inner_product_matches_original(seed: int) -> None:
    rng = np.random.default_rng(seed)
    matcher = UiMaskMatcher.load_default()
    for _ in range(30):
        image = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        assert matcher.is_ui(image) == matcher.match(image).is_ui
    for template in matcher._templates.values():
        image = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        assert matcher.is_ui(image) == matcher.match(image).is_ui


@pytest.mark.parametrize('value', [0, 128, 255])
def test_ui_uniform_and_empty(value: int) -> None:
    matcher = UiMaskMatcher({'flat': np.full(NORM_SIZE[::-1], value, np.uint8)})
    for image in (np.zeros((24, 24, 3), np.uint8), np.zeros((0, 0, 3), np.uint8)):
        assert matcher.is_ui(image) == matcher.match(image).is_ui


def test_ui_irregular_template_uses_original() -> None:
    matcher = UiMaskMatcher({'small': np.arange(12, dtype=np.uint8).reshape(3, 4)})
    image = np.zeros((24, 24, 3), np.uint8)
    assert matcher._normalized_templates is None
    assert matcher.is_ui(image) == matcher.match(image).is_ui


@pytest.mark.parametrize('field', ['frame', 't_sec', 'boards', 'raw', 'states', 'scores'])
def test_comparison_rejects_single_changed_value(field: str) -> None:
    from scripts.analyze_live_b14 import compare_arrays, OUTPUT_FIELDS
    baseline = {key: np.zeros((3, 2), dtype=np.int64) for key in OUTPUT_FIELDS}
    candidate = {key: value.copy() for key, value in baseline.items()}
    candidate[field][1, 0] = 1
    report = compare_arrays(baseline, candidate)
    assert not report['equal']
    assert report['differing_frames'][field] == 1


def test_comparison_rejects_missing_frames() -> None:
    from scripts.analyze_live_b14 import compare_arrays
    with pytest.raises(ValueError):
        compare_arrays({'frame': np.arange(2)}, {'frame': np.arange(1)})


def test_long_requires_success_and_same_source(tmp_path: Path) -> None:
    import json
    from scripts.measure_live_b14 import check_proof
    from scripts.analyze_live_b14 import source_hashes
    path = tmp_path/'proof.json'
    proof = dict(equal=True, target_met=True, source_hashes=source_hashes())
    path.write_text(json.dumps(proof))
    assert check_proof(path) == proof
    for field in ('equal', 'target_met'):
        path.write_text(json.dumps(dict(proof, **{field: False})))
        with pytest.raises(ValueError):
            check_proof(path)
    proof['source_hashes']['src/image_reader.py'] = 'changed'
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        check_proof(path)


@pytest.mark.parametrize('shape', [(48, 64), (1080, 1920)])
@pytest.mark.parametrize('offset', [0, -30, 2000])
def test_board_hsv_preserves_every_sample(shape: tuple, offset: int) -> None:
    from src.image_reader import ImageReader, BoardRegion
    from src.board import HIDDEN_ROWS, BOARD_ROWS, BOARD_COLS
    reader = ImageReader.__new__(ImageReader)
    frame = np.random.default_rng(1).integers(0, 256, (*shape, 3), dtype=np.uint8)
    regions = (BoardRegion(offset, -10, 320, 650), BoardRegion(1200, 160, 340, 680))
    expected = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    actual = reader._board_hsv_frame(frame, regions)
    height, width = shape
    for region in regions:
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                x1, y1 = max(0, min(x1, width-1)), max(0, min(y1, height-1))
                x2, y2 = max(x1+1, min(x2, width)), max(y1+1, min(y2, height))
                assert np.array_equal(actual[y1:y2, x1:x2], expected[y1:y2, x1:x2])


def test_profile_keeps_b13_odd_frame_phase() -> None:
    from scripts.profile_live_b14 import video_bounds
    assert video_bounds(60, 2580.566, 3412) == (60, 154773, 204720, 2)


def test_validation_uses_source_frame_end_and_checks_every_frame(tmp_path: Path) -> None:
    import json
    from scripts.analyze_live_b14 import validate, source_hashes, OUTPUT_FIELDS
    candidate, reference = tmp_path/'candidate', tmp_path/'reference'
    candidate.mkdir()
    reference.mkdir()
    frames = np.array([154773, 154775, 154777])
    arrays = {key: np.zeros((3, 2), dtype=np.int64) for key in OUTPUT_FIELDS}
    arrays.update(frame=frames, t_sec=frames/60)
    np.savez(candidate/'recognition.npz', **arrays)
    np.savez(reference/'recognition.npz', **arrays)
    manifest = dict(source_hashes=source_hashes(), baseline=False, original_templates=False)
    (candidate/'manifest.json').write_text(json.dumps(manifest))
    (candidate/'bounds.json').write_text(json.dumps(dict(fps=60, start=154773, end=154779, stride=2)))
    with patch('scripts.analyze_live_b14.stage_summary', return_value=dict(frames=3600, recognition_ms=[27, 32])):
        proof = validate(reference, candidate, 2579.55, 2580)
        assert proof['equal'] and proof['target_met']
        assert proof['comparison_end'] == 154779/60
        arrays = {key: value[:-1] for key, value in arrays.items()}
        np.savez(candidate/'recognition.npz', **arrays)
        with pytest.raises(ValueError, match='全フレーム'):
            validate(reference, candidate, 2579.55, 2580)


def test_long_comparison_rejects_changed_asset(tmp_path: Path) -> None:
    import json
    from scripts.measure_live_b14 import finish_comparison
    reference, current = tmp_path/'reference', tmp_path/'long'
    for path, value in ((reference, 'original'), (current, 'changed')):
        path.mkdir()
        (path/'metrics.json').write_text(json.dumps(dict(assets=dict(recognition_model_hash=value))))
    with pytest.raises(ValueError, match='recognition_model_hash'):
        finish_comparison(tmp_path, reference)
