"""画像非同梱候補の既定 OFF・失敗閉鎖・画像読込禁止を検証する。"""
from pathlib import Path

import cv2
import joblib
import numpy as np
import pytest

from src.image_free_digit import ImageFreeScoreOcr, digit_features, FEATURE_SIZE, MODEL_SCHEMA
from src.score_ocr import ScoreOcr


class ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value)


def model_file(tmp_path: Path, label: int = 3) -> Path:
    path = tmp_path / 'student.joblib'
    joblib.dump(dict(schema=MODEL_SCHEMA, feature_size=FEATURE_SIZE,
                    labels=ConstantEstimator(label), confidence=ConstantEstimator(0.9),
                    multiply=ConstantEstimator(0.8)), path)
    return path


def test_default_remains_legacy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('PUYO_IMAGE_FREE_SCORE_OCR', raising=False)
    assert type(ScoreOcr.load_default(tmp_path)) is ScoreOcr


def test_missing_model_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PUYO_IMAGE_FREE_SCORE_OCR', '1')
    monkeypatch.setenv('PUYO_IMAGE_FREE_DIGIT_MODEL', str(tmp_path / 'missing.joblib'))
    with pytest.raises(FileNotFoundError):
        ScoreOcr.load_default(tmp_path)


def test_no_image_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PUYO_IMAGE_FREE_SCORE_OCR', '1')
    monkeypatch.setenv('PUYO_IMAGE_FREE_DIGIT_MODEL', str(model_file(tmp_path)))
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('画像を読んではならない')
    monkeypatch.setattr(cv2, 'imread', forbidden)
    reader = ScoreOcr.load_default(tmp_path)
    result = reader.read(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert result.score_1p == result.score_2p == 33333333
    reader.read_formula_side(np.zeros((1080, 1920, 3), dtype=np.uint8), '1P')


def test_unknown_is_not_a_digit(tmp_path: Path) -> None:
    reader = ImageFreeScoreOcr(model_file(tmp_path, 10))
    assert reader._classify_digit(np.zeros((50, 40, 3), dtype=np.uint8)) == (None, 0.9)


def test_features_are_finite() -> None:
    values = digit_features(np.zeros((50, 40, 3), dtype=np.uint8))
    assert values.shape == (FEATURE_SIZE[0] * FEATURE_SIZE[1] * 3,)
    assert np.isfinite(values).all()


def test_untrained_threshold_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='教師閾値'):
        ImageFreeScoreOcr(model_file(tmp_path), min_confidence=0.01)


def test_wide_r1_match_is_not_silently_disabled(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='R1'):
        _ = ImageFreeScoreOcr(model_file(tmp_path))._mult_template_gray
    with pytest.raises(RuntimeError, match='R1'):
        ImageFreeScoreOcr(model_file(tmp_path)).placement_multiply_reader()


def test_wide_r1_uses_dedicated_estimator(tmp_path: Path) -> None:
    reader = ImageFreeScoreOcr(model_file(tmp_path))
    reader._student['wide_multiply'] = ConstantEstimator(0.73)
    callback = reader.placement_multiply_reader()
    assert callback(np.zeros((1080, 1920, 3), dtype=np.uint8)) == [0.73, 0.73]


@pytest.mark.parametrize('label,confidence', [(3, 0.9), (3, 0.6), (10, 0.8)])
def test_batched_read_preserves_legacy_aggregation(
    tmp_path: Path, label: int, confidence: float,
) -> None:
    reader = ImageFreeScoreOcr(model_file(tmp_path, label))
    reader._student['confidence'] = ConstantEstimator(confidence)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for side in ('1P', '2P'):
        for offset in (None, (2, 1)):
            expected = ScoreOcr._read_one_side_detail(reader, frame, side, offset)
            assert reader._read_one_side_detail(frame, side, offset) == expected
    sequence = [frame, frame // 2]
    result = reader.read_batch(sequence)
    for i, image in enumerate(sequence):
        for j, side in enumerate(('1P', '2P')):
            detail = reader.read_side_detail(image, side)
            formula = reader.read_formula_side(image, side, detail[2], detail[3])
            assert result[i][j] == (detail, formula)
    assert reader.read_batch([]) == []
