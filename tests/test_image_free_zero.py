"""ゼロ表示は盤面状態に依存せず、旧閾値と同じ採否を公開する。"""
from __future__ import annotations

import numpy as np
import pytest

from src.image_free_zero import LocalZeroModel, SCORE_REGIONS, zero_features
from src.score_zero import ZERO_NCC_THRESHOLD


class ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.full(len(data), self.value)


@pytest.mark.parametrize('side', [0, 1])
def test_zero_ignores_board_and_other_score(side: int) -> None:
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    expected = zero_features(frame, side)
    changed = rng.integers(0, 256, frame.shape, dtype=np.uint8)
    y1, y2, x1, x2 = SCORE_REGIONS[side]
    changed[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    np.testing.assert_array_equal(zero_features(changed, side), expected)


@pytest.mark.parametrize('left,right', [(False, False), (False, True), (True, False), (True, True)])
def test_zero_decisions_and_scores_agree(left: bool, right: bool) -> None:
    model = LocalZeroModel((ConstantRegressor(0.9 if left else 0.8),
                            ConstantRegressor(0.9 if right else 0.8)))
    result = model.predict_image(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert result['is_1p_zero'] == left == (result['score_1p'] >= ZERO_NCC_THRESHOLD)
    assert result['is_2p_zero'] == right == (result['score_2p'] >= ZERO_NCC_THRESHOLD)


@pytest.mark.parametrize('delta,expected', [(-1e-7, False), (0.0, True), (1e-7, True)])
def test_zero_preserves_legacy_threshold_boundary(delta: float, expected: bool) -> None:
    regressor = ConstantRegressor(ZERO_NCC_THRESHOLD + delta)
    result = LocalZeroModel((regressor, regressor)).predict_image(
        np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert result['is_1p_zero'] is expected
    assert result['is_2p_zero'] is expected


def test_learned_zero_uses_local_model_without_image_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.compare_image_free_all import mode
    from src import image_free_ui
    from src.score_zero import ScoreZeroDetector

    model = LocalZeroModel((ConstantRegressor(0.9), ConstantRegressor(0.8)))
    monkeypatch.setattr(image_free_ui, 'load', lambda path: {'tasks': {'zero': model}})
    with mode(True):
        detector = ScoreZeroDetector.load_default()
        result = detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert result.is_1p_zero and not result.is_2p_zero
    assert not result.both_zero
