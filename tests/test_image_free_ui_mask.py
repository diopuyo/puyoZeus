"""×印候補が画像を読まず、未配置モデルを黙って無効化しないことを確認する。"""
from pathlib import Path

import cv2
import joblib
import numpy as np
import pytest

from src.image_free_ui_mask import ImageFreeUiMaskMatcher, MODEL_SCHEMA
from src.ui_mask import UiMaskMatcher


class ConstantEstimator:
    def __init__(self, value: str | float) -> None:
        self.value = value

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value)


def test_default_is_legacy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('PUYO_IMAGE_FREE_UI_MASK', raising=False)
    assert type(UiMaskMatcher.load_default(tmp_path)) is UiMaskMatcher


def test_missing_model_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PUYO_IMAGE_FREE_UI_MASK', '1')
    monkeypatch.setenv('PUYO_IMAGE_FREE_UI_MASK_MODEL', str(tmp_path / 'absent'))
    with pytest.raises(FileNotFoundError):
        UiMaskMatcher.load_default(tmp_path)


@pytest.mark.parametrize('label', ['', 'x_mark'])
def test_no_image_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, label: str) -> None:
    path = tmp_path / 'ui.joblib'
    joblib.dump(dict(schema=MODEL_SCHEMA, labels=ConstantEstimator(label),
                    confidence=ConstantEstimator(0.8)), path)
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('画像を読んではならない')
    monkeypatch.setattr(cv2, 'imread', forbidden)
    reader = ImageFreeUiMaskMatcher(path)
    patch = np.zeros((64, 64, 3), dtype=np.uint8)
    assert reader.is_ui(patch) == bool(label)
    assert not reader.is_ui(patch[:0])
