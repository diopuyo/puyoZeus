"""TelopDetector 高速経路 (PUYO_FAST_TELOP): 既定OFF・決定同一・予備判定が実際に効くこと。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src import telop_detector as module
from src.telop_detector import SEARCH_H, SEARCH_W, SEARCH_X, SEARCH_Y, TelopDetector

FRAME_SHAPE = (1080, 1920, 3)
SEED = 20260930
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _templates(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """構造のある (滑らかな縞+ブロック) 灰色テンプレート 2 種。"""
    result = {}
    for name, (height, width) in (('telop_a', (150, 360)), ('telop_b', (80, 530))):
        base = rng.integers(0, 255, (height // 10, width // 10), dtype=np.uint8)
        result[name] = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
    return result


def _background(rng: np.random.Generator) -> np.ndarray:
    small = rng.integers(40, 200, (54, 96, 3), dtype=np.uint8)
    return cv2.resize(small, (FRAME_SHAPE[1], FRAME_SHAPE[0]), interpolation=cv2.INTER_LINEAR)


def _with_telop(background: np.ndarray, template: np.ndarray, x: int, y: int, alpha: float) -> np.ndarray:
    frame = background.copy()
    height, width = template.shape
    patch = frame[y:y+height, x:x+width].astype(np.float32)
    layer = np.repeat(template[:, :, None], 3, axis=2).astype(np.float32)
    frame[y:y+height, x:x+width] = (patch*(1-alpha)+layer*alpha).astype(np.uint8)
    return frame


def _frames(rng: np.random.Generator, templates: dict[str, np.ndarray]):
    background = _background(rng)
    yield background
    yield _background(rng)
    for name, template in templates.items():
        x, y = SEARCH_X+40, SEARCH_Y+30
        for alpha in ALPHAS:
            yield _with_telop(background, template, x, y, alpha)


def _key(result) -> tuple:
    return result.is_visible, result.bbox


def test_default_off_and_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    templates = _templates(np.random.default_rng(SEED))
    monkeypatch.delenv(module.FAST_ENV, raising=False)
    assert TelopDetector(templates).fast is False
    monkeypatch.setenv(module.FAST_ENV, '1')
    assert TelopDetector(templates).fast is True
    assert TelopDetector(templates, fast=False).fast is False


def test_fast_decisions_equal_reference_and_prefilter_is_exercised() -> None:
    rng = np.random.default_rng(SEED)
    templates = _templates(rng)
    slow, fast = TelopDetector(templates, fast=False), TelopDetector(templates, fast=True)
    frames = list(_frames(rng, templates))
    decisions = [(_key(slow.detect(frame)), _key(fast.detect(frame))) for frame in frames]
    assert all(a == b for a, b in decisions)
    assert any(a[0] for a, _ in decisions)          # 陽性フレームが実在する (母数の健全性)
    assert any(not a[0] for a, _ in decisions)      # 陰性も実在する
    assert fast.coarse_rejects > 0                   # 予備判定が実際に全解像度照合を省いた
    assert slow.coarse_rejects == 0                  # OFF 経路では予備判定に入らない


def test_off_path_result_fields_untouched() -> None:
    """OFF は従来どおり template_name / score を返す (高速経路だけが省略する)。"""
    rng = np.random.default_rng(SEED)
    templates = _templates(rng)
    result = TelopDetector(templates, fast=False).detect(_background(rng))
    assert result.template_name in templates and result.score > -1.0


def test_template_larger_than_shrunk_roi_falls_back_to_reference() -> None:
    """縮小後にテンプレートが入らない ROI (小さい画像) では判断せず従来経路へ (例外にしない)。"""
    rng = np.random.default_rng(SEED)
    templates = _templates(rng)
    tiny = np.zeros((700, 1000, 3), dtype=np.uint8)
    assert _key(TelopDetector(templates, fast=True).detect(tiny)) == _key(TelopDetector(templates, fast=False).detect(tiny))
