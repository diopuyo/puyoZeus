"""顔以外の画素・コマ時間とHTTP差し替えを検証する。"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image
import pytest

from src.phase_j.overlay_config import validate_overlay
from src.phase_j.overlay_http import BINARY_ASSETS
from tests.phase_j.test_distribution_app import distribution

BRAND = Path(__file__).resolve().parents[2] / 'src/phase_j/brand'
BANDS = (10, 30, 50, 70, 90)
DURATIONS = (60, 30, 20, 60, 20, 30)
# 10/7追加の目の外側の線も顔の装飾として含める。
FACE = (98, 207, 215, 272)


@pytest.mark.parametrize('band', BANDS)
def test_frames_preserve_everything_outside_face(band: int) -> None:
    original = Image.open(BRAND / 'overlay_mascot.webp')
    variant = Image.open(BRAND / f'mascot_{band}.webp')
    assert variant.size == original.size == (320, 320)
    assert variant.n_frames == original.n_frames == 6
    assert variant.info['loop'] == original.info['loop'] == 0
    outside = np.ones((320, 320), dtype=bool)
    left, top, right, bottom = FACE
    outside[top:bottom, left:right] = False
    for index, duration in enumerate(DURATIONS):
        original.seek(index)
        variant.seek(index)
        expected, actual = np.array(original.convert('RGBA')), np.array(variant.convert('RGBA'))
        assert variant.info['duration'] == original.info['duration'] == duration
        assert np.array_equal(expected[:, :, 3], actual[:, :, 3])
        visible = outside & (expected[:, :, 3] > 0)
        assert np.array_equal(expected[visible], actual[visible])
        assert np.array_equal(expected, actual) if band == 50 else not np.array_equal(expected, actual)
    if band == 50:
        assert (BRAND / 'overlay_mascot.webp').read_bytes() == (BRAND / 'mascot_50.webp').read_bytes()


@pytest.mark.parametrize('band', BANDS)
def test_http_animated_assets_and_custom_override(distribution: tuple, band: int) -> None:
    runtime, url = distribution
    route = f'/mascot/{band}.webp'
    assert BINARY_ASSETS[route] == (f'brand/mascot_{band}.webp', 'image/webp')
    with urlopen(url + route) as response:
        assert response.headers['Content-Type'] == 'image/webp'
        assert response.read() == (BRAND / f'mascot_{band}.webp').read_bytes()
    folder = runtime.root / 'mascot'
    folder.mkdir(exist_ok=True)
    replacement = (BRAND / 'overlay_mascot.webp').read_bytes()
    (folder / f'{band}.webp').write_bytes(replacement)
    with urlopen(url + route) as response:
        assert response.read() == replacement


@pytest.mark.parametrize('name', ['mascot_expressions', 'mascot_flip_right'])
def test_config_requires_booleans(name: str) -> None:
    for value in (True, False):
        assert validate_overlay({name: value}) == {name: value}
    for value in ('true', 1, None):
        with pytest.raises(ValueError):
            validate_overlay({name: value})


@pytest.mark.parametrize('edges', [
    [10, 30, 70, 90], [20, 40, 60, 80], [5.5, 25, 75, 95.5]])
def test_config_accepts_custom_band_edges(edges: list[float]) -> None:
    assert validate_overlay({'mascot_band_edges': edges}) == {'mascot_band_edges': edges}


@pytest.mark.parametrize('edges', [
    None, '10,30,70,90', [], [10, 30, 90], [10, 30, 70, 90, 95],
    [0, 30, 70, 90], [10, 30, 70, 100], [10, 30, 30, 90], [90, 70, 30, 10],
    [True, 30, 70, 90], ['10', 30, 70, 90], [float('nan'), 30, 70, 90]])
def test_config_rejects_invalid_band_edges(edges: object) -> None:
    with pytest.raises(ValueError):
        validate_overlay({'mascot_band_edges': edges})
