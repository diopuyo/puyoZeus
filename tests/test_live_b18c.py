"""画素判定の同値性、参照保持、境界判定を検証する。"""
from __future__ import annotations

import ast
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.animation_filter import AnimationFilter
from src.effect_glow_detector import is_effect_glow_active
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.phase_j.live_snapshot_quality import SnapshotQuality

VISIBLE_ROWS = frozenset(range(1, 13))


@pytest.mark.parametrize('region', [DEFAULT_P1_REGION, DEFAULT_P2_REGION])
def test_quality_matches_original_pixels_with_owned_crop(region: object) -> None:
    old, new = AnimationFilter(), SnapshotQuality()
    rng = np.random.default_rng(18)
    for tick in range(12):
        image = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        crop = image[region.y:region.y+region.height, region.x:region.x+region.width]
        expected = old.is_animation(crop.copy(), (0, 0, region.width, region.height)).is_animation
        actual, glow = new.observe(image, region)
        assert actual == expected
        assert glow == is_effect_glow_active(image, region, VISIBLE_ROWS)
        assert not np.shares_memory(new.previous, image)
        assert new.previous.nbytes == crop.nbytes
        if tick == 6:
            old.reset()
            new.reset()


@pytest.mark.parametrize('fraction', [0., .969, .97, .971, 1.])
def test_glow_threshold_is_unchanged(fraction: float) -> None:
    region = DEFAULT_P1_REGION
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    x1, y1, x2, y2 = region.cell_sample_rect(3, 2)
    patch = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    patch.reshape(-1, 3)[:round(patch.shape[0]*patch.shape[1]*fraction), 1] = 230
    image[y1:y2, x1:x2] = patch
    assert SnapshotQuality().observe(image, region)[1] == is_effect_glow_active(image, region, VISIBLE_ROWS)


@pytest.mark.parametrize('delta', [17, 18, 19, 24, 25, 26])
def test_animation_threshold_is_unchanged(delta: int) -> None:
    region = DEFAULT_P1_REGION
    old, new = AnimationFilter(), SnapshotQuality()
    for level in (0, delta):
        frame = np.full((1080, 1920, 3), level, dtype=np.uint8)
        crop = frame[region.y:region.y+region.height, region.x:region.x+region.width]
        expected = old.is_animation(crop, (0, 0, region.width, region.height)).is_animation
        assert new.observe(frame, region)[0] == expected


def test_new_functions_obey_length_limit() -> None:
    paths = ('src/phase_j/live_snapshot_quality.py', 'src/phase_j/live_snapshot.py',
             'scripts/profile_live_b18c.py', 'scripts/measure_live_b18c.py', 'scripts/verify_live_b18c.py',
             'scripts/summarize_live_b18c.py')
    long = [(path, node.name) for path in paths
            for node in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8')))
            if isinstance(node, ast.FunctionDef) and node.end_lineno-node.lineno+1 > 50]
    assert not long


def test_existing_hsv_pixels_and_scratch_are_reused() -> None:
    region = DEFAULT_P1_REGION
    a, b = SnapshotQuality(), SnapshotQuality()
    rng = np.random.default_rng(32)
    for _ in range(4):
        frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        crop = frame[region.y:region.y+region.height, region.x:region.x+region.width]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        expected = hsv.copy()
        assert a.observe(frame, region) == b.observe(frame, region, hsv)
        np.testing.assert_array_equal(hsv, expected)
        channels, integral = b.channels, b.integral
        b.prepare(crop.shape)
        assert all(x is y for x, y in zip(channels, b.channels)) and integral is b.integral


def test_reader_reuses_only_matching_hsv_roi_and_clears_next_frame() -> None:
    from dataclasses import replace
    from types import SimpleNamespace
    from src.image_reader import ImageReader
    from src.phase_j.live_snapshot import prepare_recognition
    reader = ImageReader()
    owner = SimpleNamespace(_reader=reader)
    hsv = np.zeros((1080, 1920, 3), dtype=np.uint8)
    prepare_recognition(owner)
    reader._remember_live_hsv_pixels(hsv, replace(reader._p1_region, x=reader._p1_region.x+1))
    assert reader._live_hsv_pixels == {}
    reader._remember_live_hsv_pixels(hsv, reader._p1_region)
    assert np.shares_memory(reader._live_hsv_pixels[0], hsv)
    prepare_recognition(owner)
    assert reader._live_hsv_pixels == {}


def test_irregular_roi_keeps_integral_fallback() -> None:
    from dataclasses import replace
    region = replace(DEFAULT_P1_REGION, width=379, height=713)
    frame = np.full((1080, 1920, 3), 230, dtype=np.uint8)
    value = SnapshotQuality()
    assert value.observe(frame, region)[1] == is_effect_glow_active(frame, region, VISIBLE_ROWS)
    assert value.samples is None
