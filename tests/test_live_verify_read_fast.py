"""照合読みの旗・整数差・履歴と品質理由の一致を確認する。"""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from src.animation_filter import AnimationFilter, compute_frame_diff
from src.placement_signal_runtime import PlacementSignalRuntime
from src.phase_j.live_speed import SPEED_DEFAULTS
from src.phase_j.live_stage_timing import CURRENT
from src.phase_j.launcher import parse_user_config, build_pipeline_config

FRAME_SHAPE = (1080, 1920, 3)
SEED = 20261031


class FixedCnn:
    def classify_batch(self, patches: list[np.ndarray]) -> list[int]:
        return [int(p[0, 0, 0]) % 6 for p in patches]


@pytest.mark.parametrize('region', [None, (0, 0, 17, 19), (-4, -2, 80, 60)])
@pytest.mark.parametrize('dtype', [np.uint8, np.int16])
def test_difference_is_bit_identical(region: tuple | None, dtype: type) -> None:
    rng = np.random.default_rng(SEED)
    before = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8).astype(dtype)
    after = rng.integers(0, 256, before.shape, dtype=np.uint8).astype(dtype)
    assert compute_frame_diff(before, after, region) == compute_frame_diff(before, after, region, fast=True)


def test_animation_history_is_identical() -> None:
    slow, fast = AnimationFilter(), AnimationFilter()
    rng = np.random.default_rng(SEED)
    frames = [rng.integers(0, 256, (60, 80, 3), dtype=np.uint8) for _ in range(5)]
    frames += [np.zeros_like(frames[0]), np.full_like(frames[0], 255)]
    for frame in frames:
        assert slow.is_animation(frame, (0, 0, 80, 60)) == fast.is_animation(frame, (0, 0, 80, 60), fast=True)
        assert slow._prev_stats == fast._prev_stats


def test_read_outputs_and_quality_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_LIVE_VERIFY_READ_FAST', raising=False)
    slow = PlacementSignalRuntime(FixedCnn(), None)
    assert not slow.read_fast and SPEED_DEFAULTS['live_verify_read_fast'] is False
    monkeypatch.setenv('PUYO_LIVE_VERIFY_READ_FAST', '1')
    fast = PlacementSignalRuntime(FixedCnn(), None)
    assert fast.read_fast
    frames = [np.full(FRAME_SHAPE, value, np.uint8) for value in (0, 255, 255, 0)]
    for index, frame in enumerate(frames):
        pipe = SimpleNamespace(_all_clear_pending_1p=index == 3)
        expected, actual = slow.read(frame, index, index/30, pipe), fast.read(frame, index, index/30, pipe)
        for left, right in zip(expected, actual):
            assert (left.frame, left.stamp, left.quality, left.erasing) == (right.frame, right.stamp, right.quality, right.erasing)
            assert left.cnn.tobytes() == right.cnn.tobytes()
            assert left.hsv.tobytes() == right.hsv.tobytes()


def test_instrumentation_works_with_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_LIVE_VERIFY_READ_FAST', raising=False)
    row = {'stages': {}}
    token = CURRENT.set(row)
    try:
        PlacementSignalRuntime(FixedCnn(), None).read(np.zeros(FRAME_SHAPE, np.uint8), 0, 0., SimpleNamespace())
    finally:
        CURRENT.reset(token)
    assert set(row['stages']) == {f'placement_read_{s}' for s in ('patches', 'cnn', 'hsv', 'animation', 'glow')}
    assert row['stages']['placement_read_animation']['calls'] == 2
    assert row['stages']['placement_read_glow']['calls'] == 2


def test_flag_is_forwarded_from_user_config() -> None:
    config = parse_user_config({'device_name': 'OBS Virtual Camera', 'live_verify_read_fast': True}, Path('.'))
    assert config.live_verify_read_fast is True
    assert build_pipeline_config(config)['live_verify_read_fast'] is True
