"""実機なしで入力の明示選択・拒否・再確認・解放を検査する。"""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.phase_j.live_device import DeviceConfig, DirectShowSource, PuyoScreenVerifier, border_coverage, border_pair
from src.phase_j.live_publish import initial_snapshot, result_snapshot
from src.phase_j.validator import validate_snapshot
from tests.phase_j.test_live_publish import ASSETS


class FakeDevice:
    def __init__(self, image: np.ndarray | None) -> None:
        self.image, self.released = image, False

    def isOpened(self) -> bool:
        return self.image is not None

    def set(self, key: int, value: float) -> bool:
        return True

    def read(self) -> tuple:
        return True, self.image

    def release(self) -> None:
        self.released = True


@pytest.mark.parametrize('shape,verified,accepted', [((720, 1280, 3), True, True),
    ((1080, 1920, 3), True, True), ((480, 640, 3), True, False),
    ((1080, 1920, 3), False, False), (None, True, False)])
def test_gate_and_explicit_index(shape: tuple | None, verified: bool, accepted: bool) -> None:
    capture = FakeDevice(np.zeros(shape, np.uint8) if shape else None)
    calls, holds, clock = [], [], [0.0]
    def factory(index: int, backend: int) -> FakeDevice:
        calls.append((index, backend))
        return capture
    def sleep(seconds: float) -> None:
        clock[0] += seconds
    source = DirectShowSource(DeviceConfig('OBS', 3), 0.12, lambda: holds.append(True),
        factory, lambda image: verified, lambda: clock[0], sleep)
    frames = list(source)
    assert bool(frames) == accepted and bool(holds) != accepted
    assert calls == [(3, cv2.CAP_DSHOW)] and capture.released
    assert all(frame.image.shape == (1080, 1920, 3) for frame in frames)


@pytest.mark.parametrize('name,index', [('', 0), ('OBS', -1), ('OBS', True), ('OBS', '0')])
def test_invalid_config(name: str, index: object) -> None:
    with pytest.raises(ValueError):
        DeviceConfig(name, index)


def test_no_name_only_selection_and_calibration_path(tmp_path: Path) -> None:
    path = tmp_path / 'input.json'
    path.write_text('{"name":"OBS"}')
    with pytest.raises(KeyError):
        DeviceConfig.load(path)
    assert DeviceConfig('OBS', 0).calibration_path != DeviceConfig('OBS', 1).calibration_path


def test_pending_has_no_fabricated_probability() -> None:
    row = dict(frame=0, game=0, t_sec=0, captured_at=1.0, recognized_at=1.0,
        evaluated_at=1.0, queue_depth=0, raw_probability=None, input_verifying=True)
    result = result_snapshot(initial_snapshot(ASSETS), row, 2.0, 1, 1.0)
    assert validate_snapshot(result).is_valid, validate_snapshot(result)
    assert result.display['status'] == 'hold' and result.display['message'] == '入力確認中'
    assert result.evaluations['practical']['p1_win_probability'] is None


@pytest.mark.parametrize('grid,score', [(None, 0), (True, None), (True, 0)])
def test_verifier_needs_both_detectors(grid: object, score: int | None) -> None:
    verifier = object.__new__(PuyoScreenVerifier)
    verifier.regions = [SimpleNamespace(x=100, y=100, width=384, height=720)] * 2
    detection = SimpleNamespace(top_left=(20, 20), bottom_right=(404, 740)) if grid else None
    verifier.grid = SimpleNamespace(detect=lambda image: detection)
    verifier.score = SimpleNamespace(read_side=lambda image, side: (score, 1.0))
    assert verifier(np.zeros((1080, 1920, 3), np.uint8)) == (grid is not None and score is not None)


def test_fragmented_border_coverage_does_not_double_count() -> None:
    lines = [(20, 20, 20, 80), (20, 60, 20, 120), (20, 20, 20, 80)]
    assert border_coverage(lines, 0, 100, 20) == 1.0
    assert not border_pair(lines, 0, 100, 100)
    assert border_pair(lines + [(120, 20, 120, 120)], 0, 100, 100)


def test_misplaced_border_is_rejected() -> None:
    assert not border_pair([(60, 20, 60, 120), (160, 20, 160, 120)], 0, 100, 100)


def test_verifier_failure_releases_capture() -> None:
    capture = FakeDevice(np.zeros((1080, 1920, 3), np.uint8))
    def broken(image: np.ndarray) -> bool:
        raise ValueError('検出器失敗')
    source = DirectShowSource(DeviceConfig('OBS', 0), 1.0, lambda: None,
        lambda *args: capture, broken)
    with pytest.raises(ValueError):
        list(source)
    assert capture.released
