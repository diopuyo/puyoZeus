"""実機なしで入力の明示選択・拒否・再確認・解放を検査する。"""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.phase_j.live_device import (DeviceConfig, DirectShowSource, PuyoScreenVerifier, border_coverage, border_pair,
                                         frame_border_matches)
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


# ---- 画面確認の緩和 (relaxed_verify) と連続不合格ヒステリシス (verify_fail_streak) ----
BOARD_W, BOARD_H = 384, 720
CROP_MARGIN = 20


def _frame_crop(top: bool, bottom: bool = True, left: bool = True, right: bool = True) -> np.ndarray:
    """盤面 crop (外側 20px マージン込み) に外枠の指定辺だけを描く。"""
    crop = np.zeros((BOARD_H+2*CROP_MARGIN, BOARD_W+2*CROP_MARGIN, 3), np.uint8)
    x0, y0, x1, y1 = CROP_MARGIN, CROP_MARGIN, CROP_MARGIN+BOARD_W, CROP_MARGIN+BOARD_H
    for enabled, a, b in ((top, (x0, y0), (x1, y0)), (bottom, (x0, y1), (x1, y1)),
                          (left, (x0, y0), (x0, y1)), (right, (x1, y0), (x1, y1))):
        if enabled:
            cv2.line(crop, a, b, (255, 255, 255), 3)
    return crop


def _fake_verifier(read_score: int | None, formula_valid: bool, relaxed: bool,
                   crop_has_frame: bool = False) -> PuyoScreenVerifier:
    verifier = object.__new__(PuyoScreenVerifier)
    verifier.relaxed = relaxed
    verifier.regions = [SimpleNamespace(x=100, y=100, width=BOARD_W, height=BOARD_H)] * 2
    verifier.grid = SimpleNamespace(detect=lambda image: None)
    verifier.score = SimpleNamespace(read_side=lambda image, side: (read_score, 1.0),
        read_formula_side=lambda image, side: SimpleNamespace(valid=formula_valid))
    return verifier


def test_formula_visible_frame_accepted_only_when_relaxed(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_device
    monkeypatch.setattr(live_device, 'frame_border_matches', lambda *args, **kwargs: True)
    image = np.zeros((1080, 1920, 3), np.uint8)
    assert _fake_verifier(None, True, relaxed=True)(image)
    assert not _fake_verifier(None, True, relaxed=False)(image)   # 既定は従来どおり得点必須


def test_formula_alone_does_not_pass_without_board(monkeypatch: pytest.MonkeyPatch) -> None:
    """非ぷよ: 式も得点も読めない、または盤面枠が無ければ relaxed でも不合格。"""
    from src.phase_j import live_device
    image = np.zeros((1080, 1920, 3), np.uint8)
    monkeypatch.setattr(live_device, 'frame_border_matches', lambda *args, **kwargs: False)
    assert not _fake_verifier(None, True, relaxed=True)(image)
    monkeypatch.setattr(live_device, 'frame_border_matches', lambda *args, **kwargs: True)
    assert not _fake_verifier(None, False, relaxed=True)(image)


def test_high_stack_top_edge_accepted_only_when_top_exempt() -> None:
    crop = _frame_crop(top=False)
    assert not frame_border_matches(crop, BOARD_W, BOARD_H)
    assert frame_border_matches(crop, BOARD_W, BOARD_H, exempt_top=True)
    assert frame_border_matches(_frame_crop(top=True), BOARD_W, BOARD_H)  # 4 辺あれば従来も合格


@pytest.mark.parametrize('missing', ['bottom', 'left', 'right'])
def test_top_exemption_still_requires_other_three_edges(missing: str) -> None:
    crop = _frame_crop(top=False, **{missing: False})
    assert not frame_border_matches(crop, BOARD_W, BOARD_H, exempt_top=True)


def test_non_puyo_crops_rejected_even_when_top_exempt() -> None:
    blank = np.zeros((BOARD_H+2*CROP_MARGIN, BOARD_W+2*CROP_MARGIN, 3), np.uint8)
    noise = np.random.default_rng(0).integers(0, 256, blank.shape, dtype=np.uint8)
    for crop in (blank, noise):
        assert not frame_border_matches(crop, BOARD_W, BOARD_H, exempt_top=True)


def test_options_default_off_equals_old_behavior() -> None:
    config = DeviceConfig('OBS', 0)
    assert config.relaxed_verify is False and config.verify_fail_streak == 1
    assert PuyoScreenVerifier.relaxed is False
    assert border_pair([(20, 20, 20, 80)], 0, 100, 100, exempt_first=False) == \
        border_pair([(20, 20, 20, 80)], 0, 100, 100)


@pytest.mark.parametrize('kwargs', [dict(verify_fail_streak=0), dict(verify_fail_streak=True),
                                    dict(relaxed_verify=1)])
def test_invalid_verifier_options(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DeviceConfig('OBS', 0, **kwargs)


class _MovingDevice(FakeDevice):
    """毎回画素が変わる (映像停止扱いを避ける)。"""

    def __init__(self) -> None:
        super().__init__(np.zeros((1080, 1920, 3), np.uint8))
        self.count = 0

    def read(self) -> tuple:
        self.count += 1
        self.image[0, 0, 0] = self.count % 251
        return True, self.image.copy()


def _run_streak(results: list[bool], streak: int, duration: float = 12.0) -> tuple[int, list[bool]]:
    """検証結果列を 1 秒周期で与え、(hold 回数, 各検証時点で通過中だったか) を返す。"""
    clock, holds, script = [0.0], [], list(results)
    verified_log: list[bool] = []
    def verifier(image: np.ndarray) -> bool:
        return script.pop(0) if script else True   # 列を使い切ったら以降は合格
    config = DeviceConfig('OBS', 0, verify_fail_streak=streak)
    source = DirectShowSource(config, duration, lambda: holds.append(clock[0]),
        lambda *args: _MovingDevice(), verifier, lambda: clock[0],
        lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    frames = list(source)
    verified_log.append(bool(frames))
    return len(holds), verified_log


def test_hysteresis_holds_only_after_k_consecutive_failures() -> None:
    entry = [True]
    assert _run_streak(entry + [False, False, True, True], 3)[0] == 0        # 2 連続では hold しない
    assert _run_streak(entry + [False, False, True, False, False, True], 3)[0] == 0  # 合格で連続数リセット
    assert _run_streak(entry + [False, False, False, False], 3)[0] > 0       # 3 連続で hold
    assert _run_streak(entry + [False, False, False, False], 5)[0] == 0      # K=5 なら 4 連続でも hold しない
    assert _run_streak(entry + [False], 1)[0] > 0                            # K=1 = 従来 (1 回で hold)


def test_entry_failure_holds_immediately_regardless_of_k() -> None:
    assert _run_streak([False], 5, 3.0)[0] > 0


def test_judge_transitions() -> None:
    source = DirectShowSource(DeviceConfig('OBS', 0, verify_fail_streak=3), 1.0, lambda: None)
    assert source._judge(True, False, 0) == (True, 0)
    assert source._judge(False, True, 0) == (True, 1)
    assert source._judge(False, True, 1) == (True, 2)
    assert source._judge(False, True, 2) == (False, 3)
    assert source._judge(False, False, 0) == (False, 1)  # 入口の不合格は即不合格
