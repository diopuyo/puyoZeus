"""予告アイコン CNN の 1 回 forward 化 (b3_warning_batch) と先読み (b3_warning_prefetch) が、
旧経路と同じ判定を返すこと・既定 OFF で旧経路のままであることを確かめる。"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pytest
import torch

FRAME_SHAPE = (1080, 1920, 3)


@pytest.fixture
def warning_assets(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """test_live_speed と同じ合成 CNN・合成テンプレ (未管理の学習資産に依存しない)。"""
    from src import ojama_cnn, ojama_warning
    seed = 23
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ojama_cnn.OjamaCnn().eval()
    template = np.random.default_rng(seed).integers(
        0, 256, (ojama_cnn.INPUT_HEIGHT, ojama_cnn.INPUT_WIDTH, 3), dtype=np.uint8)
    monkeypatch.setattr(ojama_cnn, 'load_cnn', lambda: deepcopy(model))
    monkeypatch.setattr(ojama_warning, '_load_templates',
                        lambda path: {ojama_warning.ICON_ROCK: [template.copy()]})
    monkeypatch.setenv('PUYO_IMAGE_FREE_ALL', '0')
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def frames() -> list[np.ndarray]:
    """乱数画素 (全セルが存在判定を通る)・平坦 (全セル empty)・半分平坦の 3 種。"""
    from src.ojama_warning import P2_BOARD_X
    rng = np.random.default_rng(7)
    noisy = rng.integers(0, 256, FRAME_SHAPE, dtype=np.uint8)
    flat = np.full(FRAME_SHAPE, 90, dtype=np.uint8)
    half = noisy.copy()
    half[:, P2_BOARD_X:] = 90
    return [noisy, flat, half, rng.integers(0, 256, FRAME_SHAPE, dtype=np.uint8)]


def detectors(threshold: float | None = None) -> tuple[Any, Any]:
    from src.ojama_warning import OjamaWarningDetector
    from src.batched_ojama_warning import BatchedOjamaWarningDetector
    serial, batched = OjamaWarningDetector(), BatchedOjamaWarningDetector()
    if threshold is not None:
        serial._cnn_confidence_min = batched._cnn_confidence_min = threshold
    return serial, batched


@pytest.mark.parametrize('threshold', [None, 0.0, 0.2, 2.0])
def test_batched_matches_serial(warning_assets: None, threshold: float | None) -> None:
    serial, batched = detectors(threshold)
    for frame in frames() * 2:  # 2 周目はキャッシュ命中の経路
        assert batched.detect(frame) == serial.detect(frame)
    for bad in (None, np.zeros((3, 3)), np.zeros((3, 3, 3), dtype=np.uint8)):
        assert batched.detect(bad) == serial.detect(bad)
    assert batched.stats['cache_hits'] > 0
    assert batched.stats['forwards'] >= 1


def test_borderline_cells_use_single_inference(warning_assets: None) -> None:
    """閾値ちょうどの confidence は単体推論で判定し直す (バッチの丸めで分岐が変わらない)。"""
    serial, batched = detectors()
    frame = frames()[0]
    patch = batched._extract_center_patch(batched._extract_cell(frame, 282, 0))
    _, conf = serial._cnn.predict_class_readonly(patch)
    serial._cnn_confidence_min = batched._cnn_confidence_min = conf
    assert batched.detect(frame) == serial.detect(frame)
    assert batched.stats['borderline'] >= 1


def test_flat_cells_skip_cnn(warning_assets: None) -> None:
    serial, batched = detectors()
    flat = frames()[1]
    assert batched.detect(flat) == serial.detect(flat)
    assert batched.stats['present_cells'] == 0 and batched.stats['forwards'] == 0


def test_prefetch_returns_same_values_and_isolates_pixels(warning_assets: None) -> None:
    from src.batched_ojama_warning import BatchedOjamaWarningDetector, PrefetchingWarning
    serial, _ = detectors()
    warning = PrefetchingWarning(BatchedOjamaWarningDetector())
    try:
        a, b, c, d = frames()
        expected_a = serial.detect(a)
        assert warning.prefetch(a)
        a_original = a.copy()
        a[:] = 0  # 先読みの後に画素が書き換わっても、先読みは複製を読む
        assert warning.detect(a) == expected_a
        a[:] = a_original
        assert warning.detect(b) == serial.detect(b)  # 先読みなし → 同期
        assert warning.prefetch(c)
        assert warning.detect(d) == serial.detect(d)  # 別コマの先読みは捨てる
        assert not warning.prefetch(None)
        assert warning.detect(None) == serial.detect(None)
        assert warning.stats['used'] == 1 and warning.stats['discarded'] == 1
    finally:
        warning.close()
        warning.close()
    assert all(not thread.is_alive() for thread in warning._pool._threads)


def test_observation_capture_wiring(warning_assets: None) -> None:
    from src.b3_observation import ObservationCapture
    from src.batched_ojama_warning import BatchedOjamaWarningDetector, PrefetchingWarning
    from src.parallel_ojama_warning import ParallelOjamaWarningDetector
    frame = frames()[0]
    reference = ObservationCapture()
    off = ObservationCapture(enable_warning_parallel=True)
    batch = ObservationCapture(enable_warning_parallel=True, enable_warning_batch=True)
    prefetch = ObservationCapture(enable_warning_batch=True, warning_prefetch=True)
    try:
        assert isinstance(off.warning, ParallelOjamaWarningDetector)
        assert isinstance(batch.warning, BatchedOjamaWarningDetector)
        assert isinstance(prefetch.warning, PrefetchingWarning)
        assert not reference.prefetch_warning(frame) and not batch.prefetch_warning(frame)
        expected = reference.warning.detect(frame)
        assert batch.warning.detect(frame) == expected
        assert prefetch.prefetch_warning(frame)
        assert prefetch.warning.detect(frame) == expected
    finally:
        for capture in (reference, off, batch, prefetch):
            capture.close()


def test_speed_flags_default_off_and_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import os
    from src.phase_j.live_speed import SPEED_DEFAULTS, configure, observation_options
    assert SPEED_DEFAULTS['b3_warning_batch'] is False
    assert SPEED_DEFAULTS['b3_warning_prefetch'] is False
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    parser = argparse.ArgumentParser()
    base = dict(source='video', cpu_threads=1)
    with pytest.raises(SystemExit):
        configure(argparse.Namespace(**base, b3_warning_prefetch=True), parser)
    configure(argparse.Namespace(**base), parser)
    assert observation_options() == {}
    configure(argparse.Namespace(**base, b3_warning_batch=True), parser)
    assert observation_options() == dict(enable_warning_batch=True)
    configure(argparse.Namespace(**base, b3_warning_batch=True, b3_warning_prefetch=True), parser)
    assert observation_options() == dict(enable_warning_batch=True, warning_prefetch=True)


def test_launcher_passes_flags_only_when_on(tmp_path: Any) -> None:
    from src.phase_j.launcher import build_pipeline_config, parse_user_config
    off = build_pipeline_config(parse_user_config(dict(device_name='test'), tmp_path))
    assert 'b3_warning_batch' not in off and 'b3_warning_prefetch' not in off
    on = build_pipeline_config(parse_user_config(
        dict(device_name='test', b3_warning_batch=True, b3_warning_prefetch=True), tmp_path))
    assert on['b3_warning_batch'] is True and on['b3_warning_prefetch'] is True


def test_live_b3_prefetch_needs_current_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_b3
    calls: list[Any] = []
    observation = SimpleNamespace(prefetch_warning=lambda image: calls.append(image) or True)
    frame = SimpleNamespace(image='image')
    pipe = SimpleNamespace(epoch=3, publishing_ready=True)
    monkeypatch.delenv(live_b3.ENV_ENABLED, raising=False)
    assert not live_b3.prefetch_warning(pipe, frame)
    monkeypatch.setenv(live_b3.ENV_ENABLED, '1')
    assert not live_b3.prefetch_warning(pipe, frame)  # 観測器がまだ無い
    pipe._b3_capture, pipe._b3_epoch = observation, (2, True)
    assert not live_b3.prefetch_warning(pipe, frame)  # 世代が違う
    pipe._b3_epoch = (3, True)
    assert live_b3.prefetch_warning(SimpleNamespace(pipe=pipe, epoch=3, publishing_ready=True), frame)
    assert calls == ['image']
