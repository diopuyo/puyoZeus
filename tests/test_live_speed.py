"""配信専用の高速化が旧経路と同じ決定を返すことを検証する。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Iterator

import numpy as np
import pytest
import torch


@pytest.fixture
def warning_assets(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """経路比較用の同一CNN・合成テンプレを用意し、未管理の学習資産に依存しない。"""
    from src import ojama_cnn, ojama_warning

    seed = 23
    pixel_values = 256
    channels = 3
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ojama_cnn.OjamaCnn().eval()
    template = np.random.default_rng(seed).integers(
        0, pixel_values, (ojama_cnn.INPUT_HEIGHT, ojama_cnn.INPUT_WIDTH, channels),
        dtype=np.uint8,
    )
    # キャッシュは各検出器が所有するため、モデルを共有しない。
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


def test_speed_flags_are_opt_in() -> None:
    from src import production_config as config
    assert config.LIVE_DISABLE_PLACEMENT_RECONCILE_DEFAULT is False
    assert config.B3_SCORE_MATMUL_DEFAULT is False
    assert config.B3_WARNING_PARALLEL_DEFAULT is False
    assert config.B3_WARNING_WORKERS == config.B3_WARNING_MAX_WORKERS == 4


def test_readonly_cnn_matches_serial_without_changing_model() -> None:
    from src.ojama_cnn import OjamaCnn
    torch.set_num_threads(1)
    model = OjamaCnn().eval()
    patches = np.random.default_rng(12).integers(0, 256, (12, 36, 36, 3), dtype=np.uint8)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    expected = [model.predict_class(patch) for patch in patches]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(model.predict_class_readonly, patches))
    assert actual == expected
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize('thread_safe', [False, True])
def test_cache_preserves_lru_and_none(thread_safe: bool) -> None:
    from scripts.b2_recollect_cache import memoized
    calls: list[int] = []

    def predict(key: int) -> int | None:
        calls.append(key)
        return key if key else None

    cached = memoized(predict, 2, lambda key: key, thread_safe=thread_safe)
    assert [cached(k) for k in (0, 1, 0, 2, 1)] == [None, 1, None, 2, 1]
    assert calls == [0, 1, 2, 1]


def test_parallel_cache_eviction_and_duplicate_misses() -> None:
    from scripts.b2_recollect_cache import memoized
    from threading import Barrier
    barrier = Barrier(4)

    def predict(key: int) -> int:
        barrier.wait(timeout=5)
        return key * 2

    cached = memoized(predict, 1, lambda key: key, thread_safe=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(cached, [1, 1, 2, 3])) == [2, 2, 4, 6]


def test_parallel_cache_does_not_cache_exceptions() -> None:
    from scripts.b2_recollect_cache import memoized

    def fail(key: int) -> Any:
        raise ValueError(key)

    cached = memoized(fail, 2, lambda key: key, thread_safe=True)
    for _ in range(2):
        with pytest.raises(ValueError, match='7'):
            cached(7)


@pytest.mark.parametrize('workers', [0, 5, True, 1.5])
def test_warning_rejects_invalid_worker_count(workers: Any) -> None:
    from src.parallel_ojama_warning import ParallelOjamaWarningDetector
    with pytest.raises(ValueError, match='worker'):
        ParallelOjamaWarningDetector(workers)


def test_warning_parallel_preserves_order_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ojama_warning import OjamaWarningDetector
    from src.parallel_ojama_warning import ParallelOjamaWarningDetector
    serial = OjamaWarningDetector()
    parallel = ParallelOjamaWarningDetector()
    # 全セルをCNN低信頼へ強制し、NCC/HSV fallbackも同じ順序で返すことを見る。
    for warning in (serial, parallel):
        monkeypatch.setattr(warning, '_cnn_confidence_min', 2.0)
    frame = np.random.default_rng(19).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    try:
        assert parallel.detect(frame) == serial.detect(frame)
        for bad in (None, np.zeros((3, 3)), np.zeros((3, 3, 3))):
            assert parallel.detect(bad) == serial.detect(bad)
    finally:
        parallel.close()
        parallel.close()
    assert all(not thread.is_alive() for thread in parallel._pool._threads)


@pytest.mark.parametrize('matmul,parallel', [(False, False), (True, False), (False, True), (True, True)])
def test_observation_uses_opt_in_paths(
    matmul: bool, parallel: bool, warning_assets: None,
) -> None:
    from src.b3_observation import ObservationCapture
    from src.parallel_ojama_warning import ParallelOjamaWarningDetector
    capture = ObservationCapture(matmul, parallel)
    try:
        assert capture.ocr._enable_matmul_ncc is matmul
        assert isinstance(capture.warning, ParallelOjamaWarningDetector) is parallel
        frame = np.random.default_rng(23).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        reference = ObservationCapture()
        try:
            assert capture.warning.detect(frame) == reference.warning.detect(frame)
            assert capture.warning.detect(frame) == reference.warning.detect(frame)
        finally:
            reference.close()
    finally:
        capture.close()


@pytest.mark.parametrize('key', ['live_disable_placement_reconcile', 'b3_score_matmul', 'b3_warning_parallel'])
@pytest.mark.parametrize('bad', [1, 'true', None])
def test_speed_bool_validation(key: str, bad: Any) -> None:
    from src.phase_j.live_speed import validate
    with pytest.raises(ValueError, match=key):
        validate({key: bad})


def _clear_recognition_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """配布の既定 (W68 は 0.6.5 で ON) が環境に残っていても、R1b の対象だけを見る。"""
    for name in ('PUYO_LIVE_DISABLE_PLACEMENT_RECONCILE', 'PUYO_W68_INFERENCE_PENDING', 'PUYO_W68_ROLLBACK',
                 'PUYO_LIVE_ROW1_GUARD', 'PUYO_ALIGN_SCORE_ROI', 'PUYO_LIVE_MATCH_END_PAIR'):
        monkeypatch.delenv(name, raising=False)


def test_live_r1b_override_leaves_video_and_defaults_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_speed import recognition_config
    original = dict(enable_placement_signal_reconcile=True, stable_frame_count=3)
    _clear_recognition_flags(monkeypatch)
    assert recognition_config(original, True) is original
    monkeypatch.setenv('PUYO_LIVE_DISABLE_PLACEMENT_RECONCILE', '1')
    assert recognition_config(original, False) is original
    assert recognition_config(original, True) == dict(original, enable_placement_signal_reconcile=False)
    assert original['enable_placement_signal_reconcile'] is True


def test_speed_env_resets_between_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import os
    from src.phase_j.live_speed import configure, observation_options
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    options = argparse.Namespace(source='dshow', cpu_threads=1, b3_warning_parallel=True,
        b3_score_matmul=True, live_disable_placement_reconcile=True)
    configure(options, argparse.ArgumentParser())
    assert observation_options() == dict(enable_score_matmul=True, enable_warning_parallel=True, warning_workers=4)
    options = argparse.Namespace(source='video', cpu_threads=1, live_disable_placement_reconcile=True)
    configure(options, argparse.ArgumentParser())
    assert observation_options() == {}
    assert os.environ['PUYO_LIVE_DISABLE_PLACEMENT_RECONCILE'] == '0'


def test_capture_close_works_through_device_proxy() -> None:
    from types import SimpleNamespace
    from src.phase_j.live_b3 import close_capture
    calls: list[bool] = []
    owner = SimpleNamespace(_b3_capture=SimpleNamespace(close=lambda: calls.append(True)))
    close_capture(SimpleNamespace(pipe=owner))
    close_capture(SimpleNamespace())
    assert calls == [True]


def test_bridge_saves_effective_r1b_setting(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from src.phase_j.live_process import ProcessRecognitionBridge
    bridge = object.__new__(ProcessRecognitionBridge)
    bridge.device = 'device.json'
    bridge.config_path = tmp_path / 'recognition.json'
    _clear_recognition_flags(monkeypatch)
    monkeypatch.setenv('PUYO_LIVE_DISABLE_PLACEMENT_RECONCILE', '1')
    actual = bridge.pipeline_config(enable_placement_signal_reconcile=True)
    assert actual == {'enable_placement_signal_reconcile': False}
    assert json.loads(bridge.config_path.read_text()) == actual
    bridge.device = None
    assert bridge.pipeline_config(enable_placement_signal_reconcile=True)['enable_placement_signal_reconcile']


def test_launcher_maps_only_explicit_speed_flags(tmp_path: Any) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config, LauncherConfigError
    from src.phase_j.live_speed import SPEED_DEFAULTS, WORKERS_KEY
    data = dict(device_name='test')
    default_on = {key for key, value in SPEED_DEFAULTS.items() if value}  # 本番ON済み (live_row1_guard 等)
    assert set((*SPEED_DEFAULTS, WORKERS_KEY)) & build_pipeline_config(parse_user_config(data, tmp_path)).keys() == default_on
    data.update({key: True for key in SPEED_DEFAULTS}, b3_warning_workers=3)
    actual = build_pipeline_config(parse_user_config(data, tmp_path))
    assert all(actual[key] is True for key in SPEED_DEFAULTS)
    assert actual[WORKERS_KEY] == 3
    with pytest.raises(LauncherConfigError, match=WORKERS_KEY):
        parse_user_config(dict(data, b3_warning_workers=True), tmp_path)


def test_cli_delivers_speed_flags_without_opening_input(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import os
    import sys
    from scripts.run_live_pipeline_20260928 import parse_args
    from src.phase_j.live_speed import observation_options
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(dict(source='dshow', name='test', index=0, cpu_threads=1,
        live_disable_placement_reconcile=True, b3_warning_parallel=True, b3_score_matmul=True)))
    monkeypatch.setattr(sys, 'argv', ['live', '--config', str(path)])
    parse_args()
    assert os.environ['PUYO_LIVE_DISABLE_PLACEMENT_RECONCILE'] == '1'
    assert observation_options() == dict(enable_score_matmul=True, enable_warning_parallel=True, warning_workers=4)
    monkeypatch.setattr(sys, 'argv', ['live', '--config', str(path), '--cpu-threads', '2'])
    with pytest.raises(SystemExit):
        parse_args()
