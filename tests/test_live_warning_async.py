"""遅い予告 CNN の待機をイベントで制御し、認識が待たないことを検査する。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.async_ojama_warning import AsyncWarning
from src.ojama_warning import OjamaWarningResult
from tests.test_live_warning_batch import warning_assets

TIMEOUT = 5.0


class SlowDetector:
    def __init__(self) -> None:
        self.entered, self.release = Event(), Event()
        self.values: list[int] = []

    def extract_cells(self, image: Any, copy: bool = False) -> list:
        return [image.copy() if copy else image]

    def detect_cells(self, cells: list) -> tuple:
        self.entered.set()
        assert self.release.wait(TIMEOUT), '試験の推論解除が届かない'
        value = int(cells[0].flat[0])
        self.values.append(value)
        return tuple(OjamaWarningResult(s, (), value) for s in ('1P', '2P'))


def completed(worker: AsyncWarning, fi: int) -> None:
    with worker._condition:
        assert worker._condition.wait_for(
            lambda: worker._source is not None and worker._source[0] == fi, TIMEOUT)


def test_latest_only_copied_and_nonblocking() -> None:
    detector = SlowDetector()
    worker = AsyncWarning(detector)
    image = np.array([7])
    try:
        first, meta = worker.update(image, 10, 1.0)
        assert [v.total_count for v in first] == [0, 0] and not meta['initialized']
        assert detector.entered.wait(TIMEOUT)
        image[:] = 99
        for fi in (11, 12):
            result, meta = worker.update(np.array([fi]), fi, fi/10)
            assert result == first and not meta['initialized']
        detector.release.set()
        completed(worker, 12)
        assert detector.values == [7, 12]
        detector.release.clear()
        result, meta = worker.update(image, 20, 2.0)
        assert [v.total_count for v in result] == [12, 12]
        assert meta == dict(source_frame=12, source_t_sec=1.2, lag_frames=8,
                            lag_sec=0.8, initialized=True)
        held, meta = worker.update(image, 21, 2.1)
        assert held == result and meta['source_frame'] == 12
    finally:
        detector.release.set()
        worker.close()
        worker.close()
    assert not worker._thread.is_alive()


def test_close_discards_pending_and_joins() -> None:
    detector = SlowDetector()
    worker = AsyncWarning(detector)
    worker.update(np.array([1]), 1, 0.1)
    assert detector.entered.wait(TIMEOUT)
    worker.update(np.array([2]), 2, 0.2)
    closer = Thread(target=worker.close)
    closer.start()
    try:
        with worker._condition:
            assert worker._condition.wait_for(lambda: worker._closed, TIMEOUT)
        assert closer.is_alive()
    finally:
        detector.release.set()
        closer.join(TIMEOUT)
    assert not closer.is_alive() and not worker._thread.is_alive()
    assert detector.values == [1]
    with pytest.raises(RuntimeError, match='終了済み'):
        worker.update(np.array([3]), 3, 0.3)


@dataclass
class Snapshot:
    value: int = 0


def capture_stub(monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> tuple:
    from src import b3_observation as obs
    capture = obs.ObservationCapture.__new__(obs.ObservationCapture)
    detector = SlowDetector()
    capture.warning = SimpleNamespace(detect=lambda image: detector.detect_cells([image]))
    capture._warning_async = AsyncWarning(detector) if asynchronous else None
    capture.ocr = SimpleNamespace(read=lambda image: None)
    capture.boundary = lambda scores: None
    capture.account, capture.states, capture.previous = None, (None, None), (None, None)
    capture.game = 0
    capture.side = lambda s, *args: dict(side=s.side)
    capture.observer = SimpleNamespace(update=lambda row: (row['sides'][0]['icon']['total_count'],))
    monkeypatch.setattr(obs, '_drive_ojama_accounting_lean', lambda *args: Snapshot())
    side = lambda name: SimpleNamespace(side=name, state=obs.BoardState.STABLE)
    result = SimpleNamespace(p1=side('1P'), p2=side('2P'), is_match_active=True, match_end_locked=False)
    return capture, detector, result


@pytest.mark.parametrize('asynchronous', [False, True])
def test_observation_result_and_derived_event(monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> None:
    capture, detector, result = capture_stub(monkeypatch, asynchronous)
    try:
        if not asynchronous:
            detector.release.set()
        row, events = capture.update(result, np.array([7]), 1, 0.1, (0, 0))
        assert events == ((0,) if asynchronous else (7,))
        assert ('warning_async' in row) is asynchronous
        if asynchronous:
            assert detector.entered.wait(TIMEOUT)
            detector.release.set()
            completed(capture._warning_async, 1)
            detector.release.clear()
            row, events = capture.update(result, np.array([8]), 2, 0.2, (0, 0))
            assert events == (7,) and row['warning_async']['lag_frames'] == 1
            row, events = capture.update(result, np.array([9]), 3, 0.3, (0, 0))
            assert events == (7,) and row['warning_async']['source_frame'] == 1
    finally:
        detector.release.set()
        capture.close()


def test_flags_and_child_delivery(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    import os
    from src.phase_j import live_speed as speed
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    assert speed.SPEED_DEFAULTS['b3_warning_async'] is False
    start = speed.TRIAL_FLAG_KEYS.index('b3_warning_async')
    assert speed.TRIAL_FLAG_KEYS[start:start+3] == ('b3_warning_async', 'live_native_hsv_main', 'live_verify_async')   # 後ろに別の鍵が続いてよい
    data = dict(device_name='試験', b3_warning_async=True, b3_warning_prefetch=True)
    assert build_pipeline_config(parse_user_config(data, tmp_path))['b3_warning_async']
    options = argparse.Namespace(source='dshow', cpu_threads=1, **data)
    speed.configure(options, argparse.ArgumentParser())
    assert speed.observation_options() == dict(enable_warning_async=True)
    assert speed.trial_flags_record(options)['warning_execution'] == 'async_latest_batched_without_prefetch'
    speed.configure(argparse.Namespace(source='dshow', cpu_threads=1), argparse.ArgumentParser())
    assert speed.observation_options() == {}


def test_diagnostic_spool_includes_lag() -> None:
    from src.phase_j.live_observation_diagnostics import archive_terminal_observation
    rows: list = []
    owner = SimpleNamespace(_live_terminal_observations=rows)
    notice = SimpleNamespace(frame=3, t_sec=0.3, captured_at=1., recognized_at=2.,
                             b3=({'warning_async': {'source_frame': 1, 'lag_frames': 2}},))
    archive_terminal_observation(owner, notice, True)
    assert rows[0]['warning_async']['lag_frames'] == 2


@pytest.mark.parametrize('batch,prefetch,skip', [(False, False, False), (True, True, False), (True, True, True)])
def test_constructor_priority(warning_assets: None, batch: bool, prefetch: bool, skip: bool) -> None:
    from src.b3_observation import ObservationCapture
    from src.batched_ojama_warning import BatchedOjamaWarningDetector
    capture = ObservationCapture(enable_warning_parallel=True, enable_warning_batch=batch,
                                 warning_prefetch=prefetch, enable_warning_async=True,
                                 skip_icon_reading=skip)
    try:
        assert not capture.prefetch_warning(None)
        if skip:
            assert capture.warning is None and capture._warning_async is None
        else:
            assert isinstance(capture.warning, BatchedOjamaWarningDetector)
            assert capture._warning_async is not None
    finally:
        capture.close()


def test_worker_failure_is_visible() -> None:
    detector = SlowDetector()
    def fail(cells: list) -> tuple:
        raise ValueError('疑似障害')
    detector.detect_cells = fail
    worker = AsyncWarning(detector)
    try:
        worker.update(np.array([1]), 1, 0.1)
        worker._thread.join(TIMEOUT)
        with pytest.raises(RuntimeError, match='推論失敗'):
            worker.update(np.array([2]), 2, 0.2)
    finally:
        worker.close()
