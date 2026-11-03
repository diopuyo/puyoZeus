"""非同期の組照合・時刻・最新一件・停止・解除を検証する。"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.async_match_end_pair import AsyncMatchEndPairDetector, PAIR_SCAN_INTERVAL_SEC, close_pair_worker
from src.match_end_detector import MatchEndDetector
from src.match_end_pair import PAIR_ROIS
from src.recognition_pipeline import RecognitionPipeline
from src.phase_j.live_observation_diagnostics import recognition_diagnostics
from tests.test_lockdown_score_zero_then_move import _make_pipe, _run, FINAL, ZERO
from tests.test_recognition_pipeline import _StubImageReader, _StubMatchDetector, _empty_board

FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)
WAIT_SEC = 3.0


@pytest.fixture
def detector() -> Iterator[AsyncMatchEndPairDetector]:
    value = AsyncMatchEndPairDetector({})
    yield value
    value.close()


def _completed(detector: AsyncMatchEndPairDetector, stamp: float) -> None:
    with detector._condition:
        assert detector._condition.wait_for(
            lambda: detector._completed is not None and detector._completed[0] == stamp,
            timeout=WAIT_SEC)


@pytest.mark.parametrize('winner', ['1P', '2P'])
def test_real_ncc_winner_and_source_time(detector: AsyncMatchEndPairDetector, winner: str) -> None:
    rng = np.random.default_rng(42)
    detector._templates = {kind: rng.integers(0, 255, (20, 35), dtype=np.uint8)
                           for kind in ('yatta', 'batan')}
    frame = FRAME.copy()
    for kind, side in [('yatta', winner), ('batan', '2P' if winner == '1P' else '1P')]:
        x, y, _, _ = PAIR_ROIS[f'{kind}_{side.lower()}']
        frame[y:y+20, x:x+35] = cv2.cvtColor(detector._templates[kind], cv2.COLOR_GRAY2BGR)
    assert not detector.update(frame, 10)
    _completed(detector, 10)
    assert detector.update(FRAME, 10.2)
    assert detector.last_detection.winner == winner
    assert detector.lock_started_t == detector.last_detected_t == 10
    owner = SimpleNamespace(_match_end_detector=detector, _enable_match_end_pair=True)
    diagnostic = recognition_diagnostics(owner)['match_end']
    assert diagnostic['observed_t_sec'] == 10
    assert diagnostic['ncc_observed_t_sec'] == dict.fromkeys(PAIR_ROIS, 10)
    assert diagnostic['scanned_roi'] == 'all'
    assert detector.is_locked(100)


@pytest.mark.parametrize('keys', [(key,) for key in PAIR_ROIS] + [tuple(PAIR_ROIS)])
def test_single_or_ambiguous_never_locks(detector: AsyncMatchEndPairDetector,
                                        keys: tuple[str, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector, '_score_crop', lambda crop, key: 0.9 if key in keys else 0.1)
    detector.update(FRAME, 1)
    _completed(detector, 1)
    assert not detector.update(FRAME, 2)


def test_nonblocking_latest_copy_and_interval(detector: AsyncMatchEndPairDetector,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    entered, unblock = Event(), Event()
    original = detector._evaluate
    starts: list[float] = []
    samples: list[tuple[float, int]] = []

    def slow(crops: dict[str, np.ndarray], stamp: float) -> object:
        starts.append(monotonic())
        entered.set()
        assert unblock.wait(WAIT_SEC)
        samples.append((stamp, int(crops['yatta_1p'][0, 0, 0])))
        return original(crops, stamp)

    monkeypatch.setattr(detector, '_evaluate', slow)
    image = FRAME.copy()
    try:
        detector.update(image, 1)
        assert entered.wait(WAIT_SEC)
        returned = Event()
        caller = Thread(target=lambda: (detector.update(image, 2), returned.set()))
        caller.start()
        assert returned.wait(WAIT_SEC), '照合が止まっていても認識側は戻る'
        caller.join()
        image.fill(7)
        detector.update(image, 3)
        image.fill(9)
        unblock.set()
        _completed(detector, 3)
        assert samples == [(1, 0), (3, 7)]
        assert starts[1] - starts[0] >= PAIR_SCAN_INTERVAL_SEC
    finally:
        unblock.set()


@pytest.mark.parametrize('reset', [False, True])
def test_release_discards_inflight_result(detector: AsyncMatchEndPairDetector, reset: bool,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    entered, unblock, finished = Event(), Event(), Event()

    def delayed(crops: dict[str, np.ndarray], stamp: float) -> object:
        entered.set()
        assert unblock.wait(WAIT_SEC)
        finished.set()
        scores = dict(yatta_1p=0.9, batan_2p=0.9)
        return detector._result(scores, scores)

    monkeypatch.setattr(detector, '_evaluate', delayed)
    try:
        detector.update(FRAME, 1)
        assert entered.wait(WAIT_SEC)
        detector.update(FRAME, 2)
        detector.reset() if reset else detector.release()
        unblock.set()
        assert finished.wait(WAIT_SEC)
        detector.close()
        assert detector._completed is None and detector._pending is None
        assert not detector.is_locked(3)
    finally:
        unblock.set()


def test_pipeline_zero_twice_releases(detector: AsyncMatchEndPairDetector,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, _ = _make_pipe([FINAL, FINAL, ZERO, ZERO], False, enable_match_end_pair=True)
    pipe._match_end_detector = detector
    monkeypatch.setattr(detector, '_score_crop',
                        lambda crop, key: 0.9 if key in ('yatta_1p', 'batan_2p') else 0.1)
    _run(pipe, 1)
    _completed(detector, 5)
    _run(pipe, 1, start=1)
    assert detector.is_locked(10)
    assert pipe._post_match_lockdown_started_time == 5
    _run(pipe, 1, start=2)
    assert detector.is_locked(10)
    _run(pipe, 1, start=3)
    assert not detector.is_locked(10)


def test_close_worker_joins_and_is_idempotent(detector: AsyncMatchEndPairDetector) -> None:
    detector.update(FRAME, 1)
    owner = SimpleNamespace(_match_end_detector=detector, _enable_match_end_pair=True)
    close_pair_worker(SimpleNamespace(pipe=owner))
    close_pair_worker(owner)
    assert not detector._thread.is_alive()
    with pytest.raises(RuntimeError, match='終了済み'):
        detector.update(FRAME, 2)


def test_factory_and_pipeline_only_on_use_worker(tmp_path: Path) -> None:
    for enabled in (False, True):
        value = MatchEndDetector.load_default(Path(str(tmp_path)), pair_mode=enabled)
        assert isinstance(value, AsyncMatchEndPairDetector) is enabled
        pipe = RecognitionPipeline(
            image_reader=_StubImageReader(_empty_board(), _empty_board()),
            match_state_detector=_StubMatchDetector(in_match=False),
            enable_match_end_pair=enabled, match_end_detector=MatchEndDetector({}))
        assert isinstance(pipe._match_end_detector, AsyncMatchEndPairDetector) is enabled


def test_thread_bridge_interrupt_closes_worker(detector: AsyncMatchEndPairDetector,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_bridge
    detector.update(FRAME, 1)
    owner = SimpleNamespace(_match_end_detector=detector, _enable_match_end_pair=True)

    def interrupted() -> Iterator[object]:
        raise KeyboardInterrupt
        yield None

    bridge = live_bridge.RecognitionBridge(False, lambda *args: None)
    bridge.source = interrupted()
    monkeypatch.setattr(live_bridge, 'recognition_guard_audits', lambda *args: {})
    bridge._recognize(owner)
    assert isinstance(bridge.error, KeyboardInterrupt)
    assert bridge.done.is_set()
    assert not detector._thread.is_alive()


def test_process_input_interrupt_closes_worker(detector: AsyncMatchEndPairDetector,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_process, live_device_session, live_stage_timing
    detector.update(FRAME, 1)
    owner = SimpleNamespace(_match_end_detector=detector, _enable_match_end_pair=True)
    monkeypatch.setattr(live_process, 'configure_media_tick', lambda *args: None)
    monkeypatch.setattr(live_device_session, 'DeviceSession',
                        lambda *args: SimpleNamespace(pipe=owner, source=()))
    monkeypatch.setattr(live_process, 'recognition_logs', lambda *args: None)
    monkeypatch.setattr(live_stage_timing, 'install', lambda *args: None)

    def interrupted(*args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(live_process, 'send_notices', interrupted)
    with pytest.raises(KeyboardInterrupt):
        live_process.run_recognition_input(None, Event(), owner, '', (30, 0, 30, 1),
                                           False, 'unused', None, False, None, None, None)
    assert not detector._thread.is_alive()
