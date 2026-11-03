"""C群の周期・遅延・画素・取得所有権を検査する。"""
from __future__ import annotations
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from src.board import Board
from src.live_speed_c import DeferredHsv, boundary_detect, resolve_hsv
from src.phase_j.live_color_roi import RegionCorrectedImage, full_image, correct_input
from src.phase_j.dshow_color import correct_601_as_709


class Detector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: object) -> int:
        self.calls += 1
        return self.calls


def test_boundary_off_reads_every_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_LIVE_BOUNDARY_HALF_SECOND', raising=False)
    detector, pipe = Detector(), SimpleNamespace()
    assert [boundary_detect(pipe, detector, None, t, 'match') for t in (0., .1, .2)] == [1, 2, 3]


def test_boundary_period_rewind_reset_and_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_BOUNDARY_HALF_SECOND', '1')
    detector, pipe = Detector(), SimpleNamespace()
    assert [boundary_detect(pipe, detector, None, t, 'match') for t in (0., .49, .5, .99, 1.)] == [1, 1, 2, 2, 3]
    assert boundary_detect(pipe, detector, None, .1, 'match') == 4
    pipe._live_boundary_cache = None
    assert boundary_detect(pipe, detector, None, .2, 'match') == 5
    assert boundary_detect(pipe, Detector(), None, .21, 'match') == 1


def test_boundary_keys_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_BOUNDARY_HALF_SECOND', '1')
    detector, pipe = Detector(), SimpleNamespace()
    assert boundary_detect(pipe, detector, None, 0., 'match') == 1
    assert boundary_detect(pipe, detector, None, 0., 'zero') == 2


def test_hsv_not_used_and_used_once() -> None:
    calls = []
    board = Board()
    lazy = DeferredHsv(lambda: calls.append(True) or board)
    assert not calls
    assert resolve_hsv(lazy) is board
    assert resolve_hsv(lazy) is board
    assert len(calls) == 1
    assert resolve_hsv(None) is None


def test_hsv_failure_is_cached_none() -> None:
    calls = []
    def fail() -> Board:
        calls.append(True)
        raise RuntimeError('test')
    lazy = DeferredHsv(fail)
    assert resolve_hsv(lazy) is None
    assert resolve_hsv(lazy) is None
    assert len(calls) == 1


@pytest.mark.parametrize('shape', [(11, 17, 3), (1080, 1920, 3)])
def test_color_regions_and_full_exact(shape: tuple[int, ...]) -> None:
    raw = np.random.default_rng(12).integers(0, 256, shape, dtype=np.uint8)
    before = raw.copy()
    expected = correct_601_as_709(raw)
    image = RegionCorrectedImage(raw)
    for key in ((slice(1, 9), slice(3, 15)), (slice(None, None, 8), slice(None, None, 8)), (slice(0, 0), slice(None))):
        assert np.array_equal(image[key], expected[key])
    assert image.complete is None
    assert np.array_equal(full_image(image), expected)
    assert np.array_equal(image.copy(), expected)
    assert np.array_equal(np.asarray(image), expected)
    assert np.array_equal(raw, before)


def test_color_off_and_input_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = np.full((10, 12, 3), 91, dtype=np.uint8)
    monkeypatch.delenv('PUYO_LIVE_COLOR_ROI', raising=False)
    assert type(correct_input(raw)) is np.ndarray
    image = RegionCorrectedImage(raw)
    expected = correct_601_as_709(raw)
    raw[:] = 0
    assert np.array_equal(image.copy(), expected)


def test_c_flags_are_off_and_roundtrip() -> None:
    from src.production_config import LIVE_C_ITEMS_DEFAULTS, LIVE_C_ITEMS_CANDIDATES
    from src.phase_j.live_speed import validate
    assert not any(LIVE_C_ITEMS_DEFAULTS.values())
    assert all(not candidate.adopted for candidate in LIVE_C_ITEMS_CANDIDATES)
    for key in LIVE_C_ITEMS_DEFAULTS:
        assert validate({key: True})[key] is True


def test_capture_thread_independent_of_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_device import DeviceConfig, DirectShowSource
    from src.phase_j.live_capture_buffer import BufferedCapture
    monkeypatch.delenv('PUYO_EVENT_PRIORITY', raising=False)
    sentinel = object()
    source = DirectShowSource(DeviceConfig('test', 0), 1., lambda: None, capture_factory=lambda *args: sentinel)
    assert source._capture() is sentinel
    monkeypatch.setenv('PUYO_LIVE_CAPTURE_THREAD', '1')
    source = DirectShowSource(DeviceConfig('test', 0), 1., lambda: None, capture_factory=lambda *args: sentinel)
    assert not source.event_priority.enabled
    assert isinstance(source._capture(), BufferedCapture)


def test_capture_worker_owns_device_and_releases() -> None:
    from src.phase_j.live_capture_buffer import BufferedCapture
    from src.phase_j.live_degrade import EventPriority
    events = []
    class Capture:
        def isOpened(self) -> bool:
            return True
        def read(self) -> tuple[bool, np.ndarray]:
            events.append(('read', threading.get_ident()))
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        def release(self) -> None:
            events.append(('release', threading.get_ident()))
    def factory() -> Capture:
        events.append(('create', threading.get_ident()))
        return Capture()
    capture = BufferedCapture(factory, (), EventPriority())
    try:
        assert capture.isOpened()
        assert capture.read()[0]
        assert capture.last_captured_at is not None
    finally:
        capture.release()
    assert {event[1] for event in events} == {capture.thread.ident}
    assert events[-1][0] == 'release'
    assert capture.thread.ident != threading.get_ident()

def test_capture_metadata_is_read_on_owner_thread() -> None:
    import cv2
    from src.phase_j.live_capture_buffer import BufferedCapture
    from src.phase_j.live_degrade import EventPriority
    ids = []
    fourcc = cv2.VideoWriter_fourcc(*'NV12')
    class Capture:
        def isOpened(self) -> bool:
            return True
        def get(self, key: int) -> float:
            ids.append(threading.get_ident())
            return float(fourcc)
        def read(self) -> tuple[bool, None]:
            return False, None
        def release(self) -> None:
            pass
    capture = BufferedCapture(Capture, (), EventPriority(), expose_metadata=True)
    try:
        assert capture.get(cv2.CAP_PROP_FOURCC) == fourcc
        assert ids == [capture.thread.ident]
    finally:
        capture.release()


def test_color_full_verifier_and_continuity() -> None:
    from src.phase_j.distribution_input import has_feedback
    from src.phase_j.live_input_guard import FrameContinuity
    raw = np.random.default_rng(45).integers(0,256,(80,120,3),dtype=np.uint8)
    full, region = correct_601_as_709(raw), RegionCorrectedImage(raw)
    assert has_feedback(region) == has_feedback(full)
    assert region.complete is not None
    a,b = FrameContinuity(1.),FrameContinuity(1.)
    region = RegionCorrectedImage(raw)
    assert a.ready(full,(120,80),0.) == b.ready(region,(120,80),0.)
    assert a.digest == b.digest
    assert region.complete is None


def test_hsv_lazy_top_ojama_matches_eager() -> None:
    from src.ojama_visual_detector import _count_top_ojama
    cnn, hsv = Board(), Board()
    hsv.set(1, 2, 9)
    assert _count_top_ojama(cnn, DeferredHsv(lambda: hsv)) == _count_top_ojama(cnn,hsv)

def test_color_native_reads_only_declared_regions() -> None:
    from src.phase_j.live_color_roi import native_image
    raw = np.random.default_rng(44).integers(0,256,(80,120,3),dtype=np.uint8)
    image = RegionCorrectedImage(raw)
    expected = correct_601_as_709(raw)
    rects = [(5,6,20,28),(25,10,32,40)]
    actual = native_image(image,rects)
    assert actual.flags.c_contiguous
    for x1,y1,x2,y2 in rects:
        assert np.array_equal(actual[y1:y2,x1:x2],expected[y1:y2,x1:x2])
    assert image.complete is None
    assert np.array_equal(image.full(), expected)
