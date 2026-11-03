"""保持先のaliasと丸め結果を確認する。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np

from src.phase_j.dshow_color import correct_601_as_709
from src.phase_j.live_color_buffer import BUFFER_COUNT, ColorBuffer
from src.phase_j.live_color_buffer import configured_corrector

SHAPE = (108, 192, 3)
FRAME_COUNT = 20


def test_default_off_and_explicit_flag(monkeypatch: object) -> None:
    monkeypatch.delenv('PUYO_LIVE_COLOR_BUFFER', raising=False)
    assert configured_corrector() is None
    monkeypatch.setenv('PUYO_LIVE_COLOR_BUFFER', '1')
    first, second = configured_corrector(), configured_corrector()
    assert first.__self__ is not second.__self__


def test_kept_outputs_and_views_remain_unchanged() -> None:
    rng = np.random.default_rng(42)
    buffer = ColorBuffer()
    retained = []
    for _ in range(FRAME_COUNT):
        frame = rng.integers(0, 256, SHAPE, dtype=np.uint8)
        expected = correct_601_as_709(frame)
        actual = buffer.correct(frame)
        assert actual.tobytes() == expected.tobytes()
        retained.append((actual[:, ::2], expected[:, ::2].copy()))
    for actual, expected in retained:
        assert actual.tobytes() == expected.tobytes()
    assert len(buffer.buffers) == BUFFER_COUNT


def test_reuses_released_buffers_without_changing_input() -> None:
    rng = np.random.default_rng(43)
    buffer = ColorBuffer()
    for _ in range(FRAME_COUNT):
        frame = rng.integers(0, 256, SHAPE, dtype=np.uint8)
        before = frame.tobytes()
        assert buffer.correct(frame).tobytes() == correct_601_as_709(frame).tobytes()
        assert frame.tobytes() == before
    assert buffer.reused == FRAME_COUNT - 1


def test_async_retention_is_not_overwritten() -> None:
    buffer = ColorBuffer()
    ready, release = Event(), Event()
    frame = np.full(SHAPE, 77, dtype=np.uint8)
    expected = correct_601_as_709(frame).tobytes()

    def read_later(image: np.ndarray) -> bytes:
        ready.set()
        assert release.wait(5)
        return image.tobytes()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(read_later, buffer.correct(frame))
        assert ready.wait(5)
        try:
            for value in range(FRAME_COUNT):
                buffer.correct(np.full(SHAPE, value, dtype=np.uint8))
        finally:
            release.set()
        assert future.result() == expected


def test_shape_and_dtype_change_preserves_exact_result() -> None:
    buffer = ColorBuffer()
    for dtype in (np.uint8, np.float32, np.uint8):
        for shape in (SHAPE, (12, 6, 3)):
            frame = np.full(shape, 121, dtype=dtype)
            assert buffer.correct(frame).tobytes() == correct_601_as_709(frame).tobytes()
