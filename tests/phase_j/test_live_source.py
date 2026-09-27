"""動画adapterの因果的なlatest-wins契約。"""
from __future__ import annotations

import numpy as np

from src.phase_j.live_source import VideoFileSource


class Capture:
    """frame番号を画素として返す入力代役。"""

    def __init__(self) -> None:
        self.index = 0

    def grab(self) -> bool:
        self.index += 1
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        frame = np.full((1080, 1920, 3), self.index, dtype=np.uint8)
        self.index += 1
        return True, frame


def test_offline_keeps_every_normalized_frame() -> None:
    source = VideoFileSource(Capture(), 60.0, 0, 8, 2)
    assert [frame.index for frame in source] == [0, 2, 4, 6]
    assert source.dropped == 0


def test_latest_wins_drops_only_past_frames() -> None:
    now = [0.0]
    source = VideoFileSource(Capture(), 60.0, 0, 60, 2, True,
                             clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt))
    stream = iter(source)
    assert next(stream).index == 0
    now[0] = 0.2
    frame = next(stream)
    assert frame.index == 12
    assert frame.captured_at <= now[0]
    assert frame.dropped_before == source.dropped == 5


def test_end_does_not_replay_stale_tail() -> None:
    now = [0.0]
    source = VideoFileSource(Capture(), 30.0, 0, 6, 1, True,
                             clock=lambda: now[0], sleep=lambda dt: None)
    stream = iter(source)
    next(stream)
    now[0] = 1.0
    assert list(stream) == []
    assert source.dropped == 5
