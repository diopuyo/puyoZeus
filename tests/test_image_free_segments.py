"""助走除外とキャッシュ命中中のフレーム別カウンターを検証する。"""
from pathlib import Path
from types import SimpleNamespace

from scripts.image_free_cache import CachedOld, CHUNK_FRAMES
from scripts.image_free_segments import RecordedRuntime, step, segment_identity
import pytest


class FakeRuntime:
    def __init__(self) -> None:
        self.state = dict(pipe=SimpleNamespace(_b3_queues=SimpleNamespace(index=0,
            shared=SimpleNamespace(game_idx=0)), _b3_capture=SimpleNamespace(game=0)))

    def step(self, frame: object, index: int, fps: float) -> dict:
        self.state['pipe']._b3_queues.index += 1
        return dict(frame=index)


def test_cached_counters_are_per_frame_not_frozen_prefix(tmp_path: Path) -> None:
    for _ in range(2):
        runtime = CachedOld(RecordedRuntime(FakeRuntime()), tmp_path, 'key')
        for index in range(CHUNK_FRAMES):
            observed = runtime.step(None, index, 60.)
            assert observed['counters']['lean_rows'] == index+1
    assert runtime.hits == 1 and runtime.computed == runtime.verify_frames


def test_warmup_does_not_own_frames() -> None:
    state = dict(runtimes=[RecordedRuntime(FakeRuntime()), RecordedRuntime(FakeRuntime())],
                 last_counters=[dict(lean_rows=0, games=0, adopted_games=0)]*2,
                 owned_counters=None, processed_frames=0)
    for index in range(5):
        row = step(state, None, index, 60., 3)
        assert (row is None) == (index < 3)
    assert state['processed_frames'] == 2
    assert state['owned_counters'][0]['lean_rows'] == 3
    assert state['last_counters'][0]['lean_rows'] == 5


def test_cached_prefix_survives_longer_end(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import image_free_segments
    monkeypatch.setattr(image_free_segments, 'identity', lambda *args: {'code_sha': 'same'})
    video = dict(path='v', start_frame=100, owned_start=200, end_frame=400)
    assert segment_identity(video, {}) == segment_identity(dict(video, end_frame=900), {})
    assert segment_identity(video, {}) != segment_identity(dict(video, start_frame=99), {})
