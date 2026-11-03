"""単独通しの起点と書込防止、排他的計測を検証する。"""
from pathlib import Path
from types import SimpleNamespace
import os

import pytest

from scripts import image_free_stream as stream
from scripts.image_free_timing import Timings


def test_cold_midstream_is_forbidden(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='前史CP'):
        stream.load_runtime(dict(start_frame=10), {}, tmp_path)


def test_legacy_sink_is_disconnected() -> None:
    calls = []
    recorder = SimpleNamespace(sink=lambda row: calls.append(row))
    runtime = SimpleNamespace(state={'pipe': SimpleNamespace(_placement_reconcile=recorder)})
    stream.detach_logs(runtime)
    recorder.sink({'x': 1})
    assert calls == []


def test_guard_denies_legacy_mutation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hooks = []
    monkeypatch.setattr(stream.sys, 'addaudithook', hooks.append)
    stream.protect_outputs([str(tmp_path/'legacy')])
    hooks[0]('open', (str(tmp_path/'legacy/state'), 'r', os.O_RDONLY))
    hooks[0]('open', (str(tmp_path/'new/state'), 'w', os.O_WRONLY))
    for event, args in [('open', (str(tmp_path/'legacy/state'), 'a', os.O_APPEND)),
                        ('os.remove', (str(tmp_path/'legacy/state'),)),
                        ('os.rename', (str(tmp_path/'new/state'), str(tmp_path/'legacy/state')))]:
        with pytest.raises(PermissionError):
            hooks[0](event, args)


def test_nested_timing_is_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import image_free_timing
    ticks = iter([0., 1., 3., 5.])
    monkeypatch.setattr(image_free_timing.time, 'perf_counter', lambda: next(ticks))
    timing = Timings()
    with timing.measure('recognition'):
        with timing.measure('inference'):
            pass
    assert timing.seconds['recognition'] == 3.
    assert timing.seconds['inference'] == 2.
    assert sum(timing.seconds.values()) == 5.


def test_audit_contains_its_detector_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import image_free_timing
    ticks = iter([0., 1., 2., 4.])
    monkeypatch.setattr(image_free_timing.time, 'perf_counter', lambda: next(ticks))
    timing = Timings()
    with timing.measure('audit'):
        with timing.measure('recognition'):
            pass
    assert timing.seconds['audit'] == 4.
    assert timing.seconds['recognition'] == 0.
