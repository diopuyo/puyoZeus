"""配布版・Python版のOBS優先監視と、監視失敗時の停止を検査する。"""
from types import SimpleNamespace
import subprocess

import pytest

from scripts import obs_priority_guard as module


def test_wait_until_launcher_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(['[123]', '[]'])
    sleeps: list[int] = []
    monkeypatch.setattr(module.subprocess, 'run',
                        lambda *a, **k: SimpleNamespace(stdout=next(replies)))
    monkeypatch.setattr(module.time, 'sleep', sleeps.append)
    module.ObsPriorityGuard().checkpoint(force=True)
    assert sleeps == [module.CHECK_SECONDS]


def test_query_failure_never_starts_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, 'powershell.exe')
    monkeypatch.setattr(module.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.ObsPriorityGuard().checkpoint(force=True)


def test_packaged_and_python_names_are_monitored() -> None:
    assert 'PuyoLive|PuyoZeus' in module.QUERY
    assert 'launcher|distribution_app' in module.QUERY
    assert module.CHECK_SECONDS <= 60
