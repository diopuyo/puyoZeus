"""資源不足では検収の子だけを停止し、失敗を成功へ変換しない。"""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


def test_guard_stops_only_owned_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'packaging'))
    import verify_live_obs_20261020b as verify
    import smoke_live_obs_20261020 as smoke
    checks, stopped, waits = [], [], []

    def guard(minimum: float = 6.5) -> None:
        checks.append(minimum)
        if len(checks) > 1:
            raise RuntimeError('空きメモリ不足')

    def wait(timeout: float) -> int:
        waits.append(timeout)
        if timeout == verify.MEMORY_POLL_SEC:
            raise subprocess.TimeoutExpired('owned-child', timeout)
        return 0

    child = SimpleNamespace(pid=12345, poll=lambda: None, wait=wait)
    monkeypatch.setattr(smoke, 'memory_guard', guard)
    monkeypatch.setattr(verify.subprocess, 'Popen', lambda *a, **k: child)
    monkeypatch.setattr(verify.subprocess, 'run', lambda args, **k: stopped.append(args))
    with pytest.raises(RuntimeError, match='空きメモリ不足'):
        verify.guarded_run(['owned-child'], {}, None)
    assert checks == [8.5, 6.5]
    assert stopped == [['taskkill', '/PID', '12345', '/T', '/F']]
    assert waits == [verify.MEMORY_POLL_SEC, 30]
