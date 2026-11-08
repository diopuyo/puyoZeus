"""実機や利用者のプロセスを使わず、停止と正常な保留を区別する。"""
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src import production_config
from src.phase_j import child_watchdog as watchdog
from src.phase_j import distribution_runtime as runtime_module
from src.phase_j.distribution_runtime import DistributionRuntime


def test_default_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert production_config.DISTRIBUTION_CHILD_WATCHDOG is False
    monkeypatch.setenv(watchdog.HEARTBEAT_ENV, str(tmp_path / 'old.txt'))
    assert watchdog.start_input_heartbeat() is None
    runtime = DistributionRuntime(tmp_path)
    path, environment = runtime._heartbeat_environment()
    assert path is None and watchdog.HEARTBEAT_ENV not in environment


def test_generation_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(production_config, 'DISTRIBUTION_CHILD_WATCHDOG', True)
    runtime = DistributionRuntime(tmp_path)
    first, environment = runtime._heartbeat_environment()
    second, _ = runtime._heartbeat_environment()
    assert first != second and first.parent == runtime.logs
    monkeypatch.setenv(watchdog.HEARTBEAT_ENV, environment[watchdog.HEARTBEAT_ENV])
    heartbeat = watchdog.start_input_heartbeat()
    try:
        heartbeat.beat()
        assert float(first.read_text()) > 0
    finally:
        heartbeat.close()


@pytest.mark.parametrize('running', [False, True])
def test_deadline(tmp_path: Path, running: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watchdog, 'STARTUP_STALL_SEC', 3)
    monkeypatch.setattr(watchdog, 'RUNNING_STALL_SEC', 2)
    path = tmp_path / 'heartbeat'
    deadline = watchdog.ProgressDeadline(0)
    if running:
        path.write_text('1')
    assert deadline.check(path, 0) is None
    limit = 2 if running else 3
    assert deadline.check(path, limit - .01) is None
    assert deadline.check(path, limit) == ('running_stall' if running else 'startup_stall', limit)


@pytest.mark.parametrize('value', ['nan', 'inf', '', 'broken'])
def test_bad_heartbeat_does_not_reset(tmp_path: Path, value: str) -> None:
    path = tmp_path / 'heartbeat'
    path.write_text(value)
    assert watchdog.ProgressDeadline(0).check(path, watchdog.STARTUP_STALL_SEC)[0] == 'startup_stall'


@pytest.mark.parametrize('kind', ['startup_stall', 'running_stall'])
def test_recovery_uses_restart_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    runtime = DistributionRuntime(tmp_path)
    launches, kills = [], []
    runtime.settings['auto_input'] = False
    monkeypatch.setattr(runtime, 'start', lambda: launches.append(True))
    monkeypatch.setattr(runtime.cancel_start, 'wait', lambda _: False)
    original_run = runtime_module.subprocess.run

    def run(command: list[str], **kwargs: object) -> object:
        if '/T' in command and '/F' in command:
            kills.append(command)
            return Mock(returncode=0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(runtime_module.subprocess, 'run', run)
    for _ in range(runtime_module.RESTART_LIMIT + 1):
        child = Mock(pid=123, stdout=BytesIO(b''), poll=Mock(return_value=0), wait=Mock(return_value=0))
        runtime.child = child
        runtime._terminate_stalled_child(child, kind, 150)
        runtime._watch_child(child)
    assert len(launches) == runtime_module.RESTART_LIMIT
    assert len(runtime.restart_times) == runtime_module.RESTART_LIMIT
    assert runtime.error and not runtime.restarting
    records = [json.loads(line) for line in (runtime.logs / 'child_watchdog.jsonl').read_text().splitlines()]
    assert len(records) == runtime_module.RESTART_LIMIT + 1
    assert all(item['event'] == kind and item['stalled_seconds'] == 150 for item in records)
    if runtime_module.os.name == 'nt':
        assert all(command[-2:] == ['/T', '/F'] for command in kills)


@pytest.mark.parametrize('running', [False, True])
def test_monitor_detects_stall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, running: bool) -> None:
    runtime = DistributionRuntime(tmp_path)
    child = Mock(poll=Mock(return_value=None))
    runtime.child = child
    path = tmp_path / 'heartbeat'
    if running:
        path.write_text('1')
    ticks = iter([0, 1, 2, 3])
    monkeypatch.setattr(watchdog, 'STARTUP_STALL_SEC', 3)
    monkeypatch.setattr(watchdog, 'RUNNING_STALL_SEC', 2)
    monkeypatch.setattr(runtime_module.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(runtime.finished, 'wait', lambda _: False)
    terminate = Mock()
    monkeypatch.setattr(runtime, '_terminate_stalled_child', terminate)
    runtime._watch_progress(child, path, 0)
    terminate.assert_called_once_with(child, 'running_stall' if running else 'startup_stall', 2 if running else 3)


@pytest.mark.parametrize('reason', ['replaced', 'stop', 'cancel', 'exited'])
def test_monitor_leaves_obsolete_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    runtime = DistributionRuntime(tmp_path)
    child = Mock(poll=Mock(return_value=0 if reason == 'exited' else None))
    runtime.child = Mock() if reason == 'replaced' else child
    runtime.stopping = reason == 'stop'
    if reason == 'cancel':
        runtime.cancel_start.set()
    monkeypatch.setattr(runtime.finished, 'wait', lambda _: False)
    terminate = Mock()
    monkeypatch.setattr(runtime, '_terminate_stalled_child', terminate)
    runtime._watch_progress(child, tmp_path / 'missing', 0)
    terminate.assert_not_called()


def test_dump_once_per_stall_and_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watchdog, '_dump_count', 0)
    private_value = 'PRIVATE_IMAGE_AND_USER_VALUE_1234'
    clock = [0.0]
    heartbeat = watchdog.InputHeartbeat(tmp_path / 'heartbeat', lambda: clock[0])
    for _ in range(watchdog.MAX_STALL_DUMPS + 1):
        heartbeat.beat()
        clock[0] += watchdog.STALL_DUMP_SEC
        heartbeat.check()
        heartbeat.check()
    files = list(tmp_path.glob('stall_stacks_*.txt'))
    assert len(files) == watchdog.MAX_STALL_DUMPS
    another = watchdog.InputHeartbeat(tmp_path / 'other', lambda: clock[0])
    clock[0] += watchdog.STALL_DUMP_SEC
    another.check()
    assert len(list(tmp_path.glob('stall_stacks_*.txt'))) == watchdog.MAX_STALL_DUMPS
    for path in files:
        content = path.read_text('utf-8')
        assert 'test_dump_once_per_stall_and_cap' in content
        assert 'Current thread' in content
        assert 'heartbeat =' not in content and 'numpy' not in content
        assert private_value not in content


@pytest.mark.parametrize('hold', ['camera_stopped', 'input_waiting', 'feedback', 'match_waiting'])
def test_hold_loop_keeps_heartbeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hold: str) -> None:
    from src.phase_j import live_device, distribution_input
    from src.phase_j.live_device import DeviceConfig, DirectShowSource, FrameContinuity
    import numpy as np
    clock = [0.0]
    heartbeat = watchdog.InputHeartbeat(tmp_path / 'heartbeat', lambda: clock[0])
    deadline = watchdog.ProgressDeadline(0)
    capture, checks, statuses = Mock(), [], []

    def sleep(seconds: float) -> None:
        checks.append(deadline.check(heartbeat.path, clock[0]))
        clock[0] += watchdog.POLL_SEC

    source = DirectShowSource(DeviceConfig('OBS', 0), watchdog.RUNNING_STALL_SEC * 3,
        lambda: None, verifier=lambda _: hold == 'match_waiting', clock=lambda: clock[0],
        sleep=sleep, on_status=statuses.append)
    source._feedback_present = hold == 'feedback'
    image = (None if hold == 'input_waiting' else distribution_input.stopped_reference()
             if hold == 'camera_stopped' else np.zeros((54, 96, 3), np.uint8))
    monkeypatch.setattr(source, '_capture', lambda: capture)
    monkeypatch.setattr(source, '_configure', lambda _: None)
    monkeypatch.setattr(source, '_screen_state', lambda _: None)
    monkeypatch.setattr(source, '_maintain', lambda cap, *_: cap)
    monkeypatch.setattr(source, '_read', lambda _: (clock[0], image, (1920, 1080)))
    monkeypatch.setattr(source, '_feedback_result', lambda _: True)
    monkeypatch.setattr(FrameContinuity, 'ready', lambda *args: True)
    monkeypatch.setattr(live_device, 'verify_screen', lambda *args: hold == 'match_waiting')
    monkeypatch.setattr(watchdog, 'start_input_heartbeat', lambda: heartbeat)
    frames = list(source)
    assert bool(frames) == (hold == 'match_waiting')
    if hold != 'match_waiting':
        expected = dict(camera_stopped='obs_stopped', input_waiting='verifying', feedback='overlay_feedback')
        assert set(statuses) == {expected[hold]}
    assert len(checks) > 1 and all(value is None for value in checks), hold
    assert clock[0] >= watchdog.RUNNING_STALL_SEC * 3
    assert heartbeat.closed.is_set()
