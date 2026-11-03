from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock
import pytest
from src.phase_j import live_power as power
from src.phase_j import live_24h_config as config

@pytest.fixture(autouse=True)
def reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(power, '_attempted_pid', None)
    monkeypatch.setenv(power.ENV_POWER, '0')

@pytest.mark.parametrize('platform', ['linux', 'darwin'])
def test_non_windows(monkeypatch: pytest.MonkeyPatch, capsys: Any, platform: str) -> None:
    monkeypatch.setattr(power.sys, 'platform', platform)
    call = Mock()
    monkeypatch.setattr(power, '_disable_execution_throttling', call)
    power.configure_power(True)
    power.apply_power_policy()
    call.assert_not_called()
    assert json.loads(capsys.readouterr().err)['status'] == 'unsupported'

def test_off_does_not_call(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    call = Mock()
    monkeypatch.setattr(power, '_disable_execution_throttling', call)
    power.configure_power(False)
    call.assert_not_called()
    assert capsys.readouterr().err == ''

@pytest.mark.parametrize('error', [None, OSError('failure'), AttributeError('missing API')])
def test_once_per_process(monkeypatch: pytest.MonkeyPatch, capsys: Any, error: Exception | None) -> None:
    monkeypatch.setattr(power.sys, 'platform', 'win32')
    call = Mock(side_effect=error)
    monkeypatch.setattr(power, '_disable_execution_throttling', call)
    power.configure_power(True)
    power.apply_power_policy()
    call.assert_called_once()
    record = json.loads(capsys.readouterr().err)
    assert record['status'] == ('failed' if error else 'applied')
    monkeypatch.setattr(power.os, 'getpid', lambda: record['pid'] + 1)
    power.apply_power_policy()
    assert call.call_count == 2

def test_cli_environment_and_explicit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(power.ENV_POWER, '1')
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_disable_power_throttling is True
    assert parser.parse_args(['--no-live-disable-power-throttling']).live_disable_power_throttling is False
    assert power.POWER_KEY in config.CONFIG_KEYS

@pytest.mark.parametrize('value', [1, 'true', None])
def test_invalid_flag(value: Any) -> None:
    with pytest.raises(ValueError):
        power.configure_power(value)

@pytest.mark.parametrize('enabled', [False, True])
def test_json_round_trip(tmp_path: Path, enabled: bool) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    user = parse_user_config(dict(device_name='OBS Virtual Camera',
                                  live_disable_power_throttling=enabled), tmp_path)
    assert build_pipeline_config(user).get(power.POWER_KEY, False) is enabled

def test_windows_api_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = Mock()
    kernel.GetCurrentProcess.return_value = 123
    kernel.SetProcessInformation.return_value = 1
    monkeypatch.setattr(power.ctypes, 'WinDLL', Mock(return_value=kernel), raising=False)
    power._disable_execution_throttling()
    args = kernel.SetProcessInformation.call_args.args
    assert args[0:2] == (123, 4)
    assert args[2]._obj.Version == 1
    assert args[2]._obj.ControlMask == 1
    assert args[2]._obj.StateMask == 0
    assert args[3] == 12


def _spawn_probe(queue: Any) -> None:
    power.apply_power_policy()
    queue.put((power.power_enabled(), power._attempted_pid == power.os.getpid()))


def test_spawn_inherits_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    import multiprocessing as mp
    monkeypatch.setenv(power.ENV_POWER, '1')
    context = mp.get_context('spawn')
    queue = context.Queue()
    child = context.Process(target=_spawn_probe, args=(queue,))
    child.start()
    try:
        assert queue.get(timeout=30) == (True, True)
        child.join(timeout=30)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join()
        queue.close()


def test_explicit_json_off_overrides_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    monkeypatch.setenv(power.ENV_POWER, '1')
    user = parse_user_config(dict(device_name='OBS Virtual Camera',
                                  live_disable_power_throttling=False), tmp_path)
    assert build_pipeline_config(user)[power.POWER_KEY] is False
