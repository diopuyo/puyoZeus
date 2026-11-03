"""起動と停止のガードの設定・配線 (launcher → 環境変数 → 子 process) を検査する。

- allow_second_instance (既定 False = 二重起動を止める) は launcher の設定と環境変数だけで、pipeline の設定へは出さない。
- live_capture_reconnect / b3_worker_graceful_stop は既定 OFF の試作フラグ (SPEED_DEFAULTS → 環境変数)。
- OFF の pipeline_config.json は従来と同じ (鍵を足さない)。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

import pytest

from src.phase_j import launcher, live_capture_reconnect, live_instance_guard, live_speed
from src.phase_j.launcher import LauncherConfigError, build_pipeline_config, parse_user_config
from tests.launcher_legacy import legacy_user_config

DSHOW = dict(source='dshow', device_name='OBS Virtual Camera', device_index=3)
NEW_FLAGS = ('live_capture_reconnect', 'b3_worker_graceful_stop')


def args(path: Path, *extra: str) -> argparse.Namespace:
    return launcher.parse_launcher_args(['--config', str(path), '--skip-manifest', *extra])


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def write_config(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / 'puyo_live.json'
    path.write_text(json.dumps(legacy_user_config(dict(DSHOW, **overrides))), encoding='utf-8')
    return path


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """prepare が書く環境変数 (較正先・CNN・許可設定) を試験の外へ漏らさない。実際の鍵は取らない。"""
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    monkeypatch.setattr(live_instance_guard, '_HELD', [])
    monkeypatch.setattr(launcher, 'missing_runtime_dlls', lambda python_dir: [])
    monkeypatch.setattr(launcher, 'check_telop_templates', lambda root=None: 1)


# ---------------------------------------------------------------- 設定

def test_defaults_keep_current_behavior(tmp_path: Path) -> None:
    user = parse_user_config(DSHOW, tmp_path)
    assert user.allow_second_instance is False
    assert user.live_capture_reconnect is False and user.b3_worker_graceful_stop is False
    config = build_pipeline_config(user)
    assert 'allow_second_instance' not in config                # launcher だけの設定。pipeline へは出さない
    assert not any(key in config for key in NEW_FLAGS)          # OFF の設定ファイルは従来と同じ (鍵を足さない)


@pytest.mark.parametrize('value', ['false', 'true', 0, 1, None])
def test_allow_second_instance_requires_bool(value: object, tmp_path: Path) -> None:
    with pytest.raises(LauncherConfigError, match='allow_second_instance'):
        parse_user_config(dict(DSHOW, allow_second_instance=value), tmp_path)


def test_allow_second_instance_true_is_accepted(tmp_path: Path) -> None:
    assert parse_user_config(dict(DSHOW, allow_second_instance=True), tmp_path).allow_second_instance is True


@pytest.mark.parametrize('key', NEW_FLAGS)
def test_new_flags_are_forwarded_only_when_on_and_must_be_bool(key: str, tmp_path: Path) -> None:
    on = build_pipeline_config(parse_user_config(dict(DSHOW, **{key: True}), tmp_path))
    assert on[key] is True
    with pytest.raises(LauncherConfigError, match=key):
        parse_user_config(dict(DSHOW, **{key: 'yes'}), tmp_path)


@pytest.mark.parametrize('key', NEW_FLAGS)
def test_new_flags_are_default_off_speed_flags(key: str) -> None:
    assert live_speed.SPEED_DEFAULTS[key] is False
    assert key in launcher.USER_KEYS


def test_environment_delivery_to_child_processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_speed.os, 'environ', {})
    options = argparse.Namespace(source='dshow', cpu_threads=1, output=tmp_path,
                                 **{key: True for key in NEW_FLAGS})
    live_speed.configure(options, argparse.ArgumentParser())
    env = live_speed.os.environ
    assert env['PUYO_LIVE_CAPTURE_RECONNECT'] == '1' and env['PUYO_B3_WORKER_GRACEFUL_STOP'] == '1'
    assert Path(env[live_capture_reconnect.OUTPUT_ENV]) == tmp_path.resolve()      # 再接続の記録先 (output)
    off = argparse.Namespace(source='dshow', cpu_threads=1, output=tmp_path)
    live_speed.configure(off, argparse.ArgumentParser())
    assert env['PUYO_LIVE_CAPTURE_RECONNECT'] == '0' and env['PUYO_B3_WORKER_GRACEFUL_STOP'] == '0'
    assert live_capture_reconnect.OUTPUT_ENV not in env           # OFF では前回の記録先を残さない


# ---------------------------------------------------------------- prepare の入口

def spy_guard(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr(live_instance_guard, 'guard_instances',
                        lambda keys, host, port, allow, **kw: calls.append((keys, host, port, allow)))
    return calls


def test_prepare_runs_the_guard_first_with_port_camera_and_allow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = spy_guard(monkeypatch)
    order: list[str] = []
    monkeypatch.setattr(launcher, 'check_bundle', lambda *a: order.append('bundle'))
    monkeypatch.setattr(live_instance_guard, 'guard_instances',
                        lambda keys, host, port, allow, **kw: (order.append('guard'), calls.append((keys, host, port, allow))))
    launcher.prepare(launcher.parse_launcher_args(['--config', str(write_config(tmp_path))]))   # 配布物の照合も行う設定
    keys, host, port, allow = calls[0]
    assert [key.kind for key in keys] == ['port', 'camera']
    assert (host, port, allow) == ('127.0.0.1', launcher.DEFAULT_PORT, False)
    assert order == ['guard', 'bundle']                            # 設定の書き出し・配布物の照合より前
    assert os.environ[live_instance_guard.ALLOW_SECOND_INSTANCE_ENV] == '0'


def test_prepare_passes_allow_and_exports_it_for_the_bind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = spy_guard(monkeypatch)
    launcher.prepare(args(write_config(tmp_path, allow_second_instance=True)))
    assert calls[0][3] is True
    assert os.environ[live_instance_guard.ALLOW_SECOND_INSTANCE_ENV] == '1'
    assert live_instance_guard.exclusive_bind_default() is False     # 許可時は HTTP の bind も従来のまま


def test_check_only_does_not_take_any_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = spy_guard(monkeypatch)
    launcher.prepare(args(write_config(tmp_path), '--check-only'))
    assert calls == []                                              # 動いている最中に設定だけ確認できる


def test_prepare_stops_with_exit_code_when_another_instance_holds_port_and_camera(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """実際の鍵 (試験専用の名前空間) で、2 つ目の launcher 相当が案内つきの終了コードで止まる。"""
    import uuid
    token = uuid.uuid4().hex[:10]
    monkeypatch.setattr(live_instance_guard, 'MUTEX_NAME_PREFIX', f'Local\\PuyoLiveTest-{token}-')
    monkeypatch.setattr(live_instance_guard, 'LOCK_DIR_NAME', f'puyo_live_locks_test_{token}')
    monkeypatch.setattr(live_instance_guard, 'ACQUIRE_GRACE_SEC', 0.0)
    port = free_port()
    path = write_config(tmp_path, port=port)
    keys = live_instance_guard.instance_keys('dshow', '127.0.0.1', port, 3)
    first = live_instance_guard.acquire_instances(keys, grace_sec=0)     # 先に動いている PuyoLive
    try:
        with pytest.raises(SystemExit) as stop:
            launcher.prepare(args(path))
        assert stop.value.code == live_instance_guard.EXIT_ALREADY_RUNNING == 5
        assert not (tmp_path / 'output').exists()                   # 出力先・設定の書き出しまで進まない
        assert 'allow_second_instance' in capsys.readouterr().err
        launcher.prepare(args(write_config(tmp_path, port=port, allow_second_instance=True)))   # 許可すれば警告だけで進む
        assert (tmp_path / 'output' / launcher.PIPELINE_CONFIG_NAME).is_file()
    finally:
        for lock in first.held:
            live_instance_guard.release_lock(lock)
        for lock in live_instance_guard._HELD:
            live_instance_guard.release_lock(lock)


# ---------------------------------------------------------------- 評価 process の bind (認識を始める前)

def test_bind_or_exit_stops_with_exit_code_and_message(capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.run_live_pipeline_20260928 import bind_or_exit

    class Busy:
        def bind(self) -> None:
            raise OSError(10048, '通常、各ソケット アドレスに対して使用できるのは 1 つのみです')

    with pytest.raises(SystemExit) as stop:
        bind_or_exit(Busy(), '127.0.0.1', 8765)
    assert stop.value.code == live_instance_guard.EXIT_ALREADY_RUNNING
    assert '127.0.0.1:8765' in capsys.readouterr().err
    bind_or_exit(type('Free', (), {'bind': lambda self: None})(), '127.0.0.1', 8765)   # 成功なら何も起きない


def test_run_live_binds_the_port_before_any_worker_is_started() -> None:
    """失敗を B3・認識 process の起動より前にする (起動後に止めると非 daemon の子が親の終了を待ち続ける)。"""
    import ast
    source = (Path(__file__).resolve().parents[1] / 'scripts' / 'run_live_pipeline_20260928.py').read_text(encoding='utf-8')
    run_live = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'run_live')
    lines: dict[str, int] = {}
    for node in ast.walk(run_live):
        if isinstance(node, ast.Call):
            name = getattr(node.func, 'id', getattr(node.func, 'attr', None))
            if name in ('bind_or_exit', 'make_bridge', 'LiveB3', 'start'):
                lines.setdefault(name, node.lineno)
    assert lines['bind_or_exit'] < lines['make_bridge'] < lines['LiveB3'] < lines['start']
