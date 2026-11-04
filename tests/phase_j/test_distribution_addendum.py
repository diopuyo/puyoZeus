"""追加指摘の失敗境界を、模擬配布先とローカル通信だけで確かめる。"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.phase_j import distribution_runtime as runtime_module
from src.phase_j import distribution_uninstall as uninstall
from src.phase_j.distribution_http import make_server
from src.phase_j.distribution_runtime import DistributionRuntime
from src.phase_j.distribution_summary import NumericSummary


def mark_product(root: Path) -> None:
    for name in uninstall.PRODUCT_MARKERS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', encoding='utf-8')


@pytest.mark.parametrize('missing', uninstall.PRODUCT_MARKERS)
def test_uninstall_requires_all_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        missing: str) -> None:
    mark_product(tmp_path)
    (tmp_path / missing).unlink()
    valuable = tmp_path / 'output/valuable.txt'
    valuable.parent.mkdir()
    valuable.write_text('keep', encoding='utf-8')
    monkeypatch.setattr(uninstall, 'ensure_stopped', lambda *a: pytest.fail('検査前に停止確認しない'))
    with pytest.raises(ValueError, match='Missing distribution markers'):
        uninstall.cleanup(tmp_path, True)
    assert valuable.read_text() == 'keep'


def test_uninstall_rejects_simulated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mark_product(tmp_path)
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    with pytest.raises(ValueError, match='Unsafe distribution root'):
        uninstall.validate_root(tmp_path)


def test_uninstall_rejects_drive_before_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 実ドライブへの操作はしない。resolve の結果を模擬するだけ。
    monkeypatch.setattr(Path, 'resolve', lambda self: Path(tmp_path.anchor))
    with pytest.raises(ValueError, match='Unsafe distribution root'):
        uninstall.validate_root(tmp_path / 'fake')


def test_outer_bind_retains_socket_and_advances(tmp_path: Path) -> None:
    first = make_server(DistributionRuntime(tmp_path / 'first'), 0)
    try:
        second_runtime = DistributionRuntime(tmp_path / 'second')
        second = make_server(second_runtime, first.server_port)
        try:
            assert second.server_port > first.server_port
            assert second_runtime.port == second.socket.getsockname()[1]
            with socket.socket() as competitor, pytest.raises(OSError):
                competitor.bind(('127.0.0.1', second.server_port))
        finally:
            second.server_close()
    finally:
        first.server_close()


@pytest.mark.parametrize('stage', ['scan', 'port', 'config', 'write', 'autostart'])
def test_start_failures_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    runtime = DistributionRuntime(tmp_path)
    runtime.settings['auto_input'] = stage == 'scan'
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError('injected startup boundary')
    if stage == 'scan':
        monkeypatch.setattr(runtime, 'scan_inputs', fail)
    elif stage == 'port':
        monkeypatch.setattr(runtime_module, 'available_port', fail)
    elif stage == 'config':
        monkeypatch.setattr(runtime_module, 'distribution_user_config', fail)
    elif stage == 'write':
        (tmp_path / 'puyo_live.json').mkdir()
    else:
        monkeypatch.setattr(runtime, 'start', fail)
    runtime.start_autostart() if stage == 'autostart' else runtime.start()
    assert runtime.error['type'] == 'startup_failure'
    assert runtime.child is None and runtime.job is None
    assert not runtime.input_waiting


def test_assign_failure_reaps_real_child_and_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DistributionRuntime(tmp_path)
    runtime.settings['auto_input'] = False
    real_popen = subprocess.Popen
    children = []
    def spawn(*args: Any, **kwargs: Any) -> subprocess.Popen:
        child = real_popen([sys.executable, '-s', '-c', 'import time; time.sleep(120)'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        children.append(child)
        return child
    def fail(*args: Any) -> None:
        raise OSError('assign_job injected')
    monkeypatch.setattr(runtime_module.subprocess, 'Popen', spawn)
    monkeypatch.setattr(runtime_module, 'assign_job', fail)
    closed = []
    original_close = runtime_module.close_job
    def close(job: Any) -> None:
        closed.append(job)
        original_close(job)
    monkeypatch.setattr(runtime_module, 'close_job', close)
    try:
        runtime.start()
        assert len(children) == 1 and children[0].poll() is not None
        assert children[0].stdout.closed
        assert runtime.child is None and runtime.job is None
        assert any(job is not None for job in closed)
        assert runtime.error['type'] == 'startup_failure'
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()


@pytest.mark.parametrize('no_proxy', ['', 'example.invalid', '127.0.0.1'])
def test_obs_uses_direct_socket_with_environment_proxy(monkeypatch: pytest.MonkeyPatch, no_proxy: str) -> None:
    from src.phase_j import distribution_obs as obs
    import websocket
    from websocket import _http
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy.audit.invalid:8888')
    monkeypatch.setenv('NO_PROXY', no_proxy)
    direct = SimpleNamespace(close=lambda: None)
    connected = []
    monkeypatch.setattr(obs.socket, 'create_connection',
                        lambda address, timeout: connected.append(address) or direct)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('プロキシの経路へ入らない')
    monkeypatch.setattr(_http, '_get_addrinfo_list', forbidden)
    responses = iter(['{"op":0,"d":{}}', '{"op":2}'])
    def connect(url: str, **kwargs: Any) -> Any:
        actual, _ = _http.connect(url, None, _http.proxy_info(**kwargs), kwargs['socket'])
        assert actual is direct
        return SimpleNamespace(recv=lambda: next(responses),
                               recv_data=lambda **kw: (websocket.ABNF.OPCODE_TEXT, next(responses)),
                               send=lambda value: None, close=lambda: None)
    monkeypatch.setattr(websocket, 'create_connection', connect)
    obs.ObsClient(4455, '')
    assert connected == [('127.0.0.1', 4455)]


def test_summary_separates_stream_and_restart(tmp_path: Path) -> None:
    runtime = DistributionRuntime(tmp_path)
    summary = runtime.numeric_summary
    def observe(stream: str) -> None:
        summary.observe(dict(identity=dict(stream_id=stream, match_id='1'), display=dict(match_phase='playing')))
    observe('first')
    observe('first')
    observe('second')
    assert summary.match_count == 2
    runtime.stop(final=False)
    observe('second')
    assert summary.match_count == 3


def test_asset_expansion_excludes_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'packaging'))
    import build_bundle
    monkeypatch.setattr(build_bundle, 'ROOT', tmp_path)
    for relative in ('data/current.json', 'data/_backup_pre_expand_yellow/old.json'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    assert build_bundle.expand_asset('data', tmp_path) == ['data/current.json']


@pytest.mark.parametrize('module', ['src.match_end_detector', 'src.prepared_template'])
def test_independent_imports(module: str) -> None:
    root = str(Path(__file__).resolve().parents[2])
    code = f'import sys; sys.path.insert(0, {root!r}); import {module}'
    result = subprocess.run([sys.executable, '-s', '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
