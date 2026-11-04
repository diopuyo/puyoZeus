"""二回目のE2E不具合を模擬root・HTTP・画像だけで確かめる。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from src.phase_j import distribution_uninstall as uninstall
from src.phase_j import distribution_ports as ports


def product(root: Path) -> Path:
    for name in uninstall.PRODUCT_MARKERS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{}', encoding='utf-8')
    return root


@pytest.mark.parametrize('answer,delete', [('y', True), ('N', False), ('', False)])
def test_interactive_folder_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   capsys: Any, answer: str, delete: bool) -> None:
    root = product(tmp_path / 'root')
    record = root / 'record.txt'
    record.write_text('keep', encoding='utf-8')
    calls: list[Path] = []
    monkeypatch.setattr(uninstall, 'schedule_folder_removal', lambda path: calls.append(path))
    monkeypatch.setattr(uninstall, 'shortcuts', lambda *a, **k: None)
    monkeypatch.setattr('sys.argv', ['uninstall', '--root', str(root), '--interactive'])
    prompts: list[str] = []
    def respond(prompt: str) -> str:
        prompts.append(prompt)
        return answer
    monkeypatch.setattr('builtins.input', respond)
    uninstall.main()
    assert calls == ([root] if delete else [])
    assert record.read_text(encoding='utf-8') == 'keep'
    assert len(prompts) == 1 and str(root) in prompts[0] and '[y/N]' in prompts[0]
    assert ('この画面を閉じると' if delete else '設定と記録はフォルダに残しました') in capsys.readouterr().out


@pytest.mark.parametrize('answer', ['Y', ''])
def test_stop_only_verified_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    root = product(tmp_path / 'root')
    (root / 'distribution.json').write_text('{"port": 8888}', encoding='utf-8')
    stopped = False
    requests: list[Any] = []
    def check(path: Path, wait_seconds: float = 0) -> None:
        assert path == root
        if not stopped:
            raise OSError('running')
    def request(req: Any, **kwargs: Any) -> Any:
        nonlocal stopped
        requests.append(req)
        stopped = True
        return __import__('io').BytesIO(b'{}')
    monkeypatch.setattr(uninstall, 'ensure_stopped', check)
    monkeypatch.setattr('builtins.input', lambda *a: answer)
    monkeypatch.setattr(ports, 'inspect_conflict', lambda path, port: dict(kind='same') if port == 8888 else {})
    monkeypatch.setattr(uninstall, 'urlopen', request)
    assert uninstall.stop_interactively(root)
    assert len(requests) == 1
    assert requests[0].full_url == 'http://127.0.0.1:8888/api/stop'
    assert requests[0].get_header('Host') == '127.0.0.1:8888'
    assert requests[0].get_header('Origin') is None


@pytest.mark.parametrize('kind', ['different', 'other'])
def test_stop_refuses_other_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    def running(*args: Any, **kwargs: Any) -> None:
        raise OSError('running')
    monkeypatch.setattr(uninstall, 'ensure_stopped', running)
    monkeypatch.setattr('builtins.input', lambda *a: 'Y')
    monkeypatch.setattr(ports, 'inspect_conflict', lambda *a: dict(kind=kind))
    monkeypatch.setattr(uninstall, 'urlopen', lambda *a, **k: pytest.fail('他のrootに送らない'))
    with pytest.raises(ValueError, match='Exit this app'):
        uninstall.stop_interactively(product(tmp_path))


@pytest.mark.parametrize('marked', [True, False])
def test_external_remover_keeps_outside(tmp_path: Path, marked: bool) -> None:
    root = tmp_path / "root & [test] ' 日本語"
    root.mkdir()
    if marked:
        product(root)
    outside = tmp_path / 'outside.txt'
    outside.write_text('keep', encoding='utf-8')
    from src.phase_j.distribution_remove import REMOVAL_SCRIPT
    result = subprocess.run([uninstall.system_tool('powershell'), '-NoProfile', '-NonInteractive',
        '-Command', REMOVAL_SCRIPT], cwd=tmp_path,
        env={**os.environ, 'PUYO_REMOVE_ROOT': str(root), 'PUYO_REMOVE_PID': '0', 'PUYO_REMOVE_PARENT': '0'},
        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=30)
    assert root.exists() is not marked, result.stderr
    assert (result.returncode == 0) is marked
    assert outside.read_text(encoding='utf-8') == 'keep'


def test_remover_refuses_junction(tmp_path: Path) -> None:
    root = product(tmp_path / 'root')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep.txt').write_text('keep', encoding='utf-8')
    link = root / 'linked'
    command = str(Path(os.environ['SystemRoot']) / 'System32/cmd.exe')
    subprocess.run([command, '/c', 'mklink', '/J', str(link), str(outside)],
                   capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    with pytest.raises(ValueError, match='Linked'):
        uninstall.validate_tree(root)
    from src.phase_j.distribution_remove import REMOVAL_SCRIPT
    result = subprocess.run([uninstall.system_tool('powershell'), '-NoProfile', '-NonInteractive',
        '-Command', REMOVAL_SCRIPT], cwd=tmp_path,
        env={**os.environ, 'PUYO_REMOVE_ROOT': str(root), 'PUYO_REMOVE_PID': '0', 'PUYO_REMOVE_PARENT': '0'},
        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=30)
    assert result.returncode != 0 and root.exists()
    assert (outside / 'keep.txt').read_text(encoding='utf-8') == 'keep'


def test_schedule_outside_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = product(tmp_path / 'root')
    calls: list[tuple] = []
    monkeypatch.setattr(uninstall.subprocess, 'Popen', lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(uninstall.tempfile, 'tempdir', str(tmp_path))
    uninstall.schedule_folder_removal(root)
    args, kwargs = calls[0]
    assert '-Command' in args[0] and '-File' not in args[0]
    assert '-ExecutionPolicy' not in args[0]
    assert Path(kwargs['cwd']) == tmp_path
    assert kwargs['creationflags'] == subprocess.CREATE_NO_WINDOW
    assert kwargs['env']['PUYO_REMOVE_PID'] == str(os.getpid())
    assert kwargs['env']['PUYO_REMOVE_PARENT'] == str(os.getppid())


def test_remover_waits_for_both_locked_files(tmp_path: Path) -> None:
    """入口とDLLのロックが両方外れるまでフォルダを残す。アプリは起動しない。"""
    import sys
    from src.phase_j.distribution_remove import REMOVAL_SCRIPT
    root = product(tmp_path / 'root')
    children = []
    remover = None
    try:
        for name in ('puyoZeus Live.exe', 'python312.dll'):
            path = root / name
            path.write_bytes(b'test lock only')
            command = [sys.executable, '-c',
                       "import sys; f=open(sys.argv[1],'rb'); print('ready',flush=True); input()", str(path)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            children.append(process)
            assert process.stdout.readline().strip() == 'ready'
        remover = subprocess.Popen([uninstall.system_tool('powershell'), '-NoProfile', '-NonInteractive',
            '-Command', REMOVAL_SCRIPT], cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, 'PUYO_REMOVE_ROOT': str(root), 'PUYO_REMOVE_PID': str(children[0].pid),
                 'PUYO_REMOVE_PARENT': str(children[1].pid)}, creationflags=subprocess.CREATE_NO_WINDOW)
        children[0].communicate('\n', timeout=10)
        assert root.exists() and children[1].poll() is None
        children[1].communicate('\n', timeout=10)
        _, error = remover.communicate(timeout=30)
        assert remover.returncode == 0 and not root.exists(), error
    finally:
        for process in [*children, remover]:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()


def test_running_input_stop_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from src.phase_j.distribution_input import stopped_reference
    from src.phase_j.live_device import DirectShowSource
    statuses: list[str] = []
    source = object.__new__(DirectShowSource)
    source.config = SimpleNamespace(verify_fail_streak=2)
    source._split_diagnostics = False
    source.on_status = statuses.append
    source.sleep = lambda *a: None
    continuity = SimpleNamespace(diagnostics={})
    source._screen_hold(True, stopped_reference(), continuity, 2, SimpleNamespace())
    source._screen_hold(True, np.zeros((54, 96, 3), dtype=np.uint8), continuity, 2, SimpleNamespace())
    assert statuses == ['obs_stopped', 'no_puyo_screen']


def test_private_memory_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_memory as memory
    calls: list[bool] = []
    def rows(include_obs: bool = True) -> list[dict]:
        calls.append(include_obs)
        return [dict(pid=1, parent=0, name='python.exe', private_bytes=123),
                dict(pid=2, parent=1, name='python.exe')]
    monkeypatch.setattr(memory, 'process_tree', rows)
    path = tmp_path / 'process_memory.jsonl'
    memory.record_memory(path)
    result = json.loads(path.read_text(encoding='utf-8'))
    assert calls == [False] and result['missing'] == 1
    assert [r['private_bytes'] for r in result['processes']] == [123, None]
    assert memory.MEMORY_INTERVAL_SECONDS == 300


def test_obs_connection_failure_has_port(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_obs as obs
    def refused(*args: Any) -> None:
        raise ConnectionRefusedError('refused')
    monkeypatch.setattr(obs, 'ObsClient', refused)
    with pytest.raises(ValueError, match=r'OBS につながりません.*\(4455\).*Cannot connect to OBS'):
        obs.integrate(dict(obs_enabled=True, obs_port=4455, obs_password=''), 'http://127.0.0.1/')
