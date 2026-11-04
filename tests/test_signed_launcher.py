"""取得・インストールと窓は模擬し、入口の分岐だけを検証する。"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packaging/launcher'))
import signed_launcher as launcher
import legacy_cleanup as legacy


@pytest.mark.parametrize('case', ['present', 'decline', 'vc', 'web', 'download_error',
    'invalid_signature', 'cancel', 'install_error', 'reboot', 'absent_after'])
def test_runtime_flow(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.tempfile, 'tempdir', str(tmp_path))
    monkeypatch.setattr(launcher, 'japanese', lambda: True)
    prompt = Mock(return_value=case != 'decline')
    get = Mock(side_effect=OSError('download') if case == 'download_error'
               else lambda url, target: target.write_bytes(b'mocked installer'))
    verify = Mock(return_value=case != 'invalid_signature')
    install = Mock(side_effect=OSError('UAC cancelled') if case == 'cancel' else None,
                   return_value=3010 if case == 'reboot' else 1 if case == 'install_error' else 0)
    monkeypatch.setattr(launcher, 'message', prompt)
    monkeypatch.setattr(launcher, 'download', get)
    monkeypatch.setattr(launcher, 'verify_microsoft', verify)
    monkeypatch.setattr(launcher, 'install', install)
    present = Mock(side_effect=[case == 'present', case != 'absent_after'])
    url = launcher.WEB_URL if case == 'web' else launcher.VC_URL
    launcher.ensure_runtime('test', url, '/install', present)
    assert get.call_count == int(case not in ('present', 'decline'))
    assert install.call_count == int(case not in ('present', 'decline', 'download_error', 'invalid_signature'))
    assert prompt.call_count == (0 if case == 'present' else 2 if case in (
        'download_error', 'invalid_signature', 'cancel', 'install_error', 'reboot', 'absent_after') else 1)
    if case == 'web':
        assert 'ブラウザ' in prompt.call_args_list[0].args[0]


@pytest.mark.parametrize('ja', [True, False])
def test_language(ja: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, 'japanese', lambda: ja)
    assert launcher.text('日本語', 'English') == ('日本語' if ja else 'English')


@pytest.mark.parametrize('names', [[], ['msvcp140.dll'], ['vcruntime140_1.dll'],
                                 ['msvcp140.dll', 'vcruntime140_1.dll']])
def test_vc_files(names: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SystemRoot', str(tmp_path))
    (tmp_path / 'System32').mkdir()
    for name in names:
        (tmp_path / 'System32' / name).touch()
    assert launcher.has_vc() == (len(names) == 2)


@pytest.mark.parametrize('version', ['0.0.0.0', 'broken', '123.0.0.1'])
def test_webview_detection(version: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import winreg
    from unittest.mock import MagicMock
    monkeypatch.setattr(winreg, 'OpenKey', MagicMock())
    monkeypatch.setattr(winreg, 'QueryValueEx', lambda *a: (version, 1))
    assert launcher.has_webview() == (version == '123.0.0.1')


@pytest.mark.parametrize('dismiss', [True, False])
def test_onedrive(dismiss: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / 'OneDrive' / 'app'
    root.mkdir(parents=True)
    monkeypatch.setattr(launcher, 'japanese', lambda: False)
    ask = Mock(return_value=dismiss)
    monkeypatch.setattr(launcher, 'message', ask)
    launcher.warn_onedrive(root)
    launcher.warn_onedrive(root)
    assert ask.call_count == (1 if dismiss else 2)
    assert not launcher.is_onedrive(tmp_path / 'OneDrive2')


@pytest.mark.parametrize('uninstall,code,timeout', [(False, 0, False), (False, 1, False),
                                                   (False, 0, True), (True, 1, False)])
def test_child(uninstall: bool, code: int, timeout: bool, tmp_path: Path,
               monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    process.wait = Mock(return_value=code, side_effect=subprocess.TimeoutExpired('app', 8) if timeout else None)
    popen = Mock(return_value=process)
    fail = Mock()
    monkeypatch.setattr(launcher.subprocess, 'Popen', popen)
    monkeypatch.setattr(launcher, 'failure', fail)
    assert launcher.run_python(tmp_path, ['--autostart'], uninstall) == (0 if timeout else code)
    command = popen.call_args.args[0]
    assert ('--interactive' if uninstall else '--autostart') in command
    assert process.wait.call_args.kwargs['timeout'] == (None if uninstall else 8)
    assert fail.call_count == int(not uninstall and code != 0)


def test_uninstall_exception_enter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, 'japanese', lambda: True)
    monkeypatch.setattr(launcher, 'run_python', Mock(side_effect=OSError('failed')))
    fail, enter = Mock(), Mock()
    monkeypatch.setattr(launcher, 'failure', fail)
    monkeypatch.setattr('builtins.input', enter)
    assert launcher.main(tmp_path, [], True) == 1
    assert not fail.called and enter.call_count == 1


def test_https_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Unexpected'):
        launcher.download('http://example.test', tmp_path / 'unused')
    with pytest.raises(ValueError, match='HTTPS'):
        launcher.HttpsRedirect().redirect_request(None, None, 302, '', {}, 'http://example.test')


def test_cleanup_content_and_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hashes = {'old.bat': [hashlib.sha256(b'old').hexdigest()], 'changed.bat': [hashlib.sha256(b'old').hexdigest()]}
    monkeypatch.setattr(legacy.json, 'loads', lambda value: hashes)
    for name, content in [('old.bat', b'old'), ('changed.bat', b'edited'), ('unknown.bat', b'old')]:
        (tmp_path / name).write_bytes(content)
    legacy.cleanup(tmp_path)
    assert not (tmp_path / 'old.bat').exists()
    assert (tmp_path / 'changed.bat').read_bytes() == b'edited'
    assert (tmp_path / 'unknown.bat').exists()
    assert 'old.bat' in (tmp_path / 'logs/launcher-cleanup.jsonl').read_text('utf-8')


def test_cleanup_junction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outside, root = tmp_path / 'outside', tmp_path / 'root'
    outside.mkdir()
    root.mkdir()
    (outside / 'old.bat').write_bytes(b'old')
    monkeypatch.setattr(legacy.json, 'loads', lambda value: {'tools/old.bat': [hashlib.sha256(b'old').hexdigest()]})
    subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(root / 'tools'), str(outside)],
                   check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    legacy.cleanup(root)
    assert (outside / 'old.bat').read_bytes() == b'old'


@pytest.mark.parametrize('code', [0, 1])
def test_signature_verifier(code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock(return_value=SimpleNamespace(returncode=code))
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    assert launcher.verify_microsoft(tmp_path / "a' & file.exe") == (code == 0)
    args, kwargs = run.call_args
    script = args[0][-1]
    assert "'Valid'" in script and "'Microsoft Corporation'" in script
    assert kwargs['env']['PUYO_INSTALLER'] == str(tmp_path / "a' & file.exe")
    assert str(tmp_path) not in script


def test_entry_pth_and_copy_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import build_launchers as build
    assert 'tools\\launcher' in build.PTH and 'import site' in build.PTH
    source, dest = tmp_path / 'source.exe', tmp_path / 'dest.exe'
    source.write_bytes(b'good')
    dest.write_bytes(b'bad')
    run = Mock()
    monkeypatch.setattr(build.subprocess, 'run', run)
    with pytest.raises(ValueError, match='不一致'):
        build.verify_copy(source, dest)
    assert not run.called
    dest.write_bytes(b'good')
    run.side_effect = subprocess.CalledProcessError(1, 'signature')
    with pytest.raises(subprocess.CalledProcessError):
        build.verify_copy(source, dest)


@pytest.mark.parametrize('memory', [True, False])
def test_zip_helper_hidden(memory: bool, tmp_path: Path) -> None:
    import zipfile
    from bundle_archive import write_bundle_zip, HIDDEN_ROOT_FILES
    root = tmp_path / 'PuyoLive'
    root.mkdir()
    for name in (*HIDDEN_ROOT_FILES, 'puyoZeus Live.exe'):
        (root / name).write_bytes(b'example')
    archive = write_bundle_zip(root, tmp_path / 'test.zip', in_memory=memory)
    with zipfile.ZipFile(archive) as stream:
        for info in stream.infolist():
            assert bool(info.external_attr & 2) == (Path(info.filename).name in HIDDEN_ROOT_FILES)


@pytest.mark.parametrize('name', ['puyoZeus Live.exe', 'アンインストール.exe', 'python.exe'])
def test_sitecustomize_entry_only(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    call, leave = Mock(return_value=0), Mock()
    monkeypatch.setattr(launcher, 'main', call)
    monkeypatch.setattr(os, '_exit', leave)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / name))
    monkeypatch.setattr(sys, 'argv', [name, '--autostart'])
    runpy.run_path(str(Path(launcher.__file__).with_name('sitecustomize.py')))
    if name == 'python.exe':
        assert not call.called and not leave.called
    else:
        call.assert_called_once_with(tmp_path, ['--autostart'], name == 'アンインストール.exe')
        leave.assert_called_once_with(0)


def test_cleanup_keeps_current_helpers(tmp_path: Path) -> None:
    """旧版と同名でも、現行の案内で使うtoolsの補助スクリプトは残す。"""
    tools = tmp_path / 'tools'
    tools.mkdir()
    manifest = json.loads(Path(legacy.__file__).with_name('legacy_hashes.json').read_text('utf-8'))
    for name in ('SendLogs.ps1', 'UninstallCleanup.ps1'):
        assert name in manifest  # 一番上に残った0.6.0の補助は対象。
        assert 'tools/' + name not in manifest
        (tools / name).write_bytes(b'current helper')
    legacy.cleanup(tmp_path)
    assert all((tools / name).read_bytes() == b'current helper'
               for name in ('SendLogs.ps1', 'UninstallCleanup.ps1'))
