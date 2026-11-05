"""署名済み入口の拒否条件と従来方式のバイト互換を確認する。"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHILD_EXIT_CODE = 37
PROCESS_TIMEOUT = 15
SAC_REASON = 'SAC blocked unsigned test build'
sys.path.insert(0, str(ROOT / 'packaging'))
import build_launchers as launchers
import build_launcher_exe as builder


def sac_blocked(executable: Path, started: datetime) -> bool:
    """今回の入口と SHA256 が一致する SAC の拒否だけを認める。"""
    script = r"""
$ErrorActionPreference = 'Stop'
try {
    $events = Get-WinEvent -FilterHashtable @{
        LogName='Microsoft-Windows-CodeIntegrity/Operational'; Id=3077
        StartTime=([DateTimeOffset]::Parse($env:TEST_STARTED)).LocalDateTime
    }
} catch {
    if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { exit 0 }
    throw
}
foreach ($event in $events) {
    $data = @{}
    ([xml]$event.ToXml()).Event.EventData.Data | ForEach-Object {
        $data[$_.Name] = $_.InnerText
    }
    if ($data['PolicyName'] -eq 'VerifiedAndReputableDesktop' -and
        $data['SHA256FlatHash'] -eq $env:TEST_SHA256) { 'matched'; break }
}
"""
    env = dict(os.environ, TEST_STARTED=started.isoformat(),
               TEST_SHA256=hashlib.sha256(executable.read_bytes()).hexdigest())
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive',
                             '-Command', script], env=env, capture_output=True,
                            timeout=PROCESS_TIMEOUT)
    return result.returncode == 0 and result.stdout.strip() == b'matched'


def run_test_exe(executable: Path, args: list[str]) -> subprocess.CompletedProcess:
    """起動失敗でも入口の SAC 拒否を立証できない限り失敗にする。"""
    started = datetime.now(timezone.utc)
    try:
        return subprocess.run([str(executable), *args], input=b'\n', capture_output=True,
                              timeout=PROCESS_TIMEOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    except OSError:
        if sac_blocked(executable, started):
            pytest.skip(SAC_REASON)
        raise


def copy_python(runtime: Path) -> None:
    """配布用または CI の PSF Python を、標準ライブラリとともに写す。"""
    source = Path(os.environ.get('SIGNPATH_TEST_PYTHON', sys.base_prefix))
    for name in ('python.exe', 'pythonw.exe', 'python312.dll'):
        shutil.copyfile(source / name, runtime / name)
    for path in source.glob('*.dll'):
        shutil.copyfile(path, runtime / path.name)
    archive = source / 'python312.zip'
    if archive.is_file():
        shutil.copyfile(archive, runtime / archive.name)
    else:
        shutil.copytree(source / 'Lib', runtime / 'Lib',
                        ignore=shutil.ignore_patterns('site-packages', '__pycache__'))
    # 元の _pth にある製品への参照を持ち込まない。
    (runtime / 'python312._pth').write_text('python312.zip\nLib\n.\n..\n', encoding='utf-8')


def write_fake_modules(root: Path) -> None:
    """起動引数そのものと、入口から渡された値の両方を保存する。"""
    launcher = root / 'tools/launcher/signed_launcher.py'
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        'import json, sys\nfrom pathlib import Path\n'
        'def main(root, args, legacy):\n'
        '    data = dict(original=sys.orig_argv[1:], root=str(root), args=args, legacy=legacy)\n'
        '    (root / "received.json").write_text(json.dumps(data), encoding="utf-8")\n'
        f'    return {CHILD_EXIT_CODE}\n', encoding='utf-8')
    uninstall = root / 'src/phase_j/distribution_uninstall.py'
    uninstall.parent.mkdir(parents=True)
    uninstall.write_text(
        'import json, sys\nfrom pathlib import Path\n'
        'def main():\n'
        '    root = Path(sys.argv[2])\n'
        '    data = dict(original=sys.orig_argv[1:], root=str(root), args=sys.argv[1:])\n'
        '    (root / "received.json").write_text(json.dumps(data), encoding="utf-8")\n'
        f'    return {CHILD_EXIT_CODE}\n'
        'sys.exit(main())\n', encoding='utf-8')


@pytest.mark.parametrize('blocked', [False, True])
def test_start_failure_requires_sac_evidence(blocked: bool, monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError('起動失敗')

    monkeypatch.setattr(subprocess, 'run', fail)
    monkeypatch.setattr(subprocess, 'CREATE_NO_WINDOW', 0, raising=False)
    monkeypatch.setattr(sys.modules[__name__], 'sac_blocked', lambda *args: blocked)
    expected = pytest.skip.Exception if blocked else OSError
    with pytest.raises(expected, match=SAC_REASON if blocked else '起動失敗'):
        run_test_exe(tmp_path / 'launcher.exe', [])


def test_child_failure_is_not_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = subprocess.CompletedProcess([], 1)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: result)
    monkeypatch.setattr(subprocess, 'CREATE_NO_WINDOW', 0, raising=False)
    monkeypatch.setattr(sys.modules[__name__], 'sac_blocked', lambda *args: pytest.fail('不要な照合'))
    assert run_test_exe(tmp_path / 'launcher.exe', []).returncode == 1


@pytest.mark.skipif(os.name != 'nt', reason='Windows のイベント照合を検証')
@pytest.mark.parametrize('same_hash,policy,expected', [
    (True, 'VerifiedAndReputableDesktop', True),
    (False, 'VerifiedAndReputableDesktop', False),
    (True, 'OtherPolicy', False),
])
def test_sac_event_match(same_hash: bool, policy: str, expected: bool,
                         monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / 'launcher.exe'
    executable.write_bytes(b'event hash fixture')
    digest = hashlib.sha256(executable.read_bytes()).hexdigest() if same_hash else '0' * 64
    xml = ('<Event><EventData><Data Name="PolicyName">' + policy + '</Data>'
           '<Data Name="SHA256FlatHash">' + digest + '</Data></EventData></Event>')
    prefix = ("function Get-WinEvent { param($FilterHashtable)\n"
              "if ($FilterHashtable.Id -ne 3077) { throw 'wrong event' }\n"
              "$event = New-Object PSObject\n"
              "$event | Add-Member ScriptMethod ToXml { '" + xml + "' }\n$event\n}\n")
    original_run = subprocess.run

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        return original_run([*command[:-1], prefix + command[-1]], **kwargs)

    monkeypatch.setattr(subprocess, 'run', run)
    assert sac_blocked(executable, datetime.now(timezone.utc)) is expected


@pytest.fixture(scope='module')
def compiled(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if os.name != 'nt':
        pytest.skip('Windows の標準 csc が必要')
    output = tmp_path_factory.mktemp('compiled')
    builder.build(output, '0.6.2')
    return output


@pytest.mark.parametrize('status,signer,accepted', [
    ('NotSigned', '', False), ('UnknownError', 'SignPath Foundation', False),
    ('HashMismatch', 'SignPath Foundation', False),
    ('Valid', 'Python Software Foundation', False), ('Valid', '', False),
    ('Valid', 'SignPath Foundation', True),
])
def test_signature(status: str, signer: str, accepted: bool,
                   monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('SystemRoot', 'C:/Windows')
    monkeypatch.setattr(launchers.subprocess, 'CREATE_NO_WINDOW', 0, raising=False)
    response = SimpleNamespace(stdout=json.dumps(dict(status=status, signer=signer)))
    monkeypatch.setattr(launchers.subprocess, 'run', lambda *a, **kw: response)
    if accepted:
        launchers.verify_signpath(tmp_path / 'launcher.exe')
    else:
        with pytest.raises(ValueError):
            launchers.verify_signpath(tmp_path / 'launcher.exe')


def test_default_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / 'bundle'
    (bundle / 'python').mkdir(parents=True)
    monkeypatch.setattr(launchers, 'verify_copy', lambda *args: None)
    for name in launchers.COPIES.values():
        (bundle / 'python' / name).write_bytes(name.encode())
    launchers.build_launchers(bundle, '0.6.0')
    expected = {name: original.encode() for name, original in launchers.COPIES.items()}
    expected.update({name + '._pth': launchers.PTH.replace('\n', os.linesep).encode() for name in
                     ('puyoZeus Live', 'アンインストール')})
    expected.update({'python/' + name: name.encode() for name in launchers.COPIES.values()})
    expected.update({'tools/launcher/' + p.relative_to(ROOT / 'packaging/launcher').as_posix():
                     p.read_bytes() for p in (ROOT / 'packaging/launcher').rglob('*') if p.is_file()})
    actual = {p.relative_to(bundle).as_posix(): p.read_bytes()
              for p in bundle.rglob('*') if p.is_file()}
    assert actual == expected


def test_signed_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, signed = tmp_path / 'bundle', tmp_path / 'signed'
    bundle.mkdir()
    signed.mkdir()
    calls = []
    for name in builder.NAMES:
        (signed / name).write_bytes(name.encode())
    monkeypatch.setattr(launchers, 'verify_signpath', lambda path: calls.append(path))
    versions = []
    monkeypatch.setattr(launchers, 'verify_launcher_version',
                        lambda path, version: versions.append((path, version)))
    launchers.build_launchers(bundle, '0.6.0', signed)
    assert len(calls) == 4
    assert versions == [(p, '0.6.0') for p in calls]
    assert {p.name for p in bundle.iterdir()} == {*builder.NAMES, 'tools'}
    assert all((bundle / name).read_bytes() == (signed / name).read_bytes() for name in builder.NAMES)


def test_reject_before_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / 'bundle'
    bundle.mkdir()

    def reject(path: Path) -> None:
        if path.name == builder.NAMES[1]:
            raise ValueError('署名なし')

    monkeypatch.setattr(launchers, 'verify_signpath', reject)
    with pytest.raises(ValueError):
        launchers.build_launchers(bundle, '0.6.0', tmp_path / 'signed')
    assert not list(bundle.iterdir())


@pytest.mark.parametrize('version', ['x', '1.0', '1.2.3"', '65535.0.0'])
def test_bad_version(version: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        builder.build(tmp_path, version)


@pytest.mark.skipif(os.name != 'nt', reason='Windows の標準 csc が必要')
def test_build_reproducible(tmp_path: Path, compiled: Path) -> None:
    first, second = compiled, tmp_path / '日本語 second'
    builder.build(second, '0.6.2')
    for name in builder.NAMES:
        assert (first / name).read_bytes() == (second / name).read_bytes()
        launchers.verify_launcher_version(first / name, '0.6.2')
        with pytest.raises(ValueError):
            launchers.verify_launcher_version(first / name, '0.6.0')
    with pytest.raises(FileExistsError):
        builder.build(first, '0.6.0')


@pytest.mark.parametrize('file,product', [('0.6.0', '0.6.2'), ('0.6.2', '0.6.0'), ('', '')])
def test_version_rejection(file: str, product: str, tmp_path: Path,
                           monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(stdout=json.dumps(dict(file=file, product=product)))
    monkeypatch.setattr(launchers.subprocess, 'run', lambda *a, **kw: response)
    monkeypatch.setattr(launchers.subprocess, 'CREATE_NO_WINDOW', 0, raising=False)
    monkeypatch.setenv('SystemRoot', 'C:/Windows')
    with pytest.raises(ValueError):
        launchers.verify_launcher_version(tmp_path / 'launcher.exe', '0.6.2')


@pytest.mark.parametrize('uninstall', [False, True])
def test_exe_child_arguments(uninstall: bool, compiled: Path, tmp_path: Path) -> None:
    root = tmp_path / '日本語 space'
    runtime = root / 'python'
    runtime.mkdir(parents=True)
    copy_python(runtime)
    write_fake_modules(root)
    name = builder.NAMES[int(uninstall)]
    shutil.copyfile(compiled / name, root / name)
    args = ['--autostart', '', '日本語 space', 'a"b', 'tail\\']
    result = run_test_exe(root / name, args)
    assert result.returncode == CHILD_EXIT_CODE
    data = json.loads((root / 'received.json').read_text(encoding='utf-8'))
    received = data['original']
    assert data['root'] == str(root)
    if uninstall:
        assert received == ['-s', '-m', 'src.phase_j.distribution_uninstall',
                            '--root', str(root), '--interactive']
        assert b'Enter' in result.stdout
        assert data['args'] == ['--root', str(root), '--interactive']
    else:
        assert received[:2] == ['-s', '-c']
        assert 'sys.exit(main(root,sys.argv[2:],False))' in received[2]
        assert received[3:] == [str(root), *args]
        assert data['args'] == args
        assert data['legacy'] is False


def test_exe_start_error_is_console(compiled: Path, tmp_path: Path) -> None:
    runtime = tmp_path / 'python'
    runtime.mkdir()
    (runtime / 'python.exe').write_bytes(b'not an executable')
    (runtime / 'python312.dll').touch()
    executable = tmp_path / builder.NAMES[1]
    shutil.copyfile(compiled / builder.NAMES[1], executable)
    result = run_test_exe(executable, [])
    assert result.returncode == 1
    assert b'Enter' in result.stdout
    assert len(result.stdout.splitlines()) >= 2
