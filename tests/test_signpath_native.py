"""署名済み計算部品の拒否条件と既定時のバイト互換。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packaging'))
import signed_native as native
import build_launchers as launchers
from native_version import stamp_version, version_blob
from build_launchers import verify_launcher_version


def bundle_files(tmp_path: Path) -> tuple[Path, Path]:
    bundle, source = tmp_path / 'bundle', tmp_path / 'signed'
    source.mkdir()
    for name, relative in native.DESTINATIONS.items():
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'old-' + name.encode())
        (source / name).write_bytes(b'signed-' + name.encode())
    return bundle, source


def hashes(bundle: Path) -> dict[str, str]:
    return {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in bundle.rglob('*') if p.is_file()}


def test_default_bytes_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, _ = bundle_files(tmp_path)
    before = hashes(bundle)
    monkeypatch.setattr(native, 'verify_contents', lambda _: pytest.fail('既定時は検査しない'))
    native.install_signed_native(bundle, None, '0.6.4')
    assert hashes(bundle) == before
    assert len(before) == 2


@pytest.mark.parametrize('check', ['verify_signpath', 'verify_launcher_version', 'verify_contents'])
@pytest.mark.parametrize('failure_file', [native.PYD, native.DLL])
def test_rejection_preserves_both_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        check: str, failure_file: str) -> None:
    bundle, source = bundle_files(tmp_path)
    before = hashes(bundle)
    for function in ('verify_signpath', 'verify_launcher_version', 'verify_contents'):
        monkeypatch.setattr(native, function, lambda *args, **kwargs: None)
    def reject(path: Path, *args: object, **kwargs: object) -> None:
        if check == 'verify_contents' or path.name == failure_file:
            raise ValueError('不一致')
    monkeypatch.setattr(native, check, reject)
    with pytest.raises(ValueError, match='不一致'):
        native.install_signed_native(bundle, source, '0.6.4')
    assert hashes(bundle) == before


def test_verified_staged_bytes_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, source = bundle_files(tmp_path)
    calls = []
    def signature(path: Path, exact_signer: bool = False) -> None:
        assert exact_signer
        calls.append(('signature', path.name))
    monkeypatch.setattr(native, 'verify_signpath', signature)
    monkeypatch.setattr(native, 'verify_launcher_version', lambda p, v: calls.append((v, p.name)))
    def verify(staged: Path) -> None:
        assert staged != source
        assert all((staged / name).read_bytes() == (source / name).read_bytes()
                   for name in native.DESTINATIONS)
        calls.append(('contents', 'both'))
    monkeypatch.setattr(native, 'verify_contents', verify)
    native.install_signed_native(bundle, source, '0.6.4')
    assert len(calls) == 5
    assert all((bundle / relative).read_bytes() == (source / name).read_bytes()
               for name, relative in native.DESTINATIONS.items())


@pytest.mark.parametrize('version', ['v0.6.4', '0.6.4\n', '0.6.4-preview', '1.2.65536'])
def test_version_rejects_invalid_input(version: str) -> None:
    with pytest.raises(ValueError):
        version_blob(version)


def test_version_is_embedded() -> None:
    resource = version_blob('0.6.4')
    assert 'FileVersion\0'.encode('utf-16le') in resource
    assert 'ProductVersion\0'.encode('utf-16le') in resource
    assert resource.count('0.6.4\0'.encode('utf-16le')) == 2


@pytest.mark.skipif(sys.platform != 'win32', reason='WindowsのPEリソースAPI')
def test_windows_reads_embedded_version(tmp_path: Path) -> None:
    target = tmp_path / native.DLL
    shutil.copyfile(ROOT / 'src/phase_j' / native.DLL, target)
    stamp_version(target, '0.6.4')
    verify_launcher_version(target, '0.6.4')
    with pytest.raises(ValueError, match='一致しません'):
        verify_launcher_version(target, '0.6.3')


def test_missing_input_preserves_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, source = bundle_files(tmp_path)
    before = hashes(bundle)
    (source / native.DLL).unlink()
    monkeypatch.setattr(native, 'verify_signpath', lambda *args, **kwargs: None)
    monkeypatch.setattr(native, 'verify_launcher_version', lambda *args: None)
    with pytest.raises(FileNotFoundError):
        native.install_signed_native(bundle, source, '0.6.4')
    assert hashes(bundle) == before


@pytest.mark.parametrize('status,signer', [('Valid', 'Other signer'),
    ('Valid', 'SignPath Foundation other'), ('NotSigned', 'SignPath Foundation')])
def test_native_requires_exact_signer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      status: str, signer: str) -> None:
    monkeypatch.setenv('SystemRoot', str(tmp_path))
    monkeypatch.setattr(launchers.subprocess, 'CREATE_NO_WINDOW', 0, raising=False)
    result = SimpleNamespace(stdout=json.dumps(dict(status=status, signer=signer)))
    monkeypatch.setattr(launchers.subprocess, 'run', lambda *args, **kwargs: result)
    with pytest.raises(ValueError, match='有効な署名がありません'):
        launchers.verify_signpath(tmp_path / native.PYD, exact_signer=True)


def test_old_unsigned_dll_is_not_executed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    golden = json.loads((ROOT / 'packaging/native_golden.json').read_text('utf-8'))
    checked = []
    checks = SimpleNamespace(verify_native=lambda _: {'fingerprint': dict.fromkeys(range(9))})
    monkeypatch.setattr(native, 'load_module', lambda *args: checks)
    def fingerprint(path: Path) -> str:
        checked.append(path)
        return golden['dll_fingerprint']
    monkeypatch.setattr(native, 'dll_fingerprint', fingerprint)
    report = native.probe(tmp_path, ROOT / 'src/phase_j' / native.DLL)
    assert checked == [tmp_path / native.DLL]
    assert report['pyd_fingerprints'] == 9
    assert report['dll_cases'] == 480
