"""配布全体を組み立てず、設定窓の依存取得と原文収録を確認する。"""
from pathlib import Path
import importlib.util
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('window_release_build', ROOT / 'packaging/build_release.py')
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_download_targets_windows312_even_with_cached_pystray(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheels = tmp_path / 'camera_wheels'
    wheels.mkdir()
    (wheels / 'pystray-0.19.5-py2.py3-none-any.whl').touch()
    run = Mock()
    monkeypatch.setattr(BUILD.subprocess, 'run', run)
    BUILD.prepare_window_wheels(tmp_path)
    assert run.call_count == 2
    download = run.call_args_list[0].args[0]
    assert 'pywebview==6.1' in download
    assert download[download.index('--platform') + 1] == 'win_amd64'
    assert download[download.index('--python-version') + 1] == '312'
    assert download[download.index('--abi') + 1] == 'cp312'
    wheel = run.call_args_list[1].args[0]
    assert 'wheel' in wheel and 'proxy_tools==0.1.0' in wheel
    assert '--no-deps' in wheel


def test_missing_wheel_licenses_are_copied_verbatim(tmp_path: Path) -> None:
    BUILD.copy_window_licenses(tmp_path)
    paths = ('proxy_tools-0.1.0/LICENSE.txt',
             'cffi-2.1.1/libffi-NOTICES.txt',
             'Microsoft.Web.WebView2-1.0.2957.106/LICENSE.txt',
             'Microsoft.Web.WebView2-1.0.2957.106/NOTICE.txt')
    assert len(list((tmp_path / 'LICENSES').rglob('*.txt'))) == len(paths)
    for name in paths:
        assert (tmp_path / 'LICENSES' / name).read_bytes() == (
            ROOT / 'packaging/licenses/app_window' / name).read_bytes()


def test_same_name_licenses_keep_relative_directories(tmp_path: Path) -> None:
    site = tmp_path / 'python/Lib/site-packages'
    for relative, text in (('numpy-2.4.4.dist-info/licenses/LICENSE', 'root'),
                           ('numpy-2.4.4.dist-info/licenses/vendor/a/LICENSE', 'vendor a'),
                           ('numpy-2.4.4.dist-info/licenses/vendor/b/LICENSE', 'vendor b'),
                           ('onnxruntime/LICENSE', 'onnx'),
                           ('onnxruntime/ThirdPartyNotices.txt', 'onnx notice')):
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    BUILD.copy_release_licenses(tmp_path)
    for relative, expected in (('LICENSE', 'root'), ('vendor/a/LICENSE', 'vendor a'),
                               ('vendor/b/LICENSE', 'vendor b')):
        assert (tmp_path / 'LICENSES/numpy-2.4.4' / relative).read_text() == expected
    assert (tmp_path / 'LICENSES/onnxruntime/ThirdPartyNotices.txt').read_text() == 'onnx notice'


def test_release_keeps_python_and_wheel_runtime_only(tmp_path: Path) -> None:
    import zipfile
    bundle = tmp_path / 'bundle'
    python = bundle / 'python'
    wheel = python / 'Lib/site-packages/example/msvcp140.dll'
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b'wheel original')
    (python / 'msvcp140.dll').write_bytes(b'system copy')
    (python / 'vcruntime140.dll').write_bytes(b'stale copy')
    downloads = tmp_path / 'downloads'
    downloads.mkdir()
    with zipfile.ZipFile(downloads / 'python-3.12.3-embed-amd64.zip', 'w') as archive:
        archive.writestr('vcruntime140.dll', b'python original')
    BUILD.retain_python_vc_runtime(bundle, tmp_path)
    assert not (python / 'msvcp140.dll').exists()
    assert (python / 'vcruntime140.dll').read_bytes() == b'python original'
    assert wheel.read_bytes() == b'wheel original'
