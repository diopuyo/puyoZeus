"""配布物 MANIFEST の生成・照合 (src/phase_j/launcher_manifest.py) の試験。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.phase_j.launcher_manifest import (ManifestError, ManifestReport, build_manifest, installed_versions,
                                           load_manifest, normalize_name, sha256_file, verify_manifest)


def make_tree(root: Path) -> list[Path]:
    (root / 'a').mkdir(parents=True)
    files = [root / 'a' / 'x.bin', root / 'y.json']
    files[0].write_bytes(b'x' * 3_000_000)  # 読み込み塊 (1MiB) を跨ぐ大きさ
    files[1].write_text('{}', encoding='utf-8')
    return files


def write_manifest(root: Path, files: list[Path], packages: dict | None = None) -> Path:
    path = root / 'MANIFEST.json'
    path.write_text(json.dumps(build_manifest(root, files, packages)), encoding='utf-8')
    return path


def test_roundtrip_ok_and_counts_population(tmp_path: Path) -> None:
    files = make_tree(tmp_path)
    report = verify_manifest(tmp_path, write_manifest(tmp_path, files))
    assert report.ok and report.checked == 2 and report.checked_bytes == 3_000_002


def test_manifest_uses_posix_relative_keys(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path, make_tree(tmp_path))
    assert set(manifest['files']) == {'a/x.bin', 'y.json'}
    assert manifest['files']['y.json']['sha256'] == sha256_file(tmp_path / 'y.json')


def test_detects_missing_size_and_hash(tmp_path: Path) -> None:
    files = make_tree(tmp_path)
    path = write_manifest(tmp_path, files)
    (tmp_path / 'y.json').write_text('{"a":1}', encoding='utf-8')  # サイズ違い
    (tmp_path / 'a' / 'x.bin').write_bytes(b'y' * 3_000_000)  # 同サイズ・別内容
    (tmp_path / 'gone.txt').write_text('z')
    manifest = load_manifest(path)
    manifest['files']['gone.txt'] = dict(size=1, sha256='0' * 64)
    (tmp_path / 'gone.txt').unlink()
    path.write_text(json.dumps(manifest), encoding='utf-8')
    report = verify_manifest(tmp_path, path)
    assert not report.ok
    assert report.missing == ['gone.txt'] and report.size_mismatch == ['y.json']
    assert report.hash_mismatch == ['a/x.bin']
    assert all(word in report.summary() for word in ('欠落', 'サイズ不一致', 'SHA256 不一致'))


def test_empty_manifest_is_not_success(tmp_path: Path) -> None:
    """母数 0 の合格は「測っていない」。ok にしない。"""
    path = write_manifest(tmp_path, [])
    report = verify_manifest(tmp_path, path)
    assert report.checked == 0 and not report.ok and '0 件' in report.summary()


def test_all_files_missing_is_not_success(tmp_path: Path) -> None:
    files = make_tree(tmp_path)
    path = write_manifest(tmp_path, files)
    for file in files:
        file.unlink()
    assert not verify_manifest(tmp_path, path).ok


@pytest.mark.parametrize('content', ['not json', '{"schema": 99, "files": {}}', '{"schema": 1}', '[]'])
def test_bad_manifest_raises(tmp_path: Path, content: str) -> None:
    path = tmp_path / 'MANIFEST.json'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_missing_manifest_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match='読めません'):
        load_manifest(tmp_path / 'none.json')


def test_installed_versions_from_dist_info(tmp_path: Path) -> None:
    for name in ('numpy-2.4.4.dist-info', 'scikit_learn-1.8.0.dist-info', 'torch-2.5.1+cpu.dist-info'):
        (tmp_path / name).mkdir()
    assert installed_versions(tmp_path) == {'numpy': '2.4.4', 'scikit-learn': '1.8.0', 'torch': '2.5.1+cpu'}
    assert normalize_name('Scikit_Learn') == 'scikit-learn'


def test_package_version_mismatch_is_reported(tmp_path: Path) -> None:
    files = make_tree(tmp_path)
    site = tmp_path / 'site'
    site.mkdir()
    (site / 'numpy-2.4.4.dist-info').mkdir()
    path = write_manifest(tmp_path, files, {'numpy': '2.4.4', 'scikit-learn': '1.8.0', 'torch': '2.5.1+cpu'})
    report = verify_manifest(tmp_path, path, site)
    assert not report.ok and len(report.package_mismatch) == 2
    assert 'scikit-learn' in report.summary() and 'None' in report.summary()


def test_summary_truncates_long_lists() -> None:
    report = ManifestReport(checked=1, missing=[f'f{i}' for i in range(50)])
    assert 'ほか 30 件' in report.summary()
