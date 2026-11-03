"""実行時ファイルの混入と、配布ステージでのsmoke実行を防ぐ。"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packaging'))
from bundle_archive import extract_smoke_copy, repack_archive, write_bundle_zip
from build_bundle import make_zip
import build_bundle
import build_release

RUNTIME_FILES = ('app/config/device_calibration/device.json', 'app/logs/run.jsonl',
                 'logs/run.log', 'app/cache/result.json', 'app/output/frame.png',
                 'smoke/result.json', 'verification/polls.json', 'app/session.log')
PRODUCT_FILES = ('puyo_live.json', 'app/config/live_defaults.json', 'app/src/live.py',
                 'app/src/__pycache__/live.cpython-312.pyc',
                 'python/Lib/site-packages/library/cache/module.py')


def put(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(relative.encode())


@pytest.mark.parametrize('runtime', RUNTIME_FILES)
def test_build_zip_excludes_existing_runtime(tmp_path: Path, runtime: str) -> None:
    """較正JSON等が既に存在するステージをZIP化しても製品へ含めない。"""
    bundle = tmp_path / 'PuyoLive'
    for relative in (*PRODUCT_FILES, runtime):
        put(bundle, relative)
    with zipfile.ZipFile(make_zip(bundle)) as archive:
        assert set(archive.namelist()) == {f'PuyoLive/{name}' for name in PRODUCT_FILES}
    assert (bundle / runtime).exists()  # 作業データ自体は削除しない。


def test_repack_preserves_all_other_bytes_and_metadata(tmp_path: Path) -> None:
    source, dest = tmp_path / 'a.zip', tmp_path / 'b.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        for name in (*PRODUCT_FILES, RUNTIME_FILES[0]):
            archive.writestr(f'PuyoLive/{name}', name.encode())
    before = source.read_bytes()
    assert repack_archive(source, dest) == [f'PuyoLive/{RUNTIME_FILES[0]}']
    assert source.read_bytes() == before
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(dest) as new:
        for entry in new.infolist():
            assert new.read(entry) == old.read(entry.filename)
            assert entry.date_time == old.getinfo(entry.filename).date_time


def test_in_memory_archive_preserves_members_and_refuses_overwrite(tmp_path: Path) -> None:
    bundle = tmp_path / 'PuyoLive'
    for relative in (*PRODUCT_FILES, RUNTIME_FILES[0]):
        put(bundle, relative)
    archives = [write_bundle_zip(bundle, tmp_path / f'{mode}.zip', in_memory=mode)
                for mode in (False, True)]
    with zipfile.ZipFile(archives[0]) as first, zipfile.ZipFile(archives[1]) as second:
        assert first.namelist() == second.namelist()
        assert second.testzip() is None
        assert all(first.read(name) == second.read(name) for name in first.namelist())
    before = archives[1].read_bytes()
    with pytest.raises(FileExistsError):
        write_bundle_zip(bundle, archives[1], in_memory=True)
    assert archives[1].read_bytes() == before


def test_declared_archive_hashes_match_reopened_members(tmp_path: Path) -> None:
    """組立時に読んだ全bytesを、完成ZIPの再読込みで独立に照合する。"""
    from io import BytesIO
    bundle = tmp_path/'PuyoLive'
    names = ['app/source.py', 'python/runtime.dll', 'output/runtime.log']
    for name in names:
        put(bundle, name)
    sources: dict[str, str] = {}
    path = write_bundle_zip(bundle, tmp_path/'declared.zip', in_memory=True,
                            relative_files=names, source_hashes=sources)
    with zipfile.ZipFile(BytesIO(path.read_bytes())) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {'PuyoLive/app/source.py', 'PuyoLive/python/runtime.dll'}
        actual = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    assert actual == sources


@pytest.mark.parametrize('preload', [False, True])
def test_smoke_writes_only_to_extracted_copy(tmp_path: Path, preload: bool) -> None:
    bundle = tmp_path / 'stage/PuyoLive'
    for relative in PRODUCT_FILES:
        put(bundle, relative)
    archive = make_zip(bundle)
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    smoke = extract_smoke_copy(archive, tmp_path / 'smoke', preload=preload)
    for relative in PRODUCT_FILES:
        assert (smoke / relative).read_bytes() == (bundle / relative).read_bytes()
    put(smoke, RUNTIME_FILES[0])
    assert not (bundle / RUNTIME_FILES[0]).exists()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        extract_smoke_copy(archive, tmp_path / 'smoke', preload=preload)


def test_smoke_requires_completed_zip(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_smoke_copy(tmp_path / 'missing.zip', tmp_path / 'smoke')
    assert not (tmp_path / 'smoke').exists()


@pytest.mark.parametrize('preload', [False, True])
@pytest.mark.parametrize('name', ['PuyoLive/../../outside.txt', 'PuyoLive/..\\outside.txt',
                                  'PuyoLive/C:/outside.txt'])
def test_smoke_rejects_outside_path(tmp_path: Path, preload: bool, name: str) -> None:
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr(name, 'bad')
    with pytest.raises(ValueError):
        extract_smoke_copy(archive, tmp_path / 'smoke', preload=preload)
    assert not (tmp_path / 'smoke').exists()


def test_asset_collection_excludes_device_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """資産ディレクトリに較正が紛れてもMANIFESTの入力へ渡さない。"""
    for name in ('config/live_defaults.json', 'config/device_calibration/device.json'):
        put(tmp_path, name)
    monkeypatch.setattr(build_bundle, 'ROOT', tmp_path)
    monkeypatch.setattr(build_bundle, 'ENTRY_MODULES', ())
    monkeypatch.setattr(build_bundle, 'read_asset_list', lambda: ['config'])
    assert build_bundle.collect_app_files(tmp_path) == ['config/live_defaults.json']


def test_installer_rejects_dirty_stage(tmp_path: Path) -> None:
    """ZIP以外の配布経路も実行状態が残ったまま生成しない。"""
    bundle = tmp_path / 'PuyoLive'
    put(bundle, RUNTIME_FILES[0])
    with pytest.raises(ValueError, match='実行時ファイル'):
        build_release.artifacts(tmp_path, bundle, 'test')
