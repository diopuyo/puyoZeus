"""標準ZIPの全内容と、対象外のコード読取りを変えないことを確認する。"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import zipimport

import pytest


def cache_module() -> object:
    source = Path(__file__).resolve().parents[1]/'packaging/obs_stdlib_cache.py'
    spec = importlib.util.spec_from_file_location('obs_stdlib_cache_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_stdlib_entries_equal_with_one_file_read(monkeypatch: pytest.MonkeyPatch) -> None:
    module = cache_module()
    archive = next(path for path in sys.path if Path(path).name == module.ARCHIVE_NAME)
    importer = zipimport.zipimporter(archive)
    expected = {name: hashlib.sha256(zipimport._get_data(archive, entry)).hexdigest()
                for name, entry in importer._files.items()}
    original, opened = zipimport._io, []

    class TrackedIO:
        def __getattr__(self, name: str) -> object:
            return getattr(original, name)

        def open_code(self, path: str) -> object:
            opened.append(path)
            return original.open_code(path)

    monkeypatch.setattr(zipimport, '_io', TrackedIO())
    monkeypatch.setenv(module.FLAG, '1')
    assert module.install([archive]) == {module.key(archive): Path(archive).stat().st_size}
    installed = zipimport._io
    cache_module()
    assert zipimport._io is installed
    actual = {name: hashlib.sha256(zipimport._get_data(archive, entry)).hexdigest()
              for name, entry in importer._files.items()}
    assert actual == expected and len(expected) > 100
    assert opened == [archive]
    assert module.install([archive]) == {module.key(archive): Path(archive).stat().st_size}
    with zipimport._io.open_code(archive) as first, zipimport._io.open_code(archive) as second:
        assert first.read(10) == second.read(10)
        assert first.read(10) == second.read(10)
    assert opened == [archive]
    assert sys.modules['_io'] is original
    source = str(Path(__file__).resolve())
    with zipimport._io.open_code(source) as stream:
        assert stream.read() == Path(source).read_bytes()
    assert opened == [archive, source]


def test_default_off_keeps_zipimport_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_OBS_STDLIB_CACHE', raising=False)
    original = zipimport._io
    module = cache_module()
    assert module.install() == {} and zipimport._io is original


def test_oversized_archive_keeps_original_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    from types import SimpleNamespace
    module = cache_module()
    original = SimpleNamespace(open_code=lambda path: io.BytesIO(b'x'*(module.MAX_ARCHIVE_BYTES+1)))
    monkeypatch.setattr(zipimport, '_io', original)
    monkeypatch.setenv(module.FLAG, '1')
    assert module.install([module.ARCHIVE_NAME]) == {}
    assert zipimport._io is original
