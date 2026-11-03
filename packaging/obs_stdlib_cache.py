"""凍結した標準ライブラリZIPだけを有界に保持する本人試験用の起動補助。"""
from __future__ import annotations

import _io
import os
import sys
import zipimport

FLAG = 'PUYO_OBS_STDLIB_CACHE'
MAX_ARCHIVE_BYTES = 16*1024*1024
ARCHIVE_NAME = f'python{sys.version_info.major}{sys.version_info.minor}.zip'


def key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


class CachedZipIO:
    """zipimport専用のIO代理。ほかのモジュールの_ioは変更しない。"""

    _puyo_stdlib_cache = True

    def __init__(self, original: object, archives: dict[str, bytes]) -> None:
        self.original = original
        self.archives = archives

    def __getattr__(self, name: str) -> object:
        return getattr(self.original, name)

    def open_code(self, path: str) -> object:
        payload = self.archives.get(key(path)) if isinstance(path, str) else None
        if payload is not None:
            return _io.BytesIO(payload)
        return getattr(self.original, 'open_code')(path)


def install(paths: list[str] | None = None) -> dict[str, int]:
    """既定OFF。展開・検証・復号の実装は元のzipimportをそのまま使う。"""
    if os.environ.get(FLAG) != '1':
        return {}
    if getattr(zipimport._io, '_puyo_stdlib_cache', False):
        return {name: len(data) for name, data in zipimport._io.archives.items()}
    original, archives = zipimport._io, {}
    for path in sys.path if paths is None else paths:
        if not isinstance(path, str) or os.path.basename(path).lower() != ARCHIVE_NAME:
            continue
        try:
            with getattr(original, 'open_code')(path) as stream:
                payload = stream.read(MAX_ARCHIVE_BYTES+1)
        except OSError:
            continue
        if len(payload) <= MAX_ARCHIVE_BYTES:
            archives[key(path)] = payload
            break
    if archives:
        zipimport._io = CachedZipIO(original, archives)
    return {name: len(data) for name, data in archives.items()}


install()
