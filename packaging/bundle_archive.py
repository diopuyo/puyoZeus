"""実行時の状態を配布せず、完成ZIPの別コピーで起動検証する。"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import zipfile
from hashlib import sha256
from typing import Callable

# 事前生成した __pycache__/*.pyc は製品資産なので保持する。
RUNTIME_DIRS = frozenset({'logs', 'log', 'cache', '.cache', '.pytest_cache',
                          'output', 'outputs', 'smoke', 'verification'})
RUNTIME_SUFFIXES = frozenset({'.log', '.tmp'})
EXTRACTION_WORKERS = 32
MAX_PRELOAD_BYTES = 1024**3


def is_runtime_file(relative: str) -> bool:
    """PuyoLive直下を基準に、実行時だけのファイルを判定する。"""
    path = PurePosixPath(relative)
    parts = tuple(part.lower() for part in path.parts)
    if parts[:3] == ('app', 'config', 'device_calibration'):
        return True
    # ライブラリのcacheモジュールなどは除外しない。
    local = parts[1:] if parts[:1] == ('app',) else parts
    return bool(local and (local[0] in RUNTIME_DIRS
                or path.suffix.lower() in RUNTIME_SUFFIXES))


def write_bundle_zip(bundle: Path, archive: Path, in_memory: bool = False,
                     relative_files: list[str] | None = None,
                     source_hashes: dict[str, str] | None = None,
                     progress: Callable[[int, int], None] | None = None) -> Path:
    """汚れた作業フォルダからでも実行時ファイルを除いてZIPを作る。"""
    if in_memory and archive.exists():
        raise FileExistsError(archive)
    destination = BytesIO() if in_memory else archive
    paths = sorted(bundle/rel for rel in relative_files) if relative_files is not None else sorted(bundle.rglob('*'))
    paths = [p for p in paths if not is_runtime_file(p.relative_to(bundle).as_posix())
             and (relative_files is not None or p.is_file())]
    with zipfile.ZipFile(destination, 'w' if in_memory else 'x', compression=zipfile.ZIP_DEFLATED) as output:
        if in_memory:
            with ThreadPoolExecutor(max_workers=EXTRACTION_WORKERS) as workers:
                for index, (path, payload) in enumerate(zip(paths, workers.map(Path.read_bytes, paths)), 1):
                    name = f'{bundle.name}/{path.relative_to(bundle).as_posix()}'
                    output.writestr(name, payload)
                    if source_hashes is not None:
                        source_hashes[name] = sha256(payload).hexdigest()
                    if progress is not None:
                        progress(index, len(paths))
        else:
            for path in paths:
                output.write(path, f'{bundle.name}/{path.relative_to(bundle).as_posix()}')
    if in_memory:
        with archive.open('xb') as stream:
            stream.write(destination.getbuffer())
    return archive


def repack_archive(source: Path, destination: Path) -> list[str]:
    """完成ZIPから実行時ファイルだけを除く。pycを再生成せず各内容を保持する。"""
    removed: list[str] = []
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, 'x') as output:
        for entry in original.infolist():
            relative = PurePosixPath(entry.filename).relative_to('PuyoLive').as_posix()
            if not entry.is_dir() and is_runtime_file(relative):
                removed.append(entry.filename)
            else:
                output.writestr(entry, original.read(entry))
    return removed


def extract_smoke_copy(archive: Path, destination: Path, preload: bool = False) -> Path:
    """ZIP作成後に、新規の別ディレクトリへ展開する。配布用ステージでは起動しない。"""
    # ネットワーク上のZIPへ細かなseekを繰り返さない。既定経路は維持する。
    if preload and archive.stat().st_size > MAX_PRELOAD_BYTES:
        raise ValueError('一括読込みは1GiB以下の検収済みZIPだけに使用する')
    source_file = BytesIO(archive.read_bytes()) if preload else archive
    with zipfile.ZipFile(source_file) as source:
        for entry in source.infolist():
            path = PurePosixPath(entry.filename)
            if ('\\' in entry.filename or ':' in entry.filename or not path.parts
                    or path.is_absolute() or '..' in path.parts or path.parts[0] != 'PuyoLive'):
                raise ValueError(f'展開できないZIPパス: {entry.filename}')
        destination.mkdir(parents=True, exist_ok=False)
        if preload:
            directories = {destination / PurePosixPath(e.filename).parent for e in source.infolist()}
            with ThreadPoolExecutor(max_workers=EXTRACTION_WORKERS) as workers:
                list(workers.map(make_directory, directories))
                list(workers.map(partial(extract_member, source, destination), source.infolist()))
        else:
            source.extractall(destination)
    return destination / 'PuyoLive'


def make_directory(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)


def extract_member(source: zipfile.ZipFile, destination: Path, entry: zipfile.ZipInfo) -> None:
    """検査済みのメンバーを、事前作成した親へ書く。余分なSMBのstatを避ける。"""
    target = destination / entry.filename
    if entry.is_dir():
        target.mkdir(exist_ok=True)
    else:
        target.write_bytes(source.read(entry))
