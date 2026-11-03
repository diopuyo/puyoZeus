"""旧経路の全項目と終端状態を保存する、検査付きの決定的キャッシュ。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pickle
from pathlib import Path
from typing import Any

from joblib.externals import cloudpickle

CHUNK_FRAMES = 300
VERIFY_FRAMES = 8
HASH_BLOCK_BYTES = 1024 * 1024
CACHE_VERSION = 1
PACKAGES = ('numpy', 'opencv-python', 'opencv-python-headless', 'torch', 'scikit-learn', 'joblib')


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_sha(paths: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for label, path in sorted(paths):
        digest.update(label.encode())
        digest.update(bytes.fromhex(file_sha(path)))
    return digest.hexdigest()


def identity(video: Path, source: Path, config: dict) -> dict:
    """パスやmtimeで代用せず、コード・モデル・旧画像・入力の内容を固定する。"""
    code = [(str(p.relative_to(source)), p) for base in ('src', 'scripts')
            for p in (source / base).rglob('*.py')]
    assets = Path(config['asset_root'])
    bases = ('models', 'config') if config.get('cache_asset_files') else ('models', 'config', 'data')
    files = [(str(p.relative_to(assets)), p) for base in bases
             for p in (assets / base).rglob('*') if p.is_file()]
    files += [(str(Path(p).relative_to(assets)), Path(p))
              for p in config.get('cache_asset_files', [])]
    files += [(key, Path(config[key])) for key in ('b3_model', 'b3_table')]
    versions = package_versions()
    native = importlib.util.find_spec('puyo_core')
    if native is not None and native.origin:
        code.append(('puyo_core', Path(native.origin)))
    environment = {k: v for k, v in os.environ.items()
                   if k.startswith('PUYO_') and not k.startswith('PUYO_IMAGE_FREE')}
    return dict(version=CACHE_VERSION, code_sha=tree_sha(code), model_sha=tree_sha(files),
                video_sha=file_sha(video), versions=versions, environment=environment, device='cpu')


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    return versions


def cache_key(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def write_chunk(path: Path, record: dict) -> None:
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        cloudpickle.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class CachedOld:
    """先頭N件を検算してから利用し、次チャンク前に確定状態へ進める。"""

    def __init__(self, runtime: Any, directory: Path, key: str,
                 verify_frames: int = VERIFY_FRAMES) -> None:
        if verify_frames < 1 or verify_frames > CHUNK_FRAMES:
            raise ValueError('検算件数は1〜チャンク長')
        self.runtime, self.directory, self.key = runtime, directory, key
        directory.mkdir(parents=True, exist_ok=True)
        self.verify_frames = verify_frames
        self.start = -1
        self.rows: list[dict] = []
        self.cached: dict | None = None
        self.hits = self.misses = self.rejected = self.computed = 0

    def begin(self, index: int) -> None:
        self.start, self.rows, self.cached = index, [], None
        path = self.directory / f'{index:09d}.pkl'
        if path.exists():
            try:
                with path.open('rb') as stream:
                    value = cloudpickle.load(stream)
                if (value['key'] != self.key or value['start'] != index
                        or len(value['rows']) != CHUNK_FRAMES
                        or not isinstance(value['rows'], list)
                        or not hasattr(value['runtime'], 'step')):
                    raise ValueError('キャッシュ識別子/連続フレーム数不一致')
                self.cached = value
            except (OSError, ValueError, KeyError, EOFError, TypeError, pickle.UnpicklingError):
                path.unlink(missing_ok=True)
                self.rejected += 1
        self.misses += int(self.cached is None)

    def step(self, frame: Any, index: int, fps: float) -> dict:
        if self.start < 0 or index == self.start + CHUNK_FRAMES:
            self.begin(index)
        offset = index - self.start
        if offset != len(self.rows):
            raise ValueError('旧キャッシュのフレーム順序が不連続')
        if self.cached is None or offset < self.verify_frames:
            actual = self.runtime.step(frame, index, fps)
            self.computed += 1
            if self.cached is not None and actual != self.cached['rows'][offset]:
                (self.directory / f'{self.start:09d}.pkl').unlink(missing_ok=True)
                self.cached = None
                self.rejected += 1
                self.misses += 1
        else:
            actual = self.cached['rows'][offset]
        self.rows.append(actual)
        if offset + 1 == CHUNK_FRAMES:
            self.finish()
        return actual

    def finish(self) -> None:
        if self.cached is not None:
            self.runtime = self.cached['runtime']
            self.hits += 1
        else:
            write_chunk(self.directory / f'{self.start:09d}.pkl', dict(key=self.key,
                start=self.start, rows=self.rows, runtime=self.runtime))
        self.cached = None
