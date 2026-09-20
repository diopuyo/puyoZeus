"""公開後比較の全値をDの原票に保ち、一行ずつ参照する。"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts.g3_context_rows import MAX_ROW_BYTES, digest, require


def encoded(row: dict) -> bytes:
    """既存JSONの値を維持し、表示用の空白だけ省く。"""
    require(type(row) is dict, 'postcommit_row_type')
    return json.dumps(row, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


class SavedRows(Sequence):
    """原append順を保持する。索引はoffset/長さ/SHAのみ、変更は拒否する。"""
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = path.open('xb')
        self.offsets: list[tuple[int, int, bytes]] = []
        self.latest: dict | None = None
        self.latest_index: int | None = None
        self.latest_hash: bytes | None = None
        self.failure: str | None = None
        self.exported: tuple[Path, str] | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def unchanged(self) -> None:
        """読取先による変更を再読込で消さない。"""
        require(self.latest is None or hashlib.sha256(encoded(self.latest)).digest()
                == self.latest_hash, 'postcommit_loaded_row_mutated')

    def append(self, row: dict) -> None:
        """保存成功後だけ件数を進め、途中書込は失敗として固定する。"""
        require(not self.stream.closed and self.failure is None and self.exported is None,
                'postcommit_append_closed_or_failed')
        try:
            self.unchanged()
            raw = encoded(row) + b'\n'
            require(len(raw) <= MAX_ROW_BYTES, 'postcommit_row_too_large')
            offset = self.stream.tell()
            require(self.stream.write(raw) == len(raw), 'postcommit_short_write')
            self.stream.flush()
            self.offsets.append((offset, len(raw), hashlib.sha256(raw).digest()))
            # 呼出元が保持するdictのaliasを所有履歴として残さない。
            self.latest = self.latest_index = self.latest_hash = None
        except BaseException as error:
            self.failure = repr(error)
            raise

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if type(index) is not int or not -len(self) <= index < len(self):
            raise IndexError(index)
        index %= len(self)
        self.unchanged()
        offset, size, expected = self.offsets[index]
        with self.path.open('rb') as stream:
            stream.seek(offset)
            raw = stream.read(size)
        require(len(raw) == size and hashlib.sha256(raw).digest() == expected,
                'postcommit_saved_row_changed')
        value = json.loads(raw)
        require(type(value) is dict, 'postcommit_saved_row_type')
        self.latest, self.latest_index = value, index
        self.latest_hash = hashlib.sha256(encoded(value)).digest()
        return value

    def __iter__(self) -> Iterator[dict]:
        count = len(self)
        for index in range(count):
            require(count == len(self), 'postcommit_iteration_changed')
            yield self[index]
        self.unchanged()
        expected = sum(size for _, size, _ in self.offsets)
        require(self.path.stat().st_size == expected, 'postcommit_unregistered_bytes')

    def export(self, path: Path) -> None:
        """原closeから既存JSON配列を排他保存する。失敗票は削除しない。"""
        require(self.failure is None and self.exported is None, 'postcommit_export_failed_or_duplicate')
        with path.open('xb') as stream:
            stream.write(b'[')
            for index, row in enumerate(self):
                stream.write((b',' if index else b'') + encoded(row))
            stream.write(b']\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.exported = (path.resolve(), digest(path))

    def saved(self, path: Path) -> SavedRows:
        """終了後の読取も実保存票の照合を必須とする。"""
        require(self.exported is not None and self.exported[0] == path.resolve(),
                'postcommit_export_missing')
        require(digest(path) == self.exported[1], 'postcommit_export_changed')
        return self

    def close(self) -> None:
        """元close失敗時もJSONLの完全行を残して所有fdを閉じる。"""
        if not self.stream.closed:
            try:
                self.stream.flush()
                os.fsync(self.stream.fileno())
            finally:
                self.stream.close()


class RowIndex(Mapping):
    """全dictではなく行番号だけを保持する元検査用Mapping。"""
    def __init__(self, rows: SavedRows, offsets: dict) -> None:
        self.rows, self.offsets = rows, offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __iter__(self) -> Iterator:
        return iter(self.offsets)

    def __getitem__(self, key: Any) -> dict:
        return self.rows[self.offsets[key][1]]


def indexed(original: Callable, rows: Any, key: Any, *, field: bool) -> Any:
    """元の重複判定をそのまま実行し、保存行だけ軽量索引へ変換する。"""
    if not isinstance(rows, SavedRows):
        return original(rows, key)
    pairs = [(row[key] if field else key(row), index) for index, row in enumerate(rows)]
    values = original(pairs, 0 if field else lambda pair: pair[0])
    return RowIndex(rows, values)
