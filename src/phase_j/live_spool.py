"""監査履歴をディスクへ逐次退避し、運転時間に依存するRAM保持を避ける。"""
from __future__ import annotations

from collections.abc import Iterator
import os
import pickle
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any

PROTOCOL = pickle.HIGHEST_PROTOCOL


def bounded_enabled() -> bool:
    """旧保持経路はOFF/ONの記録再生検証用に残す。"""
    return os.environ.get('PUYO_BOUNDED_HISTORY', '1') != '0'


class DiskRows:
    """append直後の行への追記を許し、直前一行だけRAMに残す。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open('w+b')
        self.pending: Any = None
        self.count = 0
        self.lock = RLock()

    def append(self, value: Any) -> None:
        with self.lock:
            if self.count:
                pickle.dump(self.pending, self.stream, protocol=PROTOCOL)
            self.pending = value
            self.count += 1

    def extend(self, values: Any) -> None:
        for value in values:
            self.append(value)

    def clear(self) -> None:
        with self.lock:
            self.stream.seek(0)
            self.stream.truncate()
            self.pending, self.count = None, 0

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Any]:
        with self.lock:
            self.stream.flush()
            count, pending = self.count, self.pending
        with self.path.open('rb') as stream:
            for _ in range(max(0, count-1)):
                yield pickle.load(stream)
        if count:
            yield pending

    def __getitem__(self, index: int | slice) -> Any:
        if index == -1 or index == self.count-1:
            if not self.count:
                raise IndexError(index)
            return self.pending
        if index == 0 and self.count:
            return next(iter(self))
        return list(self)[index]

    def close(self) -> None:
        if not self.stream.closed:
            if self.count:
                pickle.dump(self.pending, self.stream, protocol=PROTOCOL)
            self.stream.close()


class DiskMap:
    """SSEの重複キーをディスク索引で除き、全送出記録を保持する。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.connection.execute('CREATE TABLE IF NOT EXISTS rows (key INTEGER PRIMARY KEY, value BLOB)')
        self.connection.execute('DELETE FROM rows')

    def setdefault(self, key: int, value: Any) -> None:
        self.connection.execute('INSERT OR IGNORE INTO rows VALUES (?, ?)',
                                (key, pickle.dumps(value, protocol=PROTOCOL)))

    def __len__(self) -> int:
        return self.connection.execute('SELECT COUNT(*) FROM rows').fetchone()[0]

    def values(self) -> Iterator[Any]:
        for (raw,) in self.connection.execute('SELECT value FROM rows ORDER BY key'):
            yield pickle.loads(raw)

    def close(self) -> None:
        self.connection.close()


class DiskFIFO:
    """認識の未消費ツモを順序・件数とも保持し、RAMへ積み残さないFIFO。"""
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = path.open('w+b')
        self.reader = path.open('rb')
        self.count = 0

    def append(self, value: Any) -> None:
        pickle.dump(value, self.writer, protocol=PROTOCOL)
        self.count += 1

    def popleft(self) -> Any:
        if not self.count:
            raise IndexError('空の認識FIFOです')
        self.writer.flush()
        value = pickle.load(self.reader)
        self.count -= 1
        if not self.count:
            self.clear()
        return value

    def clear(self) -> None:
        self.writer.seek(0)
        self.writer.truncate()
        self.writer.flush()
        self.reader.seek(0)
        self.count = 0

    def __len__(self) -> int:
        return self.count

    def close(self) -> None:
        self.writer.close()
        self.reader.close()


def json_default(value: Any) -> Any:
    if isinstance(value, DiskRows):
        return list(value)
    raise TypeError(f'JSON化できない型: {type(value).__name__}')
