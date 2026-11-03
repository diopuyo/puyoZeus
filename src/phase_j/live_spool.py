"""監査履歴をディスクへ逐次退避し、運転時間に依存するRAM保持を避ける。"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
import os
from io import BytesIO
import pickle
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any
from . import live_stage_timing as timing

PROTOCOL = pickle.HIGHEST_PROTOCOL
FIFO_BUFFER_BYTES = 1024*1024


def bounded_enabled() -> bool:
    """旧保持経路はOFF/ONの記録再生検証用に残す。"""
    return os.environ.get('PUYO_BOUNDED_HISTORY', '1') != '0'


class DiskRows:
    """append直後の行への追記を許し、直前一行だけRAMに残す。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        from .live_obs_audit import open_spool
        self.stream = open_spool(path)
        self.pending: Any = None
        self.count = 0
        self.lock = RLock()

    def append(self, value: Any) -> None:
        with timing.locked('spool_lock_wait', self.lock):
            if self.count:
                timing.measure('spool_write', pickle.dump, self.pending, self.stream, protocol=PROTOCOL)
            self.pending = value
            self.count += 1

    def extend(self, values: Any) -> None:
        for value in values:
            self.append(value)

    def clear(self) -> None:
        with timing.locked('spool_lock_wait', self.lock):
            self.stream.seek(0)
            timing.measure('spool_truncate', self.stream.truncate)
            self.pending, self.count = None, 0

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Any]:
        with self.lock:
            self.stream.flush()
            count, pending = self.count, self.pending
        # 終了集計も有界の先読みでSMBの小さなreadを減らし、同じ件数だけ復元する。
        from .live_obs_audit import BUFFER_BYTES
        buffering = BUFFER_BYTES if os.environ.get('PUYO_OBS_FAST_AUDIT') == '1' else -1
        with self.path.open('rb', buffering=buffering) as stream:
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
        self.writer = None
        self.map_lock = RLock()
        if os.environ.get('PUYO_OBS_FAST_AUDIT') == '1':
            from .live_obs_audit import AsyncAudit
            self.writer = AsyncAudit(MapStream(self.connection, self.map_lock))

    def setdefault(self, key: int, value: Any) -> None:
        item = (key, pickle.dumps(value, protocol=PROTOCOL))
        if self.writer is None:
            self.connection.execute('INSERT OR IGNORE INTO rows VALUES (?, ?)', item)
        else:
            self.writer.write(item)

    def __len__(self) -> int:
        with self.read_guard():
            return self.connection.execute('SELECT COUNT(*) FROM rows').fetchone()[0]

    def values(self) -> Iterator[Any]:
        with self.read_guard():
            for (raw,) in self.connection.execute('SELECT value FROM rows ORDER BY key'):
                yield pickle.loads(raw)

    def read_guard(self) -> Any:
        if self.writer is not None:
            self.writer.flush()
            return self.map_lock
        return nullcontext()

    def close(self) -> None:
        self.writer.close() if self.writer is not None else self.connection.close()


class MapStream:
    """同じSQLite行をFIFOで挿入し、送信スレッドの外でまとめてcommitする。"""

    def __init__(self, connection: sqlite3.Connection, lock: Any) -> None:
        self.connection, self.lock = connection, lock
        connection.execute('BEGIN')

    def write(self, item: tuple[int, bytes]) -> None:
        with self.lock:
            self.connection.execute('INSERT OR IGNORE INTO rows VALUES (?, ?)', item)

    def flush(self) -> None:
        with self.lock:
            self.connection.commit()
            self.connection.execute('BEGIN')

    def close(self) -> None:
        with self.lock:
            try:
                self.connection.commit()
            finally:
                self.connection.close()


class DiskFIFO:
    """認識の未消費ツモを順序・件数とも保持し、RAMへ積み残さないFIFO。"""
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = path.open('w+b')
        self.reader = path.open('rb')
        self.count = 0
        self.fast = os.environ.get('PUYO_OBS_FAST_AUDIT') == '1'
        self.buffer: BytesIO | None = BytesIO() if self.fast else None
        self.read_position = 0

    @timing.timed('fifo_append')
    def append(self, value: Any) -> None:
        if self.buffer is not None:
            payload = pickle.dumps(value, protocol=PROTOCOL)
            if self.buffer.tell()+len(payload) <= FIFO_BUFFER_BYTES:
                self.buffer.write(payload)
                self.count += 1
                return
            self.spill()
        pickle.dump(value, self.writer, protocol=PROTOCOL)
        self.count += 1

    def spill(self) -> None:
        """上限を超えたら、既読位置と全bytesを保って従来のディスクFIFOへ戻す。"""
        self.writer.write(self.buffer.getbuffer())
        self.reader.seek(self.read_position)
        self.buffer.close()
        self.buffer = None

    @timing.timed('fifo_popleft')
    def popleft(self) -> Any:
        if not self.count:
            raise IndexError('空の認識FIFOです')
        if self.buffer is not None:
            self.buffer.seek(self.read_position)
            value = pickle.load(self.buffer)
            self.read_position = self.buffer.tell()
            self.buffer.seek(0, os.SEEK_END)
        else:
            self.writer.flush()
            value = pickle.load(self.reader)
        self.count -= 1
        if not self.count:
            self.clear()
        return value

    @timing.timed('fifo_clear')
    def clear(self) -> None:
        if self.buffer is None:
            self.writer.seek(0)
            self.writer.truncate()
            self.writer.flush()
            self.reader.seek(0)
        else:
            self.buffer.close()
        self.buffer = BytesIO() if self.fast else None
        self.read_position = 0
        self.count = 0

    def __len__(self) -> int:
        return self.count

    def close(self) -> None:
        if self.buffer is not None:
            self.writer.write(self.buffer.getbuffer())
            self.buffer.close()
            self.buffer = None
        self.writer.close()
        self.reader.close()


def json_default(value: Any) -> Any:
    if isinstance(value, DiskRows):
        return list(value)
    raise TypeError(f'JSON化できない型: {type(value).__name__}')
