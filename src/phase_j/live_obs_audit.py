"""試験の監査bytesを順序どおり保存し、ディスク待ちを計算所有者から分離する。"""
from __future__ import annotations
from queue import Queue, Full, Empty
from threading import Thread, Event, RLock
from dataclasses import dataclass, field
from typing import Any
import os
from pathlib import Path
from time import monotonic

CAPACITY = 256
CHECK_SEC = .1
BUFFER_BYTES = 1024*1024
FLUSH_SEC = 1.
NO_WRITE = object()


@dataclass
class Operation:
    name: str
    args: tuple
    done: Event = field(default_factory=Event)
    result: Any = None
    error: BaseException | None = None


def open_audit(path: Path, mode: str = 'w', buffering: int = 1) -> Any:
    enabled = os.environ.get('PUYO_OBS_FAST_AUDIT') == '1'
    stream = path.open(mode, encoding='utf-8', buffering=BUFFER_BYTES if enabled else buffering)
    return AsyncAudit(stream) if enabled else stream


def open_spool(path: Path) -> Any:
    enabled = os.environ.get('PUYO_OBS_FAST_AUDIT') == '1'
    stream = path.open('w+b', buffering=BUFFER_BYTES if enabled else -1)
    return AsyncSpool(stream) if enabled else stream


class AsyncAudit:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.queue: Queue = Queue(CAPACITY)
        self.error: BaseException | None = None
        self.closed = False
        self.thread = Thread(target=self.run, name='obs-audit-writer', daemon=True)
        self.thread.start()

    def run(self) -> None:
        flush_at = monotonic()+FLUSH_SEC
        try:
            while True:
                try:
                    value = self.queue.get(timeout=max(CHECK_SEC, flush_at-monotonic()))
                except Empty:
                    value = NO_WRITE
                if value is None:
                    break
                if isinstance(value, Operation):
                    self.perform(value)
                elif value is not NO_WRITE:
                    self.stream.write(value)
                if monotonic() >= flush_at:
                    self.stream.flush()
                    flush_at = monotonic()+FLUSH_SEC
        except BaseException as error:
            self.error = error
        finally:
            try:
                self.stream.close()
            except BaseException as error:
                self.error = error

    def write(self, value: Any) -> None:
        if self.closed:
            raise ValueError('閉じた監査stream')
        while True:
            if self.error is not None:
                raise OSError('監査書込みに失敗') from self.error
            try:
                self.queue.put(value, timeout=CHECK_SEC)
                return
            except Full:
                continue

    def perform(self, operation: Operation) -> None:
        try:
            operation.result = getattr(self.stream, operation.name)(*operation.args)
        except BaseException as error:
            operation.error = error
            raise
        finally:
            operation.done.set()

    def operation(self, name: str, *args: Any) -> Any:
        command = Operation(name, args)
        self.write(command)
        while not command.done.wait(CHECK_SEC):
            if self.error is not None:
                raise OSError('監査操作の完了に失敗') from self.error
        if command.error is not None:
            raise OSError('監査操作に失敗') from command.error
        return command.result

    def flush(self) -> None:
        self.operation('flush')

    def tell(self) -> int:
        """先行書込みを完了させ、巻き戻し用の確定位置を返す。"""
        return self.operation('tell')

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.operation('seek', offset, whence)

    def truncate(self, size: int | None = None) -> int:
        return self.operation('truncate', size)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.write(None)
            self.thread.join()
            if self.error is not None:
                raise OSError('監査保存の完了に失敗') from self.error
        finally:
            self.closed = True


class AsyncSpool(AsyncAudit):
    """バイナリ履歴の論理位置を保持し、確定位置の照会でwriterを待たない。"""

    def __init__(self, stream: Any) -> None:
        self.position = stream.tell()
        self.position_lock = RLock()
        super().__init__(stream)

    def write(self, value: Any) -> None:
        with self.position_lock:
            if isinstance(value, (bytes, bytearray, memoryview)):
                value = bytes(value)
            super().write(value)
            if isinstance(value, bytes):
                self.position += len(value)

    def tell(self) -> int:
        with self.position_lock:
            if self.closed:
                raise ValueError('閉じた監査stream')
            if self.error is not None:
                raise OSError('監査書込みに失敗') from self.error
            return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        with self.position_lock:
            self.position = super().seek(offset, whence)
            return self.position
