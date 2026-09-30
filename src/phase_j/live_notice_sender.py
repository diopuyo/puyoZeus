"""認識processの送信を評価側の詰まりから切り離す非同期送信口 (B20、既定OFF)。

背景: 評価workerが全応手探索で数秒止まると mp.Queue(30) が満杯になり、認識threadの
`queue.put` が1〜11秒ブロックして取得側が捨て frame を出す (B18 の残り約6.6%)。
本送信口は通知の内容・順序・件数を一切変えず、認識threadと mp.Queue の間に
有界の局所バッファを置くだけである。局所バッファが満杯になった時は従来どおり
ブロックし(背圧維持)、無限にメモリを使わない。
"""
from __future__ import annotations

from collections import deque
import os
from queue import Full
from threading import Condition, Event, Thread
import time
from typing import Any

ENV_FLAG = 'PUYO_ASYNC_NOTICE_QUEUE'
LOCAL_CAPACITY = 1800      # 30Hz通知で約60秒分。これ以上は背圧に戻す
POLL_SEC = 0.05
JOIN_SEC = 10.0
STALL_SEC = 0.15           # mp.Queue.put がこれ以上ブロックしたら「評価側の詰まり」と数える
SENDER_NAME = 'notice-sender'


def enabled() -> bool:
    """環境変数で有効化。spawn子processへ親の環境がそのまま継承される。"""
    return os.environ.get(ENV_FLAG) == '1'


class AsyncNoticeSender:
    """put(message) のみを持つ mp.Queue 互換の送信口。単一writer・FIFO。"""

    def __init__(self, queue: Any, cancel: Any, capacity: int = LOCAL_CAPACITY) -> None:
        self.queue, self.cancel, self.capacity = queue, cancel, capacity
        self.buffer: deque = deque()
        self.condition = Condition()
        self.closing = Event()
        self.backlog_max = 0
        self.producer_blocked_sec = 0.0
        self.sender_stall_sec = 0.0
        self.sender_stalls: list[float] = []
        self.sent = 0
        self.dropped_on_cancel = 0
        self.thread = Thread(target=self._run, name=SENDER_NAME, daemon=True)
        self.thread.start()

    def put(self, message: Any) -> None:
        """認識thread側。局所バッファが満杯の時だけ待つ。"""
        with self.condition:
            waited = time.perf_counter() if len(self.buffer) >= self.capacity else None
            while len(self.buffer) >= self.capacity and not self.cancel.is_set():
                self.condition.wait(POLL_SEC)
            if waited is not None:
                self.producer_blocked_sec += time.perf_counter()-waited
            self.buffer.append(message)
            self.backlog_max = max(self.backlog_max, len(self.buffer))
            self.condition.notify_all()

    def _next(self) -> Any:
        with self.condition:
            while not self.buffer:
                if self.closing.is_set() or self.cancel.is_set():
                    return None
                self.condition.wait(POLL_SEC)
            return self.buffer[0]

    def _deliver(self, message: Any) -> bool:
        started = time.perf_counter()
        while not self.cancel.is_set():
            try:
                self.queue.put(message, timeout=POLL_SEC)
                break
            except Full:
                continue
        else:
            return False
        elapsed = time.perf_counter()-started
        if elapsed >= STALL_SEC:
            self.sender_stall_sec += elapsed
            self.sender_stalls.append(elapsed)
        return True

    def _run(self) -> None:
        while True:
            message = self._next()
            if message is None:
                break
            delivered = self._deliver(message)
            with self.condition:
                self.buffer.popleft()
                self.sent += delivered
                self.condition.notify_all()
            if not delivered:
                break
        with self.condition:
            self.dropped_on_cancel = len(self.buffer)
            self.buffer.clear()
            self.condition.notify_all()

    def close(self) -> None:
        """未送信分を全て送ってから止める。cancel時は残りを捨てる(評価側が終了中)。"""
        self.closing.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(JOIN_SEC)

    def stats(self) -> dict[str, Any]:
        return dict(capacity=self.capacity, backlog_max=self.backlog_max, sent=self.sent,
                    producer_blocked_sec=self.producer_blocked_sec,
                    sender_stall_sec=self.sender_stall_sec,
                    sender_stall_count=len(self.sender_stalls),
                    sender_stall_max_sec=max(self.sender_stalls, default=0.0),
                    dropped_on_cancel=self.dropped_on_cancel)


def wrap_queue(queue: Any, cancel: Any) -> Any:
    """既定OFFでは元のqueueをそのまま返す(bit-identical)。"""
    return AsyncNoticeSender(queue, cancel) if enabled() else queue


def close_queue(queue: Any) -> None:
    if isinstance(queue, AsyncNoticeSender):
        queue.close()


def queue_stats(queue: Any) -> dict[str, Any] | None:
    return queue.stats() if isinstance(queue, AsyncNoticeSender) else None
