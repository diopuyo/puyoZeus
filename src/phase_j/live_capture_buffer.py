"""実カメラの取得を続け、最新3枚から認識に必要な連続frameを選ぶ。"""
from __future__ import annotations

from collections import deque
from threading import Condition, Event, Thread
import time
from typing import Any, Callable

from .live_degrade import EVENT_BACKLOG_SLOTS

CAPTURE_HZ = 30
CAPACITY = EVENT_BACKLOG_SLOTS+1
READ_TIMEOUT_SEC = 0.5
JOIN_TIMEOUT_SEC = 2.0


class BufferedCapture:
    """deviceの生成・read・releaseは取得threadだけが所有し、検証/較正は呼出側に残す。"""
    def __init__(self, factory: Callable, arguments: tuple, priority: Any,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.factory, self.arguments, self.priority, self.clock = factory, arguments, priority, clock
        self.frames: deque = deque()
        self.condition, self.stop, self.started = Condition(), Event(), Event()
        self.properties: dict = {}
        self.origin, self.duration = 0.0, float('inf')
        self.dropped = self.dropped_before = self.previous_dropped = 0
        self.last_captured_at: float | None = None
        self.opened = self.failed = self.finished = False
        self.error: BaseException | None = None
        self.thread: Thread | None = None

    def set(self, key: int, value: float) -> bool:
        if self.thread is not None:
            return False
        self.properties[key] = value
        return True

    def isOpened(self) -> bool:
        if self.thread is None:
            self.thread = Thread(target=self._run, name='capture-input', daemon=True)
            self.thread.start()
        self.started.wait(READ_TIMEOUT_SEC)
        if self.error is not None:
            raise RuntimeError('取得threadが失敗しました') from self.error
        return self.opened

    def _offer(self, captured: float, image: Any) -> None:
        with self.condition:
            while len(self.frames) >= CAPACITY:
                self.frames.popleft()
                self.dropped += 1
            self.frames.append((captured, image))
            self.failed = False
            self.condition.notify_all()

    def _run(self) -> None:
        capture = None
        try:
            capture = self.factory(*self.arguments)
            for key, value in self.properties.items():
                capture.set(key, value)
            self.opened = capture.isOpened()
            self.started.set()
            while self.opened and not self.stop.is_set() and self.clock()-self.origin < self.duration:
                captured = self.clock()
                ok, image = capture.read()
                if ok and image is not None:
                    self._offer(captured, image)
                else:
                    with self.condition:
                        self.dropped += len(self.frames)
                        self.frames.clear()
                        self.failed = True
                        self.condition.notify_all()
                self.stop.wait(max(0, captured+1/CAPTURE_HZ-self.clock()))
        except BaseException as error:
            self.error = error
        finally:
            try:
                if capture is not None:
                    capture.release()
            except BaseException as error:
                self.error = error
            with self.condition:
                self.finished = True
                self.started.set()
                self.condition.notify_all()

    def read(self) -> tuple[bool, Any]:
        with self.condition:
            self.condition.wait_for(lambda: bool(self.frames) or self.failed or self.finished,
                                    timeout=READ_TIMEOUT_SEC)
            if not self.frames:
                return False, None
            indices = [round((stamp-self.origin)*CAPTURE_HZ) for stamp, _ in self.frames]
            target = self.priority.select(indices[0], indices[-1], 1, CAPTURE_HZ)
            while indices[0] < target:
                self.frames.popleft()
                indices.pop(0)
                self.dropped += 1
            self.last_captured_at, image = self.frames.popleft()
            self.dropped_before = self.dropped-self.previous_dropped
            self.previous_dropped = self.dropped
            return True, image

    def release(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(JOIN_TIMEOUT_SEC)
            if self.thread.is_alive():
                raise RuntimeError('カメラreadが終了しません。認識processを終了します')
        with self.condition:
            self.dropped += len(self.frames)
            self.frames.clear()
        if self.error is not None:
            raise RuntimeError('取得threadが失敗しました') from self.error
