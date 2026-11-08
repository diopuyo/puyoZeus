"""入力ループの進み具合だけを記録する、既定無効の子プロセス見張り。"""
from __future__ import annotations

import faulthandler
import os
from pathlib import Path
import threading
import time
from typing import Callable

HEARTBEAT_ENV = 'PUYO_CHILD_HEARTBEAT'
# 2026-11-06: 通常起動17〜40秒、負荷時約60秒に余裕を持たせる候補値。
STARTUP_STALL_SEC = 150.0
RUNNING_STALL_SEC = 90.0
# 親の回復より45秒早く、止まった場所を保存する。
STALL_DUMP_SEC = 45.0
POLL_SEC = 5.0
HEARTBEAT_WRITE_SEC = 1.0
MAX_STALL_DUMPS = 3  # 障害時にもディスク使用量をプロセスごとに制限する。
WATCHDOG_EXIT_CODE = 1
_dump_count = 0
_dump_lock = threading.Lock()


class InputHeartbeat:
    """画像・状態・ローカル変数は保存せず、単調時計とスタックだけ残す。"""

    def __init__(self, path: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self.path, self.clock = path, clock
        self.last = clock()
        self.written = float('-inf')
        self.dumped = False
        self.closed = threading.Event()
        self.lock = threading.Lock()

    def beat(self) -> None:
        with self.lock:
            self.last = self.clock()
            self.dumped = False
            if self.last - self.written < HEARTBEAT_WRITE_SEC:
                return
            stamp = self.last
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(str(stamp), encoding='ascii')
        temporary.replace(self.path)
        self.written = stamp

    def check(self) -> None:
        global _dump_count
        with self.lock, _dump_lock:
            if (self.dumped or _dump_count >= MAX_STALL_DUMPS
                    or self.clock() - self.last < STALL_DUMP_SEC):
                return
            self.dumped = True
            _dump_count += 1
            stamp = time.time_ns()
            path = self.path.parent / f'stall_stacks_{stamp}_{os.getpid()}.txt'
            with path.open('x', encoding='utf-8') as stream:
                faulthandler.dump_traceback(file=stream, all_threads=True)

    def run(self) -> None:
        while not self.closed.wait(POLL_SEC):
            try:
                self.check()
            except OSError:
                # 証拠の保存失敗でも親の心拍監視は続く。
                continue

    def close(self) -> None:
        self.closed.set()


def start_input_heartbeat() -> InputHeartbeat | None:
    from src import production_config
    value = os.environ.get(HEARTBEAT_ENV)
    if not production_config.DISTRIBUTION_CHILD_WATCHDOG or not value:
        return None
    heartbeat = InputHeartbeat(Path(value))
    threading.Thread(target=heartbeat.run, daemon=True, name='input-stall-dump').start()
    return heartbeat


class ProgressDeadline:
    """新しい子の最初の心拍と、その後の停止を親の単調時計で判定する。"""

    def __init__(self, started: float) -> None:
        self.changed = started
        self.value: float | None = None

    def check(self, path: Path, now: float) -> tuple[str, float] | None:
        import math
        try:
            value = float(path.read_text(encoding='ascii'))
            if math.isfinite(value) and (self.value is None or value > self.value):
                self.value, self.changed = value, now
        except (OSError, ValueError):
            pass
        elapsed = now - self.changed
        kind = 'startup_stall' if self.value is None else 'running_stall'
        limit = STARTUP_STALL_SEC if self.value is None else RUNNING_STALL_SEC
        return (kind, elapsed) if elapsed >= limit else None
