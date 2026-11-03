"""判定通知とは独立したフレーム別計装。各processのspoolへ逐次保存する。"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import gc
import logging
import os
from pathlib import Path
from threading import Lock
from time import perf_counter, thread_time, process_time
from typing import Any, Callable

MS = 1000.0
STAGES = ('capture', 'normalize', 'color_correction', 'screen_verify', 'continuity',
          'board_recognition', 'b3_score', 'b3_ledger', 'b3_warning', 'b3_formula',
          'b3_observer', 'b3_queues', 'evaluation', 'publication', 'notice_queue')
CURRENT: ContextVar[dict | None] = ContextVar('live_stage_timing', default=None)
SINKS: dict[str, Any] = {}
LOCK = Lock()
GC_TOTAL_MS = 0.
GC_STARTED = 0.


def observe_gc(phase: str, info: dict) -> None:
    global GC_TOTAL_MS, GC_STARTED
    if phase == 'start':
        GC_STARTED = perf_counter()
    else:
        GC_TOTAL_MS += (perf_counter()-GC_STARTED)*MS


@contextmanager
def locked(name: str, lock: Any) -> Any:
    measure(name, lock.acquire)
    try:
        yield
    finally:
        lock.release()


def install(stack: ExitStack, directory: Path | None, role: str) -> None:
    if directory is None or os.environ.get('PUYO_LIVE_STAGE_TIMING') != '1':
        return
    directory.mkdir(parents=True, exist_ok=True)
    from .live_obs_audit import open_audit
    stream = open_audit(directory / ('frame_timing_'+role+'.jsonl'))
    SINKS[role] = stream
    if os.environ.get('PUYO_OBS_DEEP_TIMING') == '1':
        from .live_obs_recognition_timing import install_import_timing
        install_import_timing()
    if observe_gc not in gc.callbacks:
        gc.callbacks.append(observe_gc)
    stack.callback(close, role, stream)


def close(role: str, stream: Any) -> None:
    with LOCK:
        if SINKS.get(role) is stream:
            SINKS.pop(role)
        stream.close()
        if not SINKS and observe_gc in gc.callbacks:
            gc.callbacks.remove(observe_gc)


def begin(role: str, captured_at: float, frame: int | None = None, t_sec: float | None = None) -> None:
    if role not in SINKS:
        return
    CURRENT.set(dict(role=role, frame=frame, t_sec=t_sec, captured_at=captured_at,
                     started_at=perf_counter(), cpu_started=thread_time(), gc_started_ms=GC_TOTAL_MS,
                     stages={}, details={}))


def identify(frame: Any, epoch: Any = None) -> None:
    row = CURRENT.get()
    if row is None:
        begin('recognition', frame.captured_at, frame.index, frame.media_sec)
        row = CURRENT.get()
    if row is not None:
        row.update(frame=frame.index, t_sec=frame.media_sec, epoch=epoch)


def detail(**values: Any) -> None:
    row = CURRENT.get()
    if row is not None:
        row['details'].update(values)


def measure(name: str, function: Callable, *args: Any, **kwargs: Any) -> Any:
    row = CURRENT.get()
    if row is None:
        return function(*args, **kwargs)
    started, cpu, process = perf_counter(), thread_time(), process_time()
    try:
        return function(*args, **kwargs)
    finally:
        values = row['stages'].setdefault(name, dict(wall_ms=0., cpu_ms=0., calls=0))
        values['wall_ms'] += (perf_counter()-started)*MS
        values['cpu_ms'] += (thread_time()-cpu)*MS
        values['process_cpu_ms'] = values.get('process_cpu_ms', 0.)+(process_time()-process)*MS
        values['calls'] += 1


def timed(name: str) -> Callable:
    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return measure(name, function, *args, **kwargs)
        return wrapper
    return decorate


def finish(**values: Any) -> None:
    row = CURRENT.get()
    CURRENT.set(None)
    if row is None:
        return
    row.update(values, finished_at=perf_counter())
    row['wall_ms'] = (row['finished_at']-row['started_at'])*MS
    row['cpu_ms'] = (thread_time()-row.pop('cpu_started'))*MS
    row['gc_process_overlap_ms'] = GC_TOTAL_MS-row.pop('gc_started_ms')
    if 'evaluation_wall_ms' in values:
        row['stages']['evaluation'] = dict(wall_ms=values['evaluation_wall_ms'],
            cpu_ms=values['evaluation_cpu_ms'], calls=1)
    for name in STAGES:
        row['stages'].setdefault(name, dict(wall_ms=0., cpu_ms=0., calls=0))
    with LOCK:
        stream = SINKS.get(row['role'])
        if stream is not None:
            try:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            except OSError:
                # 計装のディスク障害で判定処理を変更しない。
                SINKS.pop(row['role'], None)
                logging.getLogger(__name__).exception('段別計装の保存を停止しました')


def publication(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(owner: Any, row: dict, *args: Any, **kwargs: Any) -> Any:
        token = CURRENT.set(None)
        begin('publication', row['captured_at'], row['frame'], row.get('t_sec'))
        try:
            return measure('publication', function, owner, row, *args, **kwargs)
        finally:
            finish(game=row.get('game'), evaluated_at=row.get('evaluated_at'))
            CURRENT.reset(token)
    return wrapper


def b3_result(frame: int, stamp: float, captured: float, costs: dict, **values: Any) -> None:
    """非同期計算は完了時に元frameへ結び付け、受信frameへ時間を誤帰属させない。"""
    token = CURRENT.set(None)
    begin('b3_result', captured, frame, stamp)
    try:
        detail(b3_timings_ms=costs, **values)
        finish()
    finally:
        CURRENT.reset(token)
