"""試験時だけ認識内部の壁時計・CPU・遅いimportを分解する。"""
from __future__ import annotations

import builtins
import inspect
from time import perf_counter, thread_time, process_time
from typing import Any

from . import live_stage_timing as timing

IMPORT_THRESHOLD_MS = 5.
METHODS = frozenset({'detect', 'update', 'observe', 'read', 'read_both_boards',
    'read_board', 'read_board_hsv_only', 'classify_batch', 'classify', 'predict',
    'predict_batch', 'infer', 'run', 'step'})


def install(pipe: Any) -> None:
    """未束縛メソッドを包む。deepcopy後も呼出し先は複製先のselfになる。"""
    if getattr(pipe, '_obs_timing_installed', False):
        return
    pipe._obs_timing_installed = True
    from src import cnn_onnx
    if not getattr(cnn_onnx.session_for, '_obs_timing', False):
        cnn_onnx.session_for = timing.timed('recog.onnx_session_lookup')(cnn_onnx.session_for)
        cnn_onnx.session_for._obs_timing = True
    for name, method in inspect.getmembers(type(pipe), inspect.isfunction):
        if not name.startswith('__'):
            instrument_method(type(pipe), name, 'recog.pipeline.'+name)
    seen: set[int] = {id(pipe)}
    for name, owner in list(vars(pipe).items()):
        instrument_owner(owner, name, seen, 2)
    from .live_device_session import DeviceSession
    from .live_calibration import ColorWarmup
    for cls, names in ((DeviceSession, ('update', 'emit', 'restore_calibration_metadata')),
                       (ColorWarmup, ('observe', 'advance', '_save'))):
        for name in names:
            instrument_method(cls, name, 'recog.'+cls.__name__+'.'+name)


def instrument_owner(owner: Any, label: str, seen: set[int], depth: int) -> None:
    if id(owner) in seen or not type(owner).__module__.startswith('src.'):
        return
    seen.add(id(owner))
    for name in METHODS:
        instrument_method(type(owner), name, 'recog.'+type(owner).__name__+'.'+name)
    if depth and hasattr(owner, '__dict__'):
        for name, child in list(vars(owner).items()):
            instrument_owner(child, label+'.'+name, seen, depth-1)


def instrument_method(cls: type, name: str, label: str) -> None:
    method = inspect.getattr_static(cls, name, None)
    if inspect.isfunction(method) and not getattr(method, '_obs_timing', False):
        wrapped = timing.timed(label)(method)
        wrapped._obs_timing = True
        setattr(cls, name, wrapped)


def install_import_timing() -> None:
    """既存importをそのまま呼び、遅かったものだけ原票に残す。"""
    original = builtins.__import__
    if getattr(original, '_obs_timing', False):
        return

    def imported(name: str, *args: Any, **kwargs: Any) -> Any:
        row = timing.CURRENT.get()
        if row is None:
            return original(name, *args, **kwargs)
        start, cpu, process = perf_counter(), thread_time(), process_time()
        try:
            return original(name, *args, **kwargs)
        finally:
            elapsed = (perf_counter()-start)*timing.MS
            if elapsed >= IMPORT_THRESHOLD_MS:
                row['details'].setdefault('slow_imports', []).append(dict(
                    module=name, wall_ms=elapsed, cpu_ms=(thread_time()-cpu)*timing.MS,
                    process_cpu_ms=(process_time()-process)*timing.MS))
    imported._obs_timing = True
    builtins.__import__ = imported
