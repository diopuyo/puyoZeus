"""前frameの両側採録完了後、次update本体より前に一回だけ原resetを委譲。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
from types import MethodType
from typing import Any, Callable

STRIDE, FPS = 2, 60
SIDES = ('1P', '2P')


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('reset_entry:' + reason)


def completed_metadata(state: Any, frame: int) -> None:
    sink = state['collector_metadata_sink']
    require(not sink.busy and not sink.closed and not sink.errors, 'metadata_not_ready')
    rows = sink.rows[-len(SIDES):]
    require(len(rows) == len(SIDES), 'metadata_missing')
    require([(r['frame_idx'], r['side']) for r in rows] == [(frame, s) for s in SIDES],
            'metadata_previous_frame_sides')
    require(all(r['time_sec'] == frame/FPS and 'returned' in r for r in rows), 'metadata_clock_or_return')


class Entry:
    def __init__(self, pipe: Any, state: Any, callback: Callable[[int, float], None], frame: int) -> None:
        self.pipe, self.state, self.callback, self.frame = pipe, state, callback, frame
        self.done, self.error = False, None
        self.rows: list[Any] = []

    def before(self, pipe: Any, frame: int, clock: float) -> None:
        if self.error is not None: raise self.error
        require(pipe is self.pipe, 'foreign_pipe')
        if self.done: return
        try:
            require(type(frame) is int and frame == self.frame, 'next_frame')
            require(type(clock) is float and clock == frame/FPS, 'next_clock')
            completed_metadata(self.state, frame-STRIDE)
            self.rows.append(dict(stage='before_original_update', frame=frame,
                metadata_frame=frame-STRIDE, metadata_sides=list(SIDES)))
            self.callback(frame, clock)
            self.done = True
            self.rows.append(dict(stage='reset_returned', frame=frame))
        except BaseException as exc:
            self.error = exc
            self.rows.append(dict(stage='failed_before_update', frame=frame, error=repr(exc)))
            raise


def install(stack: ExitStack, pipe: Any, state: Any, lease: Any,
            recovery: Any, frame: int) -> Entry:
    original = pipe.update
    require(inspect.ismethod(original) and original.__self__ is pipe, 'actual_update_method')
    signature = inspect.signature(original)
    names = tuple(signature.parameters)[:2]
    require(names in (('frame_idx', 'time_sec'), ('frame', 'clock')), 'actual_clock_parameters')
    entry = Entry(pipe, state, lambda f,t: lease.perform(recovery,f,t), frame)
    existed, previous = 'update' in vars(pipe), vars(pipe).get('update')
    def update(self: Any, *args: Any, **kwargs: Any) -> Any:
        values = signature.bind(*args, **kwargs).arguments
        entry.before(self, values[names[0]], values[names[1]])
        return original(*args, **kwargs)
    def restore() -> None:
        if existed: pipe.update = previous
        else: vars(pipe).pop('update', None)
    stack.callback(restore)
    pipe.update = MethodType(update, pipe)
    return entry
