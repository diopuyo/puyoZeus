"""B3専用の排他的wall/スレッドCPU計時。無効時は時計を読まない。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter, thread_time
from typing import Any, Callable, Iterator

MS = 1000.0
_ACTIVE: ContextVar[Any] = ContextVar('b3_profile', default=None)


@contextmanager
def profile(enabled: bool = True) -> Iterator[dict[str, float]]:
    values: dict[str, float] = {}
    token = _ACTIVE.set((values, []) if enabled else None)
    try:
        yield values
    finally:
        _ACTIVE.reset(token)


@contextmanager
def span(name: str) -> Iterator[None]:
    active = _ACTIVE.get()
    if active is None:
        yield
        return
    values, stack = active
    entry = [perf_counter(), thread_time(), 0.0, 0.0]
    stack.append(entry)
    try:
        yield
    finally:
        wall, cpu = perf_counter() - entry[0], thread_time() - entry[1]
        stack.pop()
        for suffix, cost in (('_ms', wall-entry[2]), ('_cpu_ms', cpu-entry[3])):
            values[name+suffix] = values.get(name+suffix, 0.0) + cost*MS
        if stack:
            stack[-1][2] += wall
            stack[-1][3] += cpu


def timed(name: str) -> Callable:
    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            with span(name):
                return function(*args, **kwargs)
        return measured
    return decorate
