"""承認済みupdate wrapper連鎖内の原本体をbindし、同code＋globalsを要求する。"""
from __future__ import annotations
from contextlib import contextmanager
import inspect
from pathlib import Path
import sys
from types import FunctionType
from typing import Any, Iterator
import firing_ticket_v2 as V2

MAX_WRAPPERS = 200


def bound_update(pipe: Any) -> Any:
    pending, seen, matches = [type(pipe).update], set(), []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        assert len(seen) <= MAX_WRAPPERS, 'firing_wrapper_limit'
        if inspect.ismethod(value):
            pending.append(value.__func__)
        elif type(value) in (tuple, list):
            pending.extend(v for v in value if inspect.isfunction(v) or inspect.ismethod(v))
        elif type(value) is FunctionType:
            if value.__code__.co_name == 'update' and Path(value.__code__.co_filename) == V2.SOURCE:
                assert V2.code_value(value.__code__) == V2.code_value(V2.expected()), 'firing_bound_code_changed'
                matches.append(value)
            wrapped = getattr(value, '__wrapped__', None)
            if wrapped is not None: pending.append(wrapped)
            pending.extend(cell.cell_contents for cell in (value.__closure__ or ())
                if inspect.isfunction(cell.cell_contents) or inspect.ismethod(cell.cell_contents)
                or type(cell.cell_contents) in (tuple, list))
    assert len(matches) == 1, 'firing_bound_update_missing_or_multiple'
    return matches[0]


@contextmanager
def installed(pipe: Any, evidence: dict[str, Any]) -> Iterator[None]:
    actual, old = bound_update(pipe), V2.OLD.qualified
    def qualified(frame: Any, factory: Any, environment: Any) -> Any:
        assert sys._getframe(1) is frame, 'firing_front_frame_identity'
        caller = frame.f_back
        assert caller.f_code is actual.__code__ and caller.f_globals is actual.__globals__, 'firing_front_code'
        assert frame.f_locals['self'] is caller.f_locals['self'] is pipe, 'firing_front_pipe'
        assert Path(environment['__file__']).resolve() == V2.SOURCE, 'firing_front_module'
        value = V2.ORIGINAL(frame, factory, environment)
        value.source.update(caller_code_verified=True, caller_globals_verified=True, immediate_frame_verified=True)
        return value
    V2.OLD.qualified = qualified
    evidence.update(bound_original_update=True, globals_same_as_formula=None)
    try:
        yield
    finally:
        V2.OLD.qualified = old
        evidence['qualified_restored'] = V2.OLD.qualified is old
