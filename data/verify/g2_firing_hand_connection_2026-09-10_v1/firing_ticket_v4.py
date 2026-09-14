"""既存NEXTの固定変換から最終updateを再生成し、実controller参照も照合する。"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from typing import Any
import firing_ticket_v3 as V3
from firing_fixed import declared

NEXT = V3.V2.ROOT/'scripts/next_enqueue_live_shadow_v1.py'
NEXT_SHA = 'e5ebff6827119c616319b4598fce1428c98643b621ff02e985736b43783d9237'


def expected(value: Any, controller: Any) -> bool:
    original = getattr(value, '__wrapped__', None)
    if type(original) is not FunctionType or value.__globals__.get('__next_live') is not controller:
        return False
    cv = V3.V2.code_value
    assert cv(original.__code__) == cv(V3.V2.expected()), 'firing_wrapped_original_changed'
    module = sys.modules[type(controller).__module__]
    assert Path(module.__file__).resolve() == NEXT and hashlib.sha256(NEXT.read_bytes()).hexdigest() == NEXT_SHA
    assert module._transformed.__globals__ is vars(module)
    assert cv(module._transformed.__code__) == cv(declared(NEXT, ('_transformed',)))
    copied = FunctionType(original.__code__, dict(original.__globals__), original.__name__, original.__defaults__, original.__closure__)
    marker = SimpleNamespace()
    generated = module._transformed(copied, marker)
    assert cv(value.__code__) == cv(generated.__code__), 'firing_final_NEXT_transform_changed'
    return True


def bound_update(pipe: Any) -> Any:
    pending, seen, matches = [type(pipe).update], set(), []
    while pending:
        value = pending.pop()
        if id(value) in seen: continue
        seen.add(id(value))
        assert len(seen) <= V3.MAX_WRAPPERS, 'firing_wrapper_limit'
        if inspect.ismethod(value): pending.append(value.__func__)
        elif type(value) in (tuple, list):
            pending.extend(v for v in value if inspect.isfunction(v) or inspect.ismethod(v))
        elif type(value) is FunctionType:
            controller = value.__globals__.get('__next_live')
            if value.__code__.co_name == 'update' and Path(value.__code__.co_filename) == V3.V2.SOURCE:
                if controller is not None and expected(value, controller): matches.append(value)
            wrapped = getattr(value, '__wrapped__', None)
            if wrapped is not None: pending.append(wrapped)
            pending.extend(cell.cell_contents for cell in (value.__closure__ or ())
                if inspect.isfunction(cell.cell_contents) or inspect.ismethod(cell.cell_contents)
                or type(cell.cell_contents) in (tuple, list))
    assert len(matches) == 1, 'firing_bound_final_update_missing_or_multiple'
    return matches[0]


installed = V3.contextmanager(FunctionType(V3.installed.__wrapped__.__code__,
    dict(vars(V3), bound_update=bound_update)))
