"""原reset25入力をempty76直後へ移す。実動画・実時刻の証明ではない。"""
from __future__ import annotations
from types import FunctionType, SimpleNamespace as N
from typing import Any

RESET, STRIDE = 34924, 2
ORIGINAL_RESET = 34798
SHIFT = RESET-ORIGINAL_RESET
POST = tuple(range(RESET, RESET+25*STRIDE, STRIDE))


def shifted(original: Any) -> Any:
    values = dict(vars(original))
    for name in ('RESET', 'DEADLINE', 'FIRST', 'LANDED', 'SECOND'):
        values[name] = getattr(original, name)+SHIFT
    values['FRAMES'] = POST
    values['QUIET'] = tuple(f+SHIFT for f in original.QUIET)
    values['geometry'] = FunctionType(original.geometry.__code__, values)
    values['install'] = FunctionType(original.install.__code__, values)
    return N(**values)
