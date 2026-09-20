"""実complete closureから原captureを解決する。別factory・同名別codeは不可。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
from types import CodeType, FunctionType, MethodType
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'g2_hidden_current_candidate_2026-09-10_v1/hidden_current_connection.py'
SOURCE_SHA = '5c8525d3063a57b208f072fd1b29c2a089f00d018eedd7c2079088bf687b076a'
MAX_FUNCTIONS = 128


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('conditional_call:' + reason)


def codes(code: CodeType) -> list[CodeType]:
    result = [code]
    for item in code.co_consts:
        if isinstance(item, CodeType):
            result.extend(codes(item))
    return result


def expected(name: str) -> CodeType:
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA, 'capture_source')
    items = codes(compile(SOURCE.read_text(encoding='utf-8'), str(SOURCE), 'exec', dont_inherit=True))
    found = [item for item in items if item.co_name == name]
    require(len(found) == 1, 'fixed_function_name')
    return found[0]


def functions(root: Any) -> list[Any]:
    pending, result, seen = [root], [], set()
    while pending:
        value = pending.pop()
        if isinstance(value, MethodType):
            value = value.__func__
        if not isinstance(value, FunctionType) or id(value) in seen:
            continue
        seen.add(id(value)); result.append(value)
        require(len(seen) <= MAX_FUNCTIONS, 'wrapper_limit')
        pending.append(getattr(value, '__wrapped__', None))
        pending.extend(cell.cell_contents for cell in (value.__closure__ or ()))
    return result


def select(factory: Any) -> tuple[Any, Any, Any]:
    control, journal = factory.controller, factory.provider.journal
    fixed = expected('completed')
    found = [fn for fn in functions(journal.complete_step) if fn.__code__ == fixed]
    require(len(found) == 1, 'original_completed_missing_or_duplicate')
    function = found[0]
    closure = dict(zip(function.__code__.co_freevars,
        (cell.cell_contents for cell in function.__closure__), strict=True))
    require(closure['control'] is control and closure['journal'] is journal, 'actual_factory')
    module = sys.modules.get(function.__globals__['__name__'])
    require(module is not None and vars(module) is function.__globals__, 'original_globals')
    require(Path(module.__file__).resolve() == SOURCE, 'original_module')
    return module, control, journal


def original(factory: Any) -> tuple[Any, Any, Any, Any]:
    module, control, journal = select(factory)
    capture = module.capture
    require(isinstance(capture, FunctionType) and capture.__code__ == expected('capture')
        and capture.__globals__ is vars(module) and capture.__closure__ is None, 'original_capture')
    return module, control, journal, capture
