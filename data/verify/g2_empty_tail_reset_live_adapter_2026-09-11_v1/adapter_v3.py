"""新adapterを固有aliasで読む。既存のgeneric adapter依存を遮蔽しない。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from types import FunctionType
from typing import Any

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_live_empty_adapter_base',ROOT/'adapter.py')
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)


def dependencies(stack: Any, old: Any) -> Any:
    before = list(sys.path)
    stack.callback(sys.path.__setitem__,slice(None),before)
    sys.path.insert(0,str(A.VERIFY/'g2_empty_tail_desync_trigger_2026-09-11_v1'))
    return A.dependencies(stack,old)


configured = FunctionType(A.configured.__code__,dict(vars(A),dependencies=dependencies))
