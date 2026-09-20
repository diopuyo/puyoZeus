"""旧facadeのconnectだけを差替え、必要な新globalsは専用dispatchへ閉じ込める。"""
from __future__ import annotations
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any
import collector_connector_v2 as BASE
import raw_upgrade_v2 as RAW

INTEGRATED = BASE.INTEGRATED
DISPATCH = '_g2_joint_connector_dispatch'


def connect(stack: Any, state: dict) -> Any:
    call = FunctionType(BASE.connect.__code__, dict(vars(BASE), RAW=RAW))
    return call(stack, state)


def forwarded_connect(stack: Any, state: dict) -> Any:
    return __import__('sys').modules['_g2_joint_connector_dispatch'].connect(stack, state)


def install(stack: Any) -> None:
    original = INTEGRATED.OLD
    facade = N(**(vars(original) | dict(connect=forwarded_connect)))
    assert {k for k in vars(original) if getattr(facade, k) is not getattr(original, k)} == {'connect'}
    assert DISPATCH not in sys.modules, 'joint_dispatch_occupied'
    sys.modules[DISPATCH] = sys.modules[__name__]
    def restore() -> None:
        if INTEGRATED.OLD is not facade or sys.modules.get(DISPATCH) is not sys.modules[__name__]:
            raise ValueError('joint_connector_replaced')
        INTEGRATED.OLD = original
        del sys.modules[DISPATCH]
    stack.callback(restore)
    INTEGRATED.OLD = facade
