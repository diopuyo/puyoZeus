"""既存AST生成prepareのcodeを保ち、後付け資格の参照だけを再束縛する。"""
from __future__ import annotations
from types import FunctionType
from typing import Any


def rebound(original: Any, predicate: Any) -> Any:
    assert 'compatible_current' in original.__code__.co_names,'combined_missing_current_access'
    values = dict(original.__globals__,compatible_current=predicate)
    result = FunctionType(original.__code__,values,original.__name__,original.__defaults__,original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    assert result.__code__ is original.__code__ and result.__closure__ is original.__closure__
    assert all(result.__globals__[k] is v for k,v in original.__globals__.items() if k!='compatible_current')
    return result


def install(stack: Any, control: Any, patch: Any) -> None:
    lifecycle = type(control).prepared.__globals__['V1'].L
    original = lifecycle.prepare
    assert 'closed_origins' in original.__globals__,'combined_expected_origin_transform'
    assert original.__globals__ is not vars(lifecycle),'combined_expected_copied_globals'
    assert original.__globals__['compatible_current'] is not lifecycle.compatible_current
    patch(stack,lifecycle,'prepare',rebound(original,lifecycle.compatible_current))
