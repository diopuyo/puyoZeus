"""私有samecallの本体・比較callee・受領型を固定ソースへ照合する。"""
from __future__ import annotations
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import CodeType, FunctionType
from typing import Any

SOURCE = Path(__file__).resolve().parent.parent/'g2_private_suffix_live_adapter_2026-09-10_v1/completion.py'
SOURCE_SHA = '24e403ddc4faf2dc43628a834af09ac6463ede34aa580c647df6486c29f5f260'
FUNCTIONS = ('verify','related','encoded','capture')
METHODS = ('__init__','__eq__','__setattr__','__delattr__')


def receipt_type(actual: Any, module: Any) -> None:
    alias = '_private_receipt_auth_reference'
    assert alias not in sys.modules
    spec = importlib.util.spec_from_file_location(alias,SOURCE)
    expected = importlib.util.module_from_spec(spec)
    sys.modules[alias] = expected
    try:
        spec.loader.exec_module(expected)
        reference = expected.Receipt
        assert type(actual) is type and actual.__bases__==(object,)
        assert actual.__module__==module.__name__ and actual.__name__=='Receipt'
        assert dataclasses.is_dataclass(actual) and actual.__dataclass_params__.frozen
        assert actual.__annotations__==reference.__annotations__
        assert tuple(actual.__dataclass_fields__)==tuple(reference.__dataclass_fields__)
        for name in METHODS:
            function,original = getattr(actual,name),getattr(reference,name)
            assert type(function) is FunctionType and function.__code__==original.__code__, 'receipt_code'
            assert function.__globals__ is vars(module), 'receipt_globals'
            for cell,other in zip(function.__closure__ or (),original.__closure__ or (),strict=True):
                assert cell.cell_contents is (actual if other.cell_contents is reference else other.cell_contents), 'receipt_closure'
    finally:
        sys.modules.pop(alias,None)


def verify(module: Any) -> None:
    raw = SOURCE.read_bytes()
    assert Path(module.__file__).resolve()==SOURCE and hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    fixed = {c.co_name:c for c in compile(raw,str(SOURCE),'exec',dont_inherit=True).co_consts if isinstance(c,CodeType)}
    for name in FUNCTIONS:
        value = getattr(module,name)
        assert type(value) is FunctionType and value.__globals__ is vars(module),'private_callee_globals'
        assert value.__code__==fixed[name] and value.__closure__ is None,'private_callee_code'
    assert module.asdict is dataclasses.asdict and module.hashlib is hashlib and module.json is json,'private_helpers'
    receipt_type(module.Receipt,module)
