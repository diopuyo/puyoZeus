"""原samecall本体とcalleeの実code/global束縛を固定ソースへ照合する。"""
from __future__ import annotations
import dataclasses
import hashlib
import json
from pathlib import Path
from types import CodeType, FunctionType
from typing import Any

SOURCE = Path(__file__).resolve().parent.parent/'g2_empty_tail_finalizer_2026-09-10_v1/empty_completion.py'
SOURCE_SHA = '953469a284cab8376258cd07abebfb07fe3db45e76e37c7a936db49118dd6a85'
FUNCTIONS = ('verify','source','related','active_call')


def verify(module: Any) -> None:
    raw = SOURCE.read_bytes()
    assert Path(module.__file__).resolve()==SOURCE and hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    code = compile(raw,str(SOURCE),'exec',dont_inherit=True)
    fixed = {c.co_name:c for c in code.co_consts if isinstance(c,CodeType)}
    for name in FUNCTIONS:
        function = getattr(module,name)
        assert type(function) is FunctionType and function.__globals__ is vars(module), 'samecall_global_identity'
        assert function.__code__==fixed[name] and function.__closure__ is None, 'samecall_callee_code'
    assert module.asdict is dataclasses.asdict and module.sha256 is hashlib.sha256
    assert module.json is json, 'samecall_standard_helpers'
