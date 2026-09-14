"""既存connection等のgeneric aliasを占有しない限定ローダー。"""
from __future__ import annotations
import importlib.util
import builtins
from pathlib import Path
import sys
from typing import Any

ROOT=Path(__file__).resolve().parent


def load(alias: str,name: str|Path,injection: dict[str,Any]|None=None) -> Any:
    path=ROOT/name
    if alias in sys.modules:
        module=sys.modules[alias]
        if Path(module.__file__).resolve()!=path.resolve(): raise RuntimeError('inflight_alias_collision')
        return module
    spec=importlib.util.spec_from_file_location(alias,path)
    module=importlib.util.module_from_spec(spec)
    if injection:
        original=builtins.__import__
        def importing(name: str,globals: Any=None,locals: Any=None,fromlist: Any=(),level: int=0) -> Any:
            return injection[name] if level==0 and name in injection else original(name,globals,locals,fromlist,level)
        module.__dict__['__builtins__']=dict(vars(builtins),__import__=importing)
    sys.modules[alias]=module
    spec.loader.exec_module(module)
    return module


GUARD=load('_g2_inflight_guard','quarantine.py')
CONNECTION=load('_g2_inflight_connection','connection.py')


def basis_connection() -> Any:
    root=ROOT.parent/'g2_reset_settled_basis_gate_2026-09-11_v1'
    core=load('_g2_settled_basis_core',root/'settled_basis_gate.py')
    gate=load('_g2_settled_basis_v2',root/'gate_v2.py',{'settled_basis_gate':core})
    return load('_g2_settled_basis_actual',root/'actual_connection.py',{'gate_v2':gate})
