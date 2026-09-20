"""元policy/Sの固定内容と使用する関数の帰属を検査する。"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import CodeType
from typing import Any

ROOT = Path(__file__).resolve().parent
PRODUCER = ROOT.parent / 'g2_machine_inventory_producer_2026-09-09_v1/producer.py'
SPLIT = ROOT.parent / 'g2_current_accounting_split_2026-09-08_v1/split_contract.py'
DIAG = ROOT.parent / 'g2_inventory_producer_diagnosis_2026-09-09_v1/probe.py'
FIXED = {PRODUCER: 'd8e796ae0ab50c10710a1455fc72c28f04319b8392b4e4e4b698aa51b40b6dee',
    SPLIT: 'd9c90ce7edf12051f1def730c641354fe0be1df00c6cf6adada66191462660c1',
    DIAG: 'f3ea5e870d344436739e3ea706b39fcec58112d8fe79b1cc08c0739e4e1dc244'}
CODE_FIELDS = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals', 'co_stacksize',
    'co_flags', 'co_code', 'co_names', 'co_varnames', 'co_filename', 'co_name', 'co_qualname',
    'co_firstlineno', 'co_linetable', 'co_exceptiontable', 'co_freevars', 'co_cellvars')


def require(value: bool, reason: str) -> None:
    if not value: raise ValueError('firing_' + reason)


def guards() -> dict[str, str]:
    values = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in FIXED}
    require(all(values[str(path)] == expected for path, expected in FIXED.items()), 'fixed_source')
    return values


def code_value(value: Any) -> Any:
    if type(value) is not CodeType: return value
    return (tuple(getattr(value, field) for field in CODE_FIELDS), tuple(code_value(c) for c in value.co_consts))


def declared(path: Path, names: tuple[str, ...]) -> Any:
    code = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
    for name in names:
        code = next(c for c in code.co_consts if type(c) is CodeType and c.co_name == name)
    return code


def verify(p: Any) -> None:
    guards()
    module = sys.modules.get(p.BoundPolicy.__module__)
    require(module is not None and Path(module.__file__).resolve() == PRODUCER, 'policy_module')
    require(p.BoundPolicy is module.BoundPolicy and p.S is module.S, 'policy_parts')
    require(Path(p.S.__file__).resolve() == SPLIT, 'split_module')
    for name in ('__init__', 'arm', 'authorize'):
        actual = getattr(p.BoundPolicy, name)
        require(actual.__globals__ is vars(module)
            and code_value(actual.__code__) == code_value(declared(PRODUCER, ('BoundPolicy', name))), 'policy_code')
    for name in ('digest', 'encoded'):
        require(getattr(p, name) is getattr(module, name), 'producer_function')


def load() -> Any:
    guards()
    alias = '_firing_original_inventory'
    require(not any(name in sys.modules for name in
        (alias, '_machine_inventory_diagnostic', '_machine_inventory_split')), 'module_collision')
    spec = importlib.util.spec_from_file_location(alias, PRODUCER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    verify(module)
    return module
