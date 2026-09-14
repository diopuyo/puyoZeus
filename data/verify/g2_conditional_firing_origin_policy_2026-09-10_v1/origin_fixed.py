"""既存sealed policyと採用simulation backendの帰属を固定する。"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import importlib
import hashlib
import sys

ROOT=Path(__file__).resolve().parent
VERIFY=ROOT.parent
PROJECTION=VERIFY/'g2_conditional_firing_projection_2026-09-10_v1'
POLICY=VERIFY/'g2_firing_policy_2026-09-10_v1'
EXPECTED={'firing_policy.py':'17fa4d5fcf92f1d86edabc1b708f17b143ec816c9ba61862f4e405027d6b5ad0',
    'firing_evidence.py':'59da32ff3d9df807e17f007e7315899b3b2f3cdade96d193ea9f2620fe52ab38',
    'projection_backend.py':'d394adfbf32b35684b02c013c5c1fbfe5f33186a56744f6ef4025f02c3d7a629',
    'projection_fixed.py':'f5dbac70fa7c919c3e02e7f687508f25c937bae446883d9451f52177c2de4937'}


def require(value: bool, reason: str) -> None:
    if not value: raise ValueError('conditional_origin_'+reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def backend() -> Any:
    for name,expected in EXPECTED.items():
        path=(PROJECTION if name.startswith('projection') else POLICY)/name
        require(sha(path)==expected,'fixed_source:'+name)
    previous=list(sys.path)
    try:
        sys.path.insert(0,str(PROJECTION))
        module=importlib.import_module('projection_backend')
        require(Path(module.__file__).resolve()==PROJECTION/'projection_backend.py','backend_alias')
        return module.verify()
    finally:
        sys.path[:]=previous


def original(base: Any) -> tuple[Any,Any]:
    arm=base.arm
    values=arm.__globals__
    require(Path(values['__file__']).resolve()==POLICY/'firing_policy.py','policy_globals')
    f,e=values['F'],values['E']
    require(Path(e.__file__).resolve()==POLICY/'firing_evidence.py','evidence_globals')
    backend()
    for name in ('__init__','arm','authorize'):
        method=getattr(base,name)
        require(method.__globals__ is values and f.code_value(method.__code__)==
            f.code_value(f.declared(POLICY/'firing_policy.py',('FiringMethods',name))),'sealed_code')
    underlying=base._base_policy
    require(Path(underlying.__init__.__globals__['__file__']).resolve()==f.PRODUCER,'base_policy')
    require(underlying.__init__.__globals__['S'] is base._inventory.S,'base_split')
    f.guards()
    return values,e
