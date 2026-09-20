"""互換対象の固定sourceと元関数を確認する。coldな実constructor前にsrcは読まない。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType
from typing import Any
import hashlib
import sys

ROOT=Path(__file__).resolve().parent
VERIFY=ROOT.parent
REVISION=VERIFY/'g2_settled_current_revision_2026-09-10_v1'
SETTLEMENT=VERIFY/'g2_conditional_firing_settlement_2026-09-10_v1'
ORIGIN=VERIFY/'g2_conditional_firing_origin_policy_2026-09-10_v1'
EXPECTED={
    REVISION/'settled_fixed.py':'79873f641dce793b76038ab2037ee337758fdf9ebce311946754a8dd48cc828e',
    REVISION/'settled_evidence.py':'d14c4e965429987a1f760e5953168532c00613fd781bf13381329bbff67614c1',
    REVISION/'recover.py':'2716befad18d05583517b109fc4d5e15fcec3bb7dde92daf5668ce5b91eda5c6',
    SETTLEMENT/'conditional_settlement.py':'d9a84cb95d71e566ac4e981f8e100757469fc676e82e71fd980682cf0256501f',
    SETTLEMENT/'settlement_provenance.py':'4898362254aef2d1182dd694b5e66bdc9e8be84145b9b17c6a00c1ca37da602f',
    ORIGIN/'origin_policy_v2.py':'e1054ee7cd8e5320896c7d6ed943355d7be9b5466613cc7b26974c1c6e78f8d9'}


def require(value: bool, reason: str) -> None:
    if not value: raise ValueError('settled_compat_'+reason)


def guards() -> dict[str,str]:
    result={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in EXPECTED}
    require(all(result[str(path)]==value for path,value in EXPECTED.items()),'source_changed')
    return result


def module(value: Any, path: Path) -> None:
    require(Path(value.__file__).resolve()==path and sys.modules.get(value.__name__) is value,'module_identity')


def function(value: Any, owner: Any, names: tuple[str,...], fixed: Any) -> None:
    require(type(value) is FunctionType and value.__globals__ is vars(owner),'function_globals')
    expected=fixed.declared(Path(owner.__file__).resolve(),names)
    require(fixed.code_value(value.__code__)==fixed.code_value(expected),'function_code')

