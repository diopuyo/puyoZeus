"""原bindingの純粋な来歴検査だけを再利用し、モデルimportを認識環境へ入れない。"""
from __future__ import annotations
import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import CodeType,FunctionType
from typing import Any

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'g2_provisional_context_capture_2026-09-09_v1/binding.py'
SOURCE_SHA='401c1e1a5c5f4965c20750a9c47ecd210deab083f958f3ef4b29bf5c3db1aa2d'
NAMES=frozenset(('ContextHold','ContextFault','require','encoded','digest','exact','integer',
    '_identity','_update','_generation'))


def original_namespace() -> dict[str,Any]:
    raw=SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA,'pure_contract_source_changed'
    nodes=[node for node in ast.parse(raw).body
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in NAMES]
    assert {node.name for node in nodes}==NAMES,'pure_contract_definition_coverage'
    namespace=dict(__name__=__name__,Any=Any,json=json,math=math,hashlib=hashlib,
        SIDES=('1P','2P'),SCHEMA='provisional-current-context/v1')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)
    codes=compile(raw,str(SOURCE),'exec').co_consts
    for code in codes:
        if isinstance(code,CodeType) and code.co_name in NAMES and code.co_name[0].islower() or (
            isinstance(code,CodeType) and code.co_name in ('_identity','_update','_generation')):
            previous=namespace[code.co_name]
            value=FunctionType(code,namespace,code.co_name,previous.__defaults__)
            value.__kwdefaults__=previous.__kwdefaults__
            namespace[code.co_name]=value
    return namespace


ORIGINAL=original_namespace()
_identity=ORIGINAL['_identity']
_update=ORIGINAL['_update']
_generation=ORIGINAL['_generation']
digest=ORIGINAL['digest']


@dataclass(frozen=True)
class RowInputs:
    """型付きモデル入力の代用ではなく、子で原_inputsを実行するための未変換原票。"""
    source_json: str


def _inputs(row: dict[str,Any]) -> RowInputs:
    return RowInputs(ORIGINAL['encoded'](row))
