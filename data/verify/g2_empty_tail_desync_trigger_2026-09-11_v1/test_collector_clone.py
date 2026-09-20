"""元driveのASTと既存複製方法でclosure依存を検出する短いCPU検査。"""
from __future__ import annotations
import ast
from pathlib import Path
from types import FunctionType
from typing import Any
import collector_connection_v2 as C

SOURCE = Path(__file__).resolve().parent.parent/'g2_hidden_two_hand_candidate_2026-09-10_v1/run_prefix.py'


def test_original_drive_cloning_contract() -> None:
    tree = ast.parse(SOURCE.read_bytes())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='drive')
    values: dict[str,Any] = dict(Any=Any)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),values)
    result = C.drive(values['drive'],values,None)
    assert not result.__code__.co_freevars
    cloned = FunctionType(result.__code__,values)
    assert cloned.__code__ is result.__code__
    assert 'collector_connection_v2' in result.__code__.co_consts
