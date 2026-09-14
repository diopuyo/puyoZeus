"""原consumer冒頭の資格拒否を通し、Client単体では見えないmodule APIを検査する。"""
from __future__ import annotations
import ast
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
JOINT = ROOT.parent / 'g2_joint_collector_runtime_2026-09-12_v1'
sys.path[:0] = [str(JOINT), str(ROOT.parent / 'g2_joint_ledger_engine_2026-09-12_v1')]
import start_client as C


def original_create() -> Any:
    path = JOINT / 'runtime_session.py'
    tree = ast.parse(path.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create')
    namespace = dict(Any=Any, CLIENT=C)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['create']


def test_original_session_stack_guard(tmp_path: Path) -> None:
    state = dict(joint_producer_capture=N(identity=dict(source_id='artificial', run_id='artificial')),
                 output=tmp_path, joint_capture_stack=object())
    with pytest.raises(ValueError, match='joint_session_stack'):
        original_create()(object(), dict(state=state))
    assert not list(tmp_path.iterdir())


def test_client_and_worker_compatibility() -> None:
    assert C.T is C.C.T
    assert issubclass(C.Client, C.C.Client)
    assert C.Client.__init__.__code__ is C.C.Client.__init__.__code__
    assert C.worker_path() == ROOT / 'start_worker.py'
