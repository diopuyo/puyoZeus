"""原live_identityのbinding参照だけを認証済み退役ownerへ向ける。元historyは変更しない。"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import sys
from types import CodeType
from typing import Any, Callable
import probability_owner as P

BASE = Path(__file__).resolve().parent.parent
SOURCE = BASE / 'g2_conditional_finalizer_compatibility_2026-09-10_v1/compat.py'
SOURCE_SHA = '405c4ac2c1ba59d55173ed4768c44ad8e2aa14dd8e6c41e9676ef6d2f521fb26'
LOOKUP = "binding = controller.history[rows[-1]['scope']['side']]"


def rewrite(function: Any, binding: Callable[..., Any]) -> Any:
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA, 'stage_identity_source'
    assert Path(function.__globals__['__file__']).resolve() == SOURCE, 'stage_identity_path'
    compiled = compile(raw, str(SOURCE), 'exec', dont_inherit=True)
    expected = next(c for c in compiled.co_consts if isinstance(c, CodeType) and c.co_name == 'live_identity')
    assert function.__code__ == expected and function.__closure__ is None, 'stage_identity_code'
    assert function.__globals__['live_identity'] is function, 'stage_identity_original_global'
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'live_identity')
    assert not any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(node)), 'stage_identity_global_write'
    sites = [n for n in ast.walk(node) if isinstance(n, ast.Assign) and ast.unparse(n) == LOOKUP]
    assert len(sites) == 1, 'stage_live_binding_site'
    sites[0].value = ast.parse('_retired_stage_binding(factory, rows)', mode='eval').body
    before = dict(function.__globals__)
    namespace = dict(before, _retired_stage_binding=binding)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), 'exec'), namespace)
    assert all(function.__globals__[k] is v for k, v in before.items()) and set(function.__globals__) == set(before)
    assert set(namespace) == set(before) | {'_retired_stage_binding'}
    assert all(namespace[k] is v for k, v in before.items() if k != 'live_identity')
    return namespace['live_identity']


def adapted(function: Any, state: Any, lease: Any, original_q: Any) -> Any:
    original = sys.modules['retired_completion']
    assert Path(original.__file__).resolve() == BASE / 'g2_empty_tail_reset_integration_2026-09-11_v1/retired_completion.py'
    assert original.E is original_q.E, 'stage_retired_module'
    def binding(factory: Any, rows: Any) -> Any:
        old = lease.archive.binding
        final = rows[-1]['scope']
        assert final['side'] == P.SIDE and final['frame_idx'] == lease.empty_evidence.frame, 'stage_retired_final_frame'
        P.retired_owner(original, state, lease, factory, old, old.owner.state, old.owner.state,
                        (final['frame_idx'], final['side']))
        return old
    return rewrite(function, binding)
