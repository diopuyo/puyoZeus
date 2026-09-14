"""原外側二検査を維持し、最内の整数所有述語だけを確率所有へ接続する。"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import sys
from types import CellType, CodeType, FunctionType
from typing import Any
import probability_owner as P

BASE = Path(__file__).resolve().parent.parent
PINS = (
    ('g2_conditional_live_adapter_2026-09-10_v3/adapter.py',
     'c060eacc3358d93d9c59272799ac1a4dc22f375d0edde3bb53d67b48be91e9eb', 'checked'),
    ('g2_conditional_live_adapter_2026-09-10_v1/adapter.py',
     'c5a3bf6bd92923262dd3cbc842e5413639d956cac0a4d7fd0890218b8ecf32e6', 'original_verify'),
    ('g2_history_publication_probe_runtime_2026-09-10_v13/repeated_connection.py',
     'eeb838318c197b881f6ec5a5a57ec16bb1f3df28ce71d2a1904290e722b0121c', None),
)


def codes(code: CodeType) -> list[CodeType]:
    result = [code]
    for child in code.co_consts:
        if isinstance(child, CodeType):
            result.extend(codes(child))
    return result


def pinned(function: Any, index: int) -> bytes:
    relative, digest, _ = PINS[index]
    path = BASE / relative
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, 'outer_source_changed'
    assert Path(function.__code__.co_filename).resolve() == path, 'outer_source_path'
    expected = [c for c in codes(compile(raw, str(path), 'exec', dont_inherit=True))
                if c.co_name == 'verify']
    assert len(expected) == 1 and function.__code__ == expected[0], 'outer_code_changed'
    return raw


def authority(state: dict[str, Any]) -> bool:
    guard = state['repeat_scope_guard']
    lease, factory = guard.reset_lease, guard.factory
    assert guard is lease.guard, 'outer_guard_identity'
    mode = state['probabilistic_tracking_mode']
    assert mode.state is state, 'outer_state_identity'
    assert mode.connection.binding is None or mode.native is not None, 'probability_binding_without_activation'
    original = sys.modules['retired_completion']
    assert Path(original.__file__).resolve() == BASE / 'g2_empty_tail_reset_integration_2026-09-11_v1/retired_completion.py'
    old = lease.archive.binding
    return P.retired_owner(original, state, lease, factory, old, old.owner.state,
                           old.owner.state, (lease.empty_evidence.frame, P.SIDE))


def leaf(function: Any) -> Any:
    raw = pinned(function, 2)
    assert function.__closure__ is None, 'outer_leaf_closure'
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
    assert not any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(node)), 'outer_global_write'
    sites = [n for n in ast.walk(node) if isinstance(n, ast.Subscript)
             and ast.unparse(n) == "guard['same_binding']"]
    assert len(sites) == 1, 'outer_predicate_site'
    class Replace(ast.NodeTransformer):
        def visit_Subscript(self, value: ast.Subscript) -> Any:
            if value is sites[0]:
                return ast.copy_location(ast.parse('_probability_outer_authority(state)', mode='eval').body, value)
            return self.generic_visit(value)
    node = Replace().visit(node)
    namespace = dict(function.__globals__, _probability_outer_authority=authority)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(BASE / PINS[2][0]), 'exec'), namespace)
    adapted = namespace['verify']
    def verify(state: dict[str, Any]) -> Any:
        if not P.tracking_selected(state):
            return function(state)
        result = adapted(state)
        return result | dict(ownership_basis='probabilistic_retired', probability_owner_verified=True)
    return verify


def clone(function: Any, index: int = 0) -> Any:
    if index == 2:
        return leaf(function)
    pinned(function, index)
    name = PINS[index][2]
    names, cells = function.__code__.co_freevars, function.__closure__
    assert cells is not None and names.count(name) == 1, 'outer_wrapper_closure'
    position = names.index(name)
    child = clone(cells[position].cell_contents, index + 1)
    replaced = tuple(CellType(child) if i == position else cell for i, cell in enumerate(cells))
    result = FunctionType(function.__code__, function.__globals__, function.__name__,
                          function.__defaults__, replaced)
    result.__kwdefaults__, result.__annotations__ = function.__kwdefaults__, function.__annotations__
    return result


def install(stack: Any, main: Any) -> None:
    repeated = main.__globals__['REPEAT']
    original, raw_status = repeated.verify, repeated.scope_status
    adapted = clone(original)
    def restore() -> None:
        assert repeated.verify is adapted and repeated.scope_status is raw_status, 'outer_restore_ownership'
        repeated.verify = original
    stack.callback(restore)
    repeated.verify = adapted
