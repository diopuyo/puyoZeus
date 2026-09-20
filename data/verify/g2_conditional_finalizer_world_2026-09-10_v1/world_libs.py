"""既存ライブラリと型付きPB読取を再用。実Bindingを合成しない。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
ATTACH = VERIFY / 'g2_conditional_constructor_attach_2026-09-10_v1'
CURRENT = VERIFY / 'g2_hidden_current_candidate_2026-09-10_v1/conditional_current.py'
TYPED = VERIFY / 'g2_conditional_firing_origin_policy_2026-09-10_v1/settlement_saved_review_v1/check_saved.py'
SCOPE = VERIFY / 'g2_same_scope_stop_guard_2026-09-10_v1/scope_stop_evidence.py'


def load(alias: str, path: Path) -> Any:
    if alias in sys.modules:
        value = sys.modules[alias]
        assert Path(value.__file__).resolve() == path
        return value
    spec = importlib.util.spec_from_file_location(alias, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[alias] = value
    spec.loader.exec_module(value)
    return value


def libraries() -> Any:
    if 'conditional_current' not in sys.modules:
        with ExitStack() as stack:
            previous = list(sys.path)
            stack.callback(sys.path.__setitem__, slice(None), previous)
            sys.path.insert(0, str(ATTACH))
            reused = load('_world_attach_assets', ATTACH / 'run_cpu.py')
            reused.modules(stack)
    current = sys.modules['conditional_current']
    assert Path(current.__file__).resolve() == CURRENT
    tree = ast.parse(TYPED.read_bytes())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name in ('typed_fields', 'typed_pb')]
    assert len(selected) == 2
    scope: dict[str, Any] = dict(Any=Any)
    exec(compile(ast.Module(selected, []), str(TYPED), 'exec'), scope)
    return SimpleNamespace(C=current, P=current.P, typed_fields=scope['typed_fields'],
        typed_pb=scope['typed_pb'], E=load('_world_original_scope', SCOPE))
