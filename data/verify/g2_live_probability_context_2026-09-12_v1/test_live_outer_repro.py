"""原外側verifyが正常な確率所有を拒否するCPU反例。実constructor資格の証明ではない。"""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent.parent / 'g2_history_publication_probe_runtime_2026-09-10_v13'
END = 34980


def original() -> Any:
    path = ROOT / 'repeated_connection.py'
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('verify', 'scope_status')]
    assert len(nodes) == 2
    def require(condition: Any, reason: str) -> None:
        if not condition:
            raise AssertionError(reason)
    ns = dict(Any=Any, K=N(require=require, FRAMES=(END,)))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns['verify']


def state(integer: bool) -> dict[str, Any]:
    binding = object() if integer else None
    factory = N(controller=N(history={'1P': binding} if integer else {}))
    guard = N(error=None, record=None, frame=END, binding=binding, factory=factory)
    evidence = dict(attempts=1, installed=True, closed=True, references_restored=True,
                    qualification=dict(qualified_restored=True), rows=[])
    return dict(repeated_firing_constructor=evidence, repeat_scope_guard=guard)


def test_original_integer_normal_control() -> None:
    assert original()(state(True))['same_scope_guard']['same_binding'] is True


def test_probability_normal_shape_rejected_by_old_outer() -> None:
    with pytest.raises(AssertionError, match='^same_scope_not_closed$'):
        original()(state(False))


@pytest.mark.parametrize('key', ['installed', 'closed', 'references_restored'])
def test_constructor_rejection_remains_prior(key: str) -> None:
    value = state(False)
    value['repeated_firing_constructor'][key] = False
    with pytest.raises(AssertionError, match='^repeated_constructor_or_restore$'):
        original()(value)
