"""元bodyの構造一致でcurrentを合成。前段が再採番した行番号に依存しない。"""
from __future__ import annotations

import ast
import copy
from typing import Any
import history_ast as A

COUNTER_LINE, MEMORY_LINE = 8160, 7343
ROUTER = A.ROUTER


def contains_call(node: Any, name: str) -> bool:
    return any(isinstance(n, ast.Call) and ast.unparse(n.func) == ROUTER + '.' + name for n in ast.walk(node))


def add_current(tree: ast.Module) -> ast.Module:
    A.guards()
    result = copy.deepcopy(tree)
    step = A.unique(result, ast.FunctionDef, lambda n: n.name == '_step_side', 'current_step')
    A.require(contains_call(step, 'legacy_accounting') and not contains_call(step, 'current_counter'),
        'current_requires_history_once')
    original = ast.parse(A.SOURCE.read_bytes())
    memory = A.unique(original, ast.If, lambda n: n.lineno == MEMORY_LINE, 'original_memory')
    actual = A.unique(step, ast.If, lambda n: A.key(n) == A.key(memory), 'unchanged_memory')
    A.require(A.key(memory) == A.key(actual), 'current_memory_changed')
    old = A.unique(original, ast.Assign, lambda n: n.lineno == COUNTER_LINE, 'original_counter')
    counter = A.unique(step, ast.Assign, lambda n: A.key(n) == A.key(old), 'current_counter_assignment')
    counter.value = A.call('current_counter', [counter.value], counter.value)
    inference = A.unique(step, ast.If, lambda n: isinstance(n.test, ast.Call)
        and ast.unparse(n.test.func) == ROUTER + '.gate'
        and isinstance(n.test.args[0], ast.Constant) and n.test.args[0].value == 'infer', 'current_infer')
    A.require(inference in step.body, 'current_infer_not_top_level')
    gate = A.call('current_recovered', [], inference)
    # 新しい複製は実生成済みの位置情報を保持。原memory本文は完全同値。
    sync = ast.copy_location(ast.If(gate, [copy.deepcopy(actual)], []), inference)
    step.body.insert(step.body.index(inference) + 1, sync)
    return ast.fix_missing_locations(result)
