"""実更新署名で輸送し、設置失敗時にも元例外をfinallyで隠さない派生。"""
from __future__ import annotations
import ast
import inspect
from typing import Any
import continuation as C
import extension as X
import entry_boundary_v2 as E

ORIGINAL = X.continuation


def derived() -> Any:
    previous = ORIGINAL()
    values = dict(previous.__globals__, ENTRY=E)
    tree = ast.parse(inspect.getsource(C.extend))
    tree.body[0].body.insert(0, ast.parse('entry = None').body[0])
    counts, calls, assertions, saved = 0, 0, 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 76:
            node.value += X.COUNT
            counts += 1
        if not isinstance(node, (ast.Try, ast.FunctionDef)): continue
        for index, item in enumerate(list(node.body)):
            text = ast.unparse(item)
            if text == 'lease.perform(recovery, I.RESET, I.RESET / 60)':
                node.body[index] = ast.parse('entry = ENTRY.install(stack, pipe, state, lease, recovery, I.RESET)').body[0]
                calls += 1
            if text.startswith('checked = verify('):
                node.body.insert(index, ast.parse('assert entry.done and entry.error is None').body[0])
                assertions += 1
        if isinstance(node, ast.Try):
            for index, item in enumerate(list(node.finalbody)):
                if ast.unparse(item).startswith('data = dict('):
                    node.finalbody.insert(index+1, ast.parse("data['entry_boundary'] = None if entry is None else entry.rows").body[0])
                    saved += 1
    assert (counts,calls,assertions,saved)==(2,1,1,1), 'entry_original_anchors'
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), values)
    return values['extend']
