"""精算済みorigin履歴は削除せず、同Bindingの閉鎖票を条件に継続する。"""
from __future__ import annotations
import ast
import inspect
import textwrap
from types import FunctionType, SimpleNamespace
from typing import Any

TOKEN = ast.dump(ast.parse('view.added[0]', mode='eval').body)
ORIGINS = ast.dump(ast.parse('not state.origins', mode='eval').body)


def closed_origins(state: Any, binding: Any) -> bool:
    if not state.origins:
        return True
    ticket = getattr(binding, 'firing_ticket', None)
    return (ticket is not None and ticket.consumed and getattr(ticket, 'settled', False)
        and ticket.binding is binding and ticket.origin in state.origins and not state.debts
        and all(origin.origin_id in state.consumed_ids for origin in state.origins))


def rewritten(original: Any, *, token: bool) -> Any:
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    class Replace(ast.NodeTransformer):
        token_count, origin_count = 0, 0
        def visit_Subscript(self, node: Any) -> Any:
            if token and ast.dump(node) == TOKEN:
                self.token_count += 1
                return ast.copy_location(ast.parse("call['prepared']['proof']['new_token']", mode='eval').body, node)
            return self.generic_visit(node)
        def visit_UnaryOp(self, node: Any) -> Any:
            if ast.dump(node) == ORIGINS:
                self.origin_count += 1
                return ast.copy_location(ast.parse('closed_origins(state, binding)', mode='eval').body, node)
            return self.generic_visit(node)
    patch = Replace()
    tree = ast.fix_missing_locations(patch.visit(tree))
    assert patch.token_count == int(token)
    assert patch.origin_count == int(original.__name__ in ('prepare', 'committed'))
    scope = dict(original.__globals__, closed_origins=closed_origins)
    exec(compile(tree, inspect.getsourcefile(original), 'exec', dont_inherit=True), scope)
    return scope[original.__name__]


def token_function(original: Any) -> Any:
    return rewritten(original, token=True)


def configure(stack: Any, base: Any, control: Any) -> Any:
    v1 = type(control).prepared.__globals__['V1']
    lifecycle = v1.prepared.__globals__['L']
    base.patch(stack, lifecycle, 'prepare', rewritten(lifecycle.prepare, token=False))
    values = dict(vars(base), token_function=token_function)
    calls = FunctionType(base.install_calls.__code__, values)
    return SimpleNamespace(**(values | {'install_calls': calls}))
