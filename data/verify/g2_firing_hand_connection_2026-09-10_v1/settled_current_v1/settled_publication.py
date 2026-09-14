"""原J cleanup/捕捉票を維持し、配置popと精算currentの発行理由を分ける。"""
from __future__ import annotations
import ast
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
from types import CodeType
from typing import Any

KIND = 'settled_history_current'
SOURCE = Path(__file__).resolve().parents[2]/'g2_history_current_publication_2026-09-09_v2/ticket.py'
SOURCE_SHA = 'c4b5e7ae1748764a8f75416d611329660b92a97e5985707bdcfd15e036dbb03c'


def settled_call(call: Any) -> bool:
    ticket = call.get('settlement_current_ticket')
    assert ticket is not None and ticket.current_published and ticket.settled
    assert call['prepared'] is None and call['consumed'] is False, 'settled_publication_fake_pop'
    assert ticket.binding is call['binding'] and ticket.scope == call['view'].scope
    proof, state = call['current_proof'], call['binding'].owner.state
    assert proof['kind'] == KIND and proof['settlement'] == ticket.settlement_proof
    assert ticket.origin.origin_id in state.consumed_ids and not state.debts
    return True


def issue_function(original: Any) -> Any:
    assert Path(inspect.getsourcefile(original)).resolve() == SOURCE
    data = SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SOURCE_SHA, 'settled_publication_source'
    code = compile(data, str(SOURCE), 'exec', dont_inherit=True)
    expected_code = next(c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == 'issue')
    assert original.__code__ == expected_code, 'settled_publication_original_code'
    assert original.__globals__ is vars(sys.modules[original.__module__]), 'settled_publication_original_globals'
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected = ast.dump(ast.parse("call['consumed']", mode='eval').body)
    class Rewrite(ast.NodeTransformer):
        count = 0
        def visit_Subscript(self, node: Any) -> Any:
            if ast.dump(node) == expected:
                self.count += 1
                return ast.copy_location(ast.parse('settled_call(call)', mode='eval').body, node)
            return self.generic_visit(node)
    rewrite = Rewrite()
    tree = ast.fix_missing_locations(rewrite.visit(tree))
    assert rewrite.count == 1, 'settled_issue_single_consume_test'
    scope = dict(original.__globals__, KIND=KIND, settled_call=settled_call)
    exec(compile(tree, inspect.getsourcefile(original), 'exec', dont_inherit=True), scope)
    return scope[original.__name__]


def install(stack: Any, state: Any, patch: Any) -> None:
    receiver = state['postcommit_current_receiver']
    module = type(receiver).complete.__globals__['T']
    original = module.issue
    candidate = issue_function(original)
    def issue(controller: Any, captured: Any, result: Any, scope: Any, serialize: Any) -> Any:
        value = captured.call.get('current_proof', {})
        method = candidate if value.get('kind') == KIND else original
        return method(controller, captured, result, scope, serialize)
    patch(stack, module, 'issue', issue)
