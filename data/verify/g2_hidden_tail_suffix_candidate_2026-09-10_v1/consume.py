"""元tail消費の空FIFO条件だけを、投票時suffixの完全保持条件へ派生する。"""
from __future__ import annotations
import ast
import hashlib
import inspect
import marshal
from pathlib import Path
from types import CodeType
from typing import Any, Callable
import tail_suffix_provenance as SP

ORIGINAL_SHA = 'b22902aaaa2618dbf5642128e36f435e0679f99e0caccb1d01ff895598904328'
OLD_CONDITION = 'len(view.queue) == 0'


def same_refs(left: Any, right: Any) -> bool:
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


def suffix_preserved(control: Any, call: Any, votes: Any, values: Any) -> bool:
    """pop自体は行わず、元Jのpop後所有列と消費前の不変票を照合する。"""
    view, binding = call['view'], call['binding']
    refs = getattr(votes, 'queue_refs', None)
    tokens = getattr(votes, 'queue_tokens', None)
    if not (type(refs) is tuple and type(tokens) is tuple and len(refs) > 1
        and len(refs) == len(tokens) and same_refs(refs, view.refs)
        and tokens == view.tokens and len(set(tokens)) == len(tokens)
        and getattr(votes, 'queue_scope', None) == view.scope == binding.scope
        and votes.queue is view.queue and votes.head is refs[0]
        and votes.token == tokens[0] == binding.next_token
        and votes.token not in binding.consumed_tokens
        and same_refs(view.queue, refs[1:])):
        return False
    actual = control.provider.journal.fifo.entries.get((id(values['self']), values['side']))
    owned = (type(actual) is dict and values['side'] == view.scope[-1]
        and actual['queue'] is view.queue and tuple(actual['tokens']) == tokens[1:]
        and same_refs(actual['refs'], refs[1:]))
    held = getattr(binding, 'hidden_tail_successor', None)
    if not owned or type(held) is not SP.Successor:
        return False
    if SP.content(getattr(votes, 'successor_proof', None)) != held.sealed:
        return False
    row = control.provider.link.current(view)
    def require(ok: bool, reason: str) -> None:
        if not ok:
            raise ValueError(reason)
    SP.check(control, binding, votes.item, view, row, held, require)
    return True


def derive(original: Callable[..., Any]) -> Callable[..., Any]:
    """凍結元関数の一比較のみ置換。元関数・元globalsは変更しない。"""
    path = Path(inspect.getsourcefile(original))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_SHA, 'tail_consume_source'
    assert original.__name__ == 'consumed' and original.__closure__ is None
    declared = compile(path.read_bytes(), original.__code__.co_filename, 'exec', dont_inherit=True)
    expected = [node for node in declared.co_consts if isinstance(node, CodeType)
        and node.co_name == 'consumed']
    assert len(expected) == 1 and marshal.dumps(expected[0]) == marshal.dumps(original.__code__), 'tail_consume_code'
    tree = ast.parse(inspect.getsource(original))
    matches = [node for node in ast.walk(tree)
        if isinstance(node, ast.Compare) and ast.unparse(node) == OLD_CONDITION]
    assert len(matches) == 1, 'tail_consume_unique_condition'
    target = matches[0]
    class Replace(ast.NodeTransformer):
        def visit_Compare(self, node: ast.Compare) -> Any:
            if node is target:
                return ast.copy_location(ast.parse(
                    '_suffix_preserved(control, call, votes, values)', mode='eval').body, node)
            return self.generic_visit(node)
    updated = ast.fix_missing_locations(Replace().visit(tree))
    namespace = dict(original.__globals__, _suffix_preserved=suffix_preserved)
    exec(compile(updated, str(Path(__file__).resolve()) + ':derived_tail', 'exec'), namespace)
    derived = namespace['consumed']
    def consumed(control: Any, call: Any, caller: Any, rows: list[Any]) -> None:
        selected = original if len(call['view'].refs) <= 1 else derived
        selected(control, call, caller, rows)
    return consumed
