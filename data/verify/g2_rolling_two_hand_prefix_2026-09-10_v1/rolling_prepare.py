"""原2手prepareを専用pending票で再用する限定候補。消費権は原Heldに委譲。"""
from __future__ import annotations
import ast
import inspect
from types import SimpleNamespace as N
from typing import Any

VOTES = 'hidden_rolling_prefix_votes'


def renamed(function: Any) -> Any:
    """原vote/clearの保存先だけを変更し、判定本体は維持する。"""
    tree = ast.parse(inspect.getsource(function))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'hidden_prefix_votes':
            node.value = VOTES
            changed += 1
        if isinstance(node, ast.Attribute) and node.attr == 'hidden_prefix_votes':
            node.attr = VOTES
            changed += 1
    assert changed == (2 if function.__name__ == 'vote' else 1), 'rolling_vote_source_shape'
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(function), 'exec'), namespace)
    return namespace[function.__name__]


def derive(history: Any) -> Any:
    """消費済み初回prefixの再入拒否一箇所だけを外側の厳密資格検査へ移す。"""
    tree = ast.parse(inspect.getsource(history.prepare))
    first = tree.body[0].body[0]
    expected = ast.parse("len(view.refs)!=P.PAIR_SIZE or getattr(binding,'hidden_prefix_consumed',False)",
        mode='eval').body
    assert isinstance(first, ast.If) and ast.dump(first.test) == ast.dump(expected), 'rolling_prepare_shape'
    first.test = first.test.values[0]
    witness = N(**(vars(history.W) | dict(vote=renamed(history.W.vote), clear=renamed(history.W.clear))))
    namespace = dict(history.prepare.__globals__, W=witness)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(history.prepare), 'exec'), namespace)
    result = namespace['prepare']
    result.__kwdefaults__ = history.prepare.__kwdefaults__
    return result


def qualify(provenance: Any, owner: Any, control: Any, binding: Any, item: Any, view: Any) -> Any:
    p = N(require=provenance.require, PAIR_SIZE=provenance.PAIR_SIZE)
    old = getattr(binding, 'hidden_prefix_votes', None)
    p.require(old is not None and old.binding is binding, 'rolling_old_prefix')
    provenance.validate(control, binding, old)
    p.require(getattr(binding, 'hidden_prefix_consumed', False)
        and not getattr(binding, 'hidden_tail_consumed', False), 'rolling_consumption_phase')
    p.require(binding.grid == old.support.prefix and old.held.consumed, 'rolling_prefix_grid')
    state = binding.owner.state
    p.require(not state.origins and not state.debts and getattr(binding, 'firing_ticket', None) is None,
        'rolling_nonfiring_lane')
    p.require(item.baseline == binding.grid and item.queue is view.queue, 'rolling_item_baseline')
    p.require(view.scope == binding.scope == old.held.scope, 'rolling_scope')
    p.require(len(view.refs) == len(view.tokens) == len(view.queue) == p.PAIR_SIZE, 'rolling_slots')
    p.require(view.queue is old.held.queue and view.refs[0] is old.held.refs[1] is item.pair,
        'rolling_head_reference')
    p.require(view.tokens[0] == old.held.tokens[1] == item.token == binding.next_token
        and item.token not in binding.consumed_tokens, 'rolling_head_token')
    p.require(view.frame > old.last[0] and binding.next_started is not None
        and view.frame > binding.next_started[0], 'rolling_clock')
    owner(control, view, p.require)
    return old


def prepare(original: Any, provenance: Any, owner: Any, lifecycle: Any, held: Any, core: Any,
            control: Any, binding: Any, item: Any, sm: Any, signals: Any, view: Any) -> Any:
    old = qualify(provenance, owner, control, binding, item, view)
    # 原Linkの成功票は追加callで保持する。観測がまだ着地を示さなくても消費しない。
    held.capture(control, binding, item, view)
    proposal = original(lifecycle, held, core, control, binding, item, sm, signals, view)
    provenance.require(binding.hidden_prefix_votes is old, 'rolling_old_votes_overwritten')
    if proposal is not None:
        proposal = proposal | dict(rolling_previous_votes=old)
    return proposal
