"""単headは原委譲。suffixがある場合だけ原J所有列全体へtailを束縛する。"""
from __future__ import annotations
from typing import Any
import tail_suffix_provenance as SP


def owner(control: Any, view: Any, require: Any) -> Any:
    provider, side = control.provider, view.scope[-1]
    row = provider.link.current(view)
    journal = provider.journal
    actual = journal.fifo.entries[(id(row['pipe']), side)]
    require(actual['queue'] is view.queue and tuple(actual['tokens']) == view.tokens
        and len(actual['refs']) == len(view.refs) == len(view.queue)
        and all(old is new for old,new in zip(actual['refs'],view.refs))
        and all(old is new for old,new in zip(view.queue,view.refs)), 'tail_suffix_J_owner')
    scope = journal.scope(row['pipe'], side, view.frame, view.clock)
    enqueue = provider.enqueues.get(side)
    provider._parts.V.Provider.check_enqueue(provider, enqueue, scope, row['epoch'], actual)
    require(tuple(enqueue['added_occurrence_tokens']) == view.added, 'tail_suffix_J_added')
    require(len(set(view.tokens)) == len(view.tokens)
        and len(set(view.added)) == len(view.added)
        and all(token in view.tokens[1:] for token in view.added), 'tail_suffix_added')
    return row


def source(original_source_function: Any, control: Any, binding: Any,
           item: Any, view: Any) -> Any:
    if len(view.refs) <= 1:
        return original_source_function(control, binding, item, view)
    parts = original_source_function.__globals__
    require, validate = parts['P'].require, parts['V'].validate
    previous = getattr(binding, 'hidden_prefix_votes', None)
    require(previous is not None and previous.binding is binding, 'tail_prefix_source')
    validate(control, binding, previous)
    held = previous.held
    require(held.consumed and held.binding is binding and binding.grid == previous.support.prefix,
        'tail_prefix_not_committed')
    require(view.scope == binding.scope == held.scope
        and len(view.refs) == len(view.tokens) == len(view.queue), 'tail_scope_or_slots')
    require(view.queue is held.queue and view.refs[0] is held.refs[1] is item.pair
        and item.queue is view.queue and view.queue[0] is item.pair, 'tail_reference')
    require(binding.next_token == item.token == view.tokens[0] == held.tokens[1]
        and item.token not in binding.consumed_tokens, 'tail_token')
    require(view.scope[-1] not in control.provider.handoff_proofs, 'tail_stale_held')
    row = owner(control, view, require)
    if not SP.ready(control, binding, item, view, row, require):
        return None
    require(binding.next_started is not None and view.frame > binding.next_started[0]
        and view.frame > previous.last[0], 'tail_started_clock')
    return previous
