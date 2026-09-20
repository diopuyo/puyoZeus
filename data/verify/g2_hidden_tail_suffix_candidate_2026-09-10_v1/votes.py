"""元TailVotesへqueue内容の不変snapshotを付記し、suffix変化時は新二票を要求する。"""
from __future__ import annotations
from copy import deepcopy
from typing import Any


def matches(previous: Any, refs: tuple[Any, ...], tokens: tuple[str, ...],
            scope: tuple[Any, ...]) -> bool:
    saved = getattr(previous, 'queue_refs', None)
    return (type(saved) is tuple and len(saved) == len(refs)
        and all(old is new for old, new in zip(saved, refs))
        and getattr(previous, 'queue_tokens', None) == tokens
        and getattr(previous, 'queue_scope', None) == scope)


def vote(original_vote_function: Any, control: Any, binding: Any, item: Any,
         final: Any, raw: Any, proof: Any, view: Any) -> Any:
    refs, tokens, scope = tuple(view.refs), tuple(view.tokens), tuple(view.scope)
    previous = getattr(binding, 'hidden_tail_votes', None)
    if previous is not None and not matches(previous, refs, tokens, scope):
        binding.hidden_tail_votes = None
    value = original_vote_function(control, binding, item, final, raw, proof, view)
    value.queue_refs, value.queue_tokens, value.queue_scope = refs, tokens, scope
    successor = getattr(binding, 'hidden_tail_successor', None)
    if len(refs) > 1 and successor is not None:
        value.successor_proof = deepcopy(successor.proof)
    return value
