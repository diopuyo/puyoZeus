"""原Link成功票をそのcallで保持する。NEXT更新・原handoff枠・消費権を作らない。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

PAIR_SIZE, FPS = 2, 60
KIND = 'directional_commit_original_J/v1'


def content(value: Any) -> Any:
    """tuple/list/boolを区別した不変内容を作り、保持票の改変を検出する。"""
    if type(value) is dict:
        return ('dict', tuple(sorted((key, content(v)) for key, v in value.items())))
    if type(value) in (list, tuple):
        return (type(value).__name__, tuple(content(v) for v in value))
    if type(value) in (str, int, float, bool) or value is None:
        return (type(value).__name__, value)
    raise ValueError('tail_suffix_proof_value')


@dataclass(frozen=True)
class Successor:
    binding: Any
    state: Any
    item: Any
    item_content: Any
    queue: Any
    refs: tuple[Any, ...]
    tokens: tuple[str, ...]
    scope: tuple[Any, ...]
    proof: dict[str, Any]
    sealed: Any


def item_content(item: Any) -> Any:
    return content((item.token, item.scope, item.started, item.next_pair, item.dnext_pair))


def capture(control: Any, binding: Any, item: Any, view: Any, require: Any) -> Any:
    held = getattr(binding, 'hidden_tail_successor', None)
    require(len(view.refs) == len(view.tokens) == PAIR_SIZE, 'tail_suffix_supported_slots')
    if held is None and view.added:
        proof = control.provider.link.proof(item, view, control.provider.enqueues.get(view.scope[-1]))
        require(type(proof) is dict and proof['kind'] == KIND, 'tail_suffix_original_proof')
        held = Successor(binding, binding.owner.state, item, item_content(item), view.queue,
            tuple(view.refs), tuple(view.tokens), tuple(view.scope), deepcopy(proof), content(proof))
        binding.hidden_tail_successor = held
    require(held is None or type(held) is Successor, 'tail_suffix_foreign_source')
    return held


def check(control: Any, binding: Any, item: Any, view: Any, row: Any,
          held: Successor, require: Any) -> None:
    proof = held.proof
    require(held.binding is binding and held.state is binding.owner.state
        and held.item is item and item_content(item) == held.item_content, 'tail_suffix_state_item')
    require(content(proof) == held.sealed, 'tail_suffix_proof_changed')
    require(view.scope == held.scope == binding.scope and view.queue is held.queue
        and tuple(view.tokens) == held.tokens and len(view.refs) == len(held.refs)
        and all(a is b for a, b in zip(view.refs, held.refs)), 'tail_suffix_source_refs')
    require(proof['old_token'] == item.token == held.tokens[0]
        and proof['new_token'] == held.tokens[1]
        and proof['new_token'] not in control.provider.link.used, 'tail_suffix_source_tokens')
    require(type(view.frame) is int and view.frame >= proof['available_frame']
        and view.clock == view.frame / FPS and proof['available_time'] == proof['available_frame'] / FPS,
        'tail_suffix_source_clock')
    require(row['epoch'] == proof['software_epoch'] and row['segment'] == proof['segment_id'],
        'tail_suffix_source_epoch_segment')
    require(row['accepted'] == proof['new_accepted'] and row['dnext'] == proof['dnext'],
        'tail_suffix_source_next')
    require(not view.added or (view.frame == proof['available_frame']
        and view.added == (proof['new_token'],)), 'tail_suffix_source_added')


def ready(control: Any, binding: Any, item: Any, view: Any, row: Any, require: Any) -> bool:
    held = capture(control, binding, item, view, require)
    if held is None:
        return False
    check(control, binding, item, view, row, held, require)
    return (view.quiet is True and view.next_pair == held.proof['new_accepted']
        and view.dnext_pair == held.proof['dnext'])
