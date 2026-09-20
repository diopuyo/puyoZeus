"""原O/J証拠の参照を既存handoff枠で保持し、現在callで一回だけ再認可する。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

PAIR_SIZE, FPS = 2, 60


def require(value: bool, reason: str) -> None:
    if not value:
        raise RuntimeError('deferred_' + reason)


@dataclass
class Held:
    proof: dict[str, Any]
    proof_digest: str
    binding: Any
    state: Any
    item: Any
    scope: Any
    queue: Any
    refs: tuple[Any, ...]
    tokens: tuple[str, ...]
    view: Any = None
    consumed: bool = False


def check(control: Any, held: Held, binding: Any, item: Any, view: Any) -> None:
    provider, proof = control.provider, held.proof
    require(not held.consumed and held.binding is binding and held.item is item, 'holder_identity')
    require(binding.owner.state is held.state and binding.scope == held.scope == view.scope, 'state_scope')
    require(control.inventory.digest(proof) == held.proof_digest, 'proof_changed')
    require(type(view.frame) is int and view.frame >= proof['available_frame']
        and view.clock == view.frame/FPS, 'consumer_clock')
    require(view.queue is held.queue and tuple(view.tokens) == held.tokens
        and len(view.refs) == len(held.refs) == PAIR_SIZE
        and all(a is b for a,b in zip(view.refs, held.refs)), 'fifo_identity')
    require(item.queue is view.queue and item.pair is view.refs[0]
        and item.token == view.tokens[0] and binding.next_token == item.token, 'head_identity')
    require(proof['new_token'] == view.tokens[1] and proof['old_token'] == item.token
        and proof['new_token'] not in provider.link.used
        and item.token not in binding.consumed_tokens, 'token_used_or_changed')
    row = provider.link.current(view)
    require(row['epoch'] == proof['software_epoch'] and row['segment'] == proof['segment_id'], 'epoch_segment')
    require(row['accepted'] == proof['new_accepted'] and row['dnext'] == proof['dnext'], 'next_basis')
    owner = provider.journal.fifo.entries[(id(row['pipe']), view.scope[-1])]
    require(owner['queue'] is view.queue and tuple(owner['tokens']) == held.tokens
        and len(owner['refs']) == PAIR_SIZE
        and all(a is b for a,b in zip(owner['refs'], held.refs)), 'journal_owner')


def capture(control: Any, binding: Any, item: Any, view: Any) -> Held | None:
    provider, side = control.provider, view.scope[-1]
    held = provider.handoff_proofs.get(side)
    require(held is None or type(held) is Held, 'foreign_handoff_slot')
    if held is None and view.added:
        proof = provider.link.proof(item, view, provider.enqueues.get(side))
        require(type(proof) is dict and proof['kind'] == 'directional_commit_original_J/v1', 'origin_proof')
        held = Held(deepcopy(proof), control.inventory.digest(proof), binding, binding.owner.state,
                    item, view.scope, view.queue, tuple(view.refs), tuple(view.tokens))
        provider.handoff_proofs[side] = held
    if held is not None:
        check(control, held, binding, item, view)
    return held


def ready(held: Held, view: Any) -> bool:
    return (view.quiet is True and view.next_pair == held.proof['new_accepted']
            and view.dnext_pair == held.proof['dnext'])


def authorize(control: Any, held: Held, binding: Any, item: Any, view: Any) -> dict[str, Any]:
    check(control, held, binding, item, view)
    require(held.view is None and ready(held, view), 'authorization_twice_or_unready')
    held.view = view
    return dict(kind='directional_deferred_consumption/v1',
        source_available_frame=held.proof['available_frame'], source_available_time=held.proof['available_time'],
        consumer_frame=view.frame, consumer_time=view.clock, old_token=held.tokens[0], new_token=held.tokens[1],
        original_directional_sha256=held.proof_digest, physical_certified=False)


def from_call(call: Any, *, consumed: bool) -> Held:
    held = call.get('deferred_handoff')
    require(type(held) is Held and held.consumed is consumed and held.view is call['view'], 'call_authorization')
    require(held.binding is call['binding'] and call['prepared']['proof']['new_token'] == held.tokens[1]
        and call['prepared']['proof']['token'] == held.tokens[0], 'call_token')
    return held


def consume(provider: Any, call: Any) -> None:
    held = from_call(call, consumed=False)
    require(provider.handoff_proofs.get(call['view'].scope[-1]) is held, 'stored_holder')
    require(held.tokens[1] not in provider.link.used and len(held.queue) == PAIR_SIZE-1
        and held.queue[0] is held.refs[1], 'native_pop')
    provider.link.used.add(held.tokens[1])
    held.consumed = True
    del provider.handoff_proofs[call['view'].scope[-1]]
