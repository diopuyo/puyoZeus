"""精算前のNEXTは原queueへ保持し、精算後の現在時計で次actionを始める。"""
from __future__ import annotations
from copy import deepcopy
from types import MethodType
from typing import Any


def current(control: Any, view: Any) -> Any:
    provider, side = control.provider, view.scope[-1]
    row = provider.link.current(view)
    owner = provider.journal.fifo.entries[(id(row['pipe']), side)]
    scope = provider.journal.scope(row['pipe'], side, view.frame, view.clock)
    enqueue = provider.enqueues[side]
    provider._parts.V.Provider.check_enqueue(provider, enqueue, scope, row['epoch'], owner)
    assert tuple(enqueue['added_occurrence_tokens']) == view.added
    assert owner['queue'] is view.queue and tuple(owner['tokens']) == view.tokens
    return row, enqueue


def retain(control: Any, ticket: Any, view: Any) -> None:
    row, enqueue = current(control, view)
    assert len(view.refs) <= 1 and tuple(view.queue) == view.refs, 'firing_pending_multiple_hands'
    saved = getattr(ticket, 'next_witness', None)
    if view.added:
        assert saved is None and len(view.refs) == 1 and view.added == view.tokens
        assert view.tokens[0] != ticket.token and row['committed'] is True
        proof = dict(enqueue=deepcopy(enqueue), frame=view.frame, scope=view.scope, token=view.tokens[0])
        ticket.next_witness = dict(proof=proof, digest=control.inventory.digest(proof),
                                   queue=view.queue, head=view.refs[0], token=view.tokens[0])
    elif saved is not None:
        assert control.inventory.digest(saved['proof']) == saved['digest']
        assert view.queue is saved['queue'] and view.tokens == (saved['token'],)
        assert len(view.refs) == 1 and view.refs[0] is saved['head']


def install(stack: Any, control: Any, patch: Any, rows: list[Any]) -> None:
    cls, original, no_origin = type(control), control.hand, control.provider.no_origin
    def hand(self: Any, binding: Any, view: Any) -> Any:
        ticket = getattr(binding, 'firing_ticket', None)
        if ticket is None or not ticket.consumed or getattr(ticket, 'next_started', False):
            return original(binding, view)
        assert ticket.scope == binding.scope == view.scope
        retain(self, ticket, view)
        if not getattr(ticket, 'settled', False) or not view.refs or not view.quiet:
            return None
        saved = ticket.next_witness
        assert binding.next_token is None and view.tokens == (saved['token'],)
        assert view.refs[0] is saved['head'] and saved['proof']['frame'] <= view.frame
        self._parts.C.H.start(self.inventory, binding, saved['token'], view.frame, view.clock)
        ticket.next_started = True
        rows.append(dict(stage='next_action_started', frame=view.frame, source_frame=saved['proof']['frame'],
            token=saved['token'], action=binding.owner.state.action, backdated=False))
        return original(binding, view)
    def allowed(self: Any, pipe: Any, side: str, view: Any) -> bool:
        ticket = getattr(control.history.get(side), 'firing_ticket', None)
        if ticket is not None and ticket.consumed and ticket.scope == view.scope:
            active = getattr(pipe, '_active_chain_'+side.lower())
            return (active is None or active is ticket.event) and ticket.origin in ticket.binding.owner.state.origins
        return no_origin(pipe, side, view)
    patch(stack, cls, 'hand', hand)
    patch(stack, control.provider, 'no_origin', MethodType(allowed, control.provider))
