"""既存原native popと同Bindingの配置を使う、未公開の発火分岐。"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, replace
import inspect
from types import MethodType
from typing import Any, Iterator
import firing_ticket as F

ORIGIN_OPERATION = 2


def start(pipe: Any, ticket: F.Ticket, clock: float) -> Any:
    env = ticket.environment
    score, base, estimated = pipe._fill_pseudo_chain_score(ticket.prediction)
    end = clock + pipe._chain_hold_base_sec + pipe._chain_hold_per_step_sec*ticket.prediction.chain_count
    event = env['ChainEvent'](trigger_sec=clock, end_sec=end, before_board=ticket.before,
        chain_count=ticket.prediction.chain_count, total_erased=0, total_score=score,
        base_score=base, all_clear_bonus_applied=0, ojama_sent=0, leftover_score=0,
        is_all_clear=False, mechanism=env['CHAIN_MECHANISM_LANDING'], score_estimated=estimated)
    pipe._start_chain_estimate(F.SIDE, event)
    pipe._active_chain_1p, pipe._chain_until_1p = event, end
    assert F.F.key(event.before_board) == ticket.grid
    ticket.event = event
    return event


def origin(control: Any, ticket: F.Ticket, call: Any, pipe: Any) -> Any:
    p, binding, view = control.inventory, ticket.binding, call['view']
    event = start(pipe, ticket, view.clock)
    now = control._parts.C.H.clock(p, view.frame, view.clock, ORIGIN_OPERATION)
    final = F.F.key(ticket.prediction.final_board)
    erased = tuple(a-b for a,b in zip(p.S.color_counts(ticket.grid), p.S.color_counts(final)))
    identifier = p.digest(dict(token=ticket.token, frame=view.frame, scope=ticket.scope, before=ticket.grid))
    value = p.S.OriginEvidence(binding.owner.state.scope, identifier, binding.owner.state.action,
        now, now, ticket.grid, final, erased, '', 'actual_landing_object:'+str(id(event)))
    return replace(value, prediction_sha256=p.S.prediction_digest(value))


def consumed(control: Any, call: Any, caller: Any, rows: list[Any]) -> None:
    binding, view, prepared = call['binding'], call['view'], call['prepared']
    ticket, p, pipe = prepared['ticket'], control.inventory, caller.f_locals['self']
    assert not ticket.consumed and binding.owner.state is prepared['old_state']
    assert caller.f_locals['committed'] is ticket.head and len(view.queue) == 0
    control.unchanged(call, pipe, F.SIDE)
    control.provider._parts.V.Provider.after_history_consume(control.provider, call, caller)
    binding.policy.arm('placement', prepared['evidence'], binding.owner.state, prepared['proof'])
    binding.owner.add_placement(prepared['evidence'], prepared['evidence'].available_at)
    binding.grid = ticket.grid
    ticket.origin = origin(control, ticket, call, pipe)
    proof = origin_proof(control, ticket, prepared)
    binding.policy.arm('origin', ticket.origin, binding.owner.state, proof)
    binding.owner.register_debt(ticket.origin, ticket.origin.available_at)
    binding.consumed_tokens.add(ticket.token)
    binding.next_token = binding.next_started = binding.candidate = None
    binding.clear_grid = binding.clear_first = binding.clear_last = None
    binding.clear_count, ticket.consumed, call['consumed'] = 0, True, True
    control.unchanged(call, pipe, F.SIDE)
    assert F.F.key(pipe._sm_1p.context.confirmed_board) == binding.current
    rows.append(dict(stage='firing_registered', frame=view.frame, origin=asdict(ticket.origin),
        private_state=asdict(binding.owner.state), current=binding.current, original_J_pop=True,
        native_counter_unchanged=True, predicted_final_published=False))


def origin_proof(control: Any, ticket: F.Ticket, prepared: Any) -> dict[str, Any]:
    return dict(kind='firing_origin/v1', evidence=asdict(ticket.origin), source_token=ticket.token,
                placement_event_id=prepared['evidence'].event_id)


@contextmanager
def installed(factory: Any, pipe: Any, base: Any, policy: Any, rows: list[Any]) -> Iterator[None]:
    from contextlib import ExitStack
    control, provider = factory.controller, factory.provider
    cls, original_prepared = type(control), type(control).prepared
    v1 = original_prepared.__globals__['V1']
    prepare, called, consume, no_origin = v1.prepared, cls.call, cls.consumed_history, provider.no_origin
    accounting = inspect.getclosurevars(control.inventory.BoundPolicy.__init__).nonlocals['p']
    def prepared(self: Any, binding: Any, *args: Any) -> Any:
        ticket = getattr(binding, 'firing_ticket', None)
        return F.prepare(self, binding, *args) if ticket is not None and not ticket.consumed else prepare(self, binding, *args)
    def call(self: Any, caller: Any, binding: Any, view: Any, obj: Any, side: str, proposal: Any) -> Any:
        if proposal is not None and proposal.get('kind') == 'firing':
            return self._parts.C.Controller.call(self, caller, binding, view, obj, side, proposal)
        return called(self, caller, binding, view, obj, side, proposal)
    def completed(self: Any, value: Any, caller: Any) -> None:
        if value['prepared'].get('kind') == 'firing':
            return consumed(self, value, caller, rows)
        return consume(self, value, caller)
    def allowed(self: Any, obj: Any, side: str, view: Any) -> bool:
        ticket = getattr(control.history.get(side), 'firing_ticket', None)
        if ticket is None or not ticket.consumed:
            return no_origin(obj, side, view)
        return (ticket.scope == view.scope and getattr(obj, '_active_chain_'+side.lower()) is ticket.event
                and ticket.origin in ticket.binding.owner.state.origins)
    with ExitStack() as stack:
        assert not control.history, 'firing_install_after_baseline'
        base.patch(stack, accounting, 'BoundPolicy', policy.policy_type(accounting))
        base.patch(stack, v1, 'prepared', prepared)
        base.patch(stack, cls, 'call', call)
        base.patch(stack, cls, 'consumed_history', completed)
        base.patch(stack, provider, 'no_origin', MethodType(allowed, provider))
        F.install(stack, factory, pipe, base.patch, rows)
        yield
    assert cls.prepared is original_prepared
