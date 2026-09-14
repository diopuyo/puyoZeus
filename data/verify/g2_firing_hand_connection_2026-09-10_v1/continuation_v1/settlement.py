"""同callerの実raw二票でのみ私有精算する。予測は公開盤面へ書かない。"""
from __future__ import annotations
from dataclasses import asdict
import sys
from typing import Any

STRIDE, OBSERVED_OPERATION, AVAILABLE_OPERATION = 2, 0, 1


def observe(control: Any, binding: Any, ticket: Any, raw: Any, signals: Any, view: Any, pipe: Any) -> Any:
    if getattr(ticket, 'settled', False):
        return None
    assert binding is ticket.binding and binding.scope == ticket.scope == view.scope
    if raw != ticket.origin.predicted_final or signals.effect_gate_window_active is not False:
        ticket.final_observations = []
        return None
    p, state = control.inventory, binding.owner.state
    current_raw, capture = control.provider.raw(pipe, view.scope[-1], view)
    assert current_raw == raw and capture['captured_frame'] == view.frame
    assert state.counter == p.S.color_counts(ticket.grid) and ticket.origin in state.origins
    clock = control._parts.C.H.clock
    observed = clock(p, view.frame, view.clock, OBSERVED_OPERATION)
    available = clock(p, view.frame, view.clock, AVAILABLE_OPERATION)
    row = dict(kind='captured_clear_raw/v1', scope=asdict(state.scope), source_token=ticket.token,
        action=state.action, observed_at=asdict(observed), available_at=asdict(available),
        raw_grid=raw, effect_window=False, capture_ref=p.digest(dict(capture=capture, scope=view.scope)))
    previous = getattr(ticket, 'final_observations', [])
    if previous and previous[-1]['observed_at']['frame']+STRIDE != view.frame:
        previous = []
    ticket.final_observations = (previous+[row])[-STRIDE:]
    return complete(control, binding, ticket, available) if len(ticket.final_observations) == STRIDE else None


def complete(control: Any, binding: Any, ticket: Any, now: Any) -> dict[str, Any]:
    p, state = control.inventory, binding.owner.state
    observations = tuple(ticket.final_observations)
    identifier = p.digest(dict(token=ticket.token, origin=asdict(ticket.origin), observations=observations))
    evidence = p.S.SettlementEvidence(state.scope, ticket.origin, now, p.S.counter_claim(state), identifier)
    proof = dict(kind='firing_settlement/v1', source_token=ticket.token,
                 evidence=asdict(evidence), observations=observations)
    binding.policy.arm('settlement', evidence, state, proof)
    binding.owner.settle(evidence, now)
    assert not binding.owner.state.debts and ticket.origin.origin_id in binding.owner.state.consumed_ids
    assert binding.owner.state.counter == p.S.color_counts(ticket.origin.predicted_final)
    binding.grid, ticket.settled = ticket.origin.predicted_final, True
    ticket.settlement_proof = proof
    return dict(stage='actual_raw_settled', frame=now.frame, proof=proof,
                private_state=asdict(binding.owner.state), current=binding.current,
                predicted_final_published=False, physical_certified=False)


def install(stack: Any, control: Any, patch: Any, rows: list[Any]) -> None:
    cls, original = type(control), type(control).observed
    def observed(self: Any, sm: Any, signals: Any, view: Any, pipe: Any, side: str) -> Any:
        caller = sys._getframe(1)
        assert caller.f_code is self._parts.C.Controller.update.__code__, 'settlement_actual_controller_call'
        assert caller.f_locals['signals'] is signals and caller.f_locals['pipe'] is pipe
        raw = original(self, sm, signals, view, pipe, side)
        binding = self.history.get(side)
        ticket = getattr(binding, 'firing_ticket', None)
        if ticket is not None and ticket.consumed:
            result = observe(self, binding, ticket, raw, signals, view, pipe)
            if result is not None: rows.append(result)
        return raw
    patch(stack, cls, 'observed', observed)
