"""既存会計policyへ委譲し、currentは履歴・最終全格子票に一回だけ束縛する。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any
import history_state as H

CURRENT_OPERATION = 3


def committed(p: Any, call: Any) -> Any:
    binding, prepared, view = call['binding'], call['prepared'], call['view']
    state, evidence = binding.owner.state, prepared['evidence']
    H.require(call['consumed'] and binding.scope == view.scope, 'current_uncommitted_scope')
    H.require(binding.grid == prepared['grid'] and binding.next_token == view.added[0]
        and view.tokens[0] in binding.consumed_tokens, 'current_history_binding')
    H.require(state.action == evidence.action + 1 and state.history[-1].event_id == evidence.event_id,
        'current_history_action')
    H.require(state.counter == p.S.color_counts(binding.grid) and not state.origins and not state.debts,
        'current_inventory_unknown')
    H.require(evidence.event_id == p.digest(prepared['proof']), 'current_placement_proof')
    H.require((state.action_since.frame, state.action_since.time_sec) == (view.frame, view.clock),
        'current_next_clock')
    return state


def proof(p: Any, call: Any, *, raw: Any, sm: Any, returned: Any, probability: Any) -> dict[str, Any]:
    binding, view, state = call['binding'], call['view'], committed(p, call)
    for grid in (raw, sm, returned, probability):
        p.S.validate_grid(grid)
        H.require(grid == binding.grid, 'current_final_channel_mismatch')
    return {'kind': 'completed_history_current', 'scope': asdict(state.scope), 'action': state.action,
        'frame': view.frame, 'clock': view.clock, 'grid': binding.grid,
        'placement': call['prepared']['proof'], 'physical_certified': False,
        'channels': {'raw': raw, 'sm': sm, 'returned': returned, 'probability': probability}}


def policy_type(p: Any) -> type:
    class Policy:
        """通常会計は元policy。一回current票をarmする前の検査はrecoverが担当。"""
        def __init__(self) -> None:
            self.accounting, self.current = p.BoundPolicy(), None

        def arm(self, purpose: str, evidence: Any, state: Any, value: Any) -> None:
            H.require(self.current is None, 'current_policy_busy')
            H.require(purpose != 'current', 'current_requires_final_channels')
            self.accounting.arm(purpose, evidence, state, value)

        def bind_current(self, evidence: Any, state: Any, value: Any) -> None:
            H.require(self.current is None and self.accounting._permit is None, 'current_policy_busy')
            H.require(evidence.evidence_id == p.digest(value), 'current_proof_id')
            self.current = (p.encoded(asdict(evidence)), p.encoded(asdict(state)))

        def authorize(self, purpose: str, evidence: Any, state: Any) -> None:
            if purpose != 'current':
                return self.accounting.authorize(purpose, evidence, state)
            H.require(self.current is not None, 'current_unbound_evidence')
            expected = (p.encoded(asdict(evidence)), p.encoded(asdict(state)))
            H.require(self.current == expected, 'current_unbound_evidence')
            self.current = None
    return Policy


def recover(p: Any, call: Any, value: Any) -> Any:
    state, binding, view = committed(p, call), call['binding'], call['view']
    H.require(value == proof(p, call, **value['channels']), 'current_proof_binding')
    now = H.clock(p, view.frame, view.clock, CURRENT_OPERATION)
    evidence = p.S.CurrentEvidence(state.scope, state.action, now, now, 'STABLE', binding.grid, p.digest(value))
    binding.policy.bind_current(evidence, state, value)
    try:
        return binding.owner.recover_current(evidence, now)
    finally:
        binding.policy.current = None
