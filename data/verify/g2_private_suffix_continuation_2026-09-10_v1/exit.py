"""元SMの自然STABLE選択と空FIFO資格は保持し、最終私有配置だけ照合する。"""
from __future__ import annotations
from dataclasses import replace
import json
from typing import Any
import natural_exit as N
import conditional_current as C


def eligible(basis: Any, placement: Any, control: Any, binding: Any, signals: Any, call: Any) -> Any:
    view, state = call['view'], binding.owner.state
    if getattr(binding, 'hidden_current', None) is not None:
        return None
    if (call['prepared'] is not None or call['consumed'] or binding.next_token is not None
        or view.refs or view.tokens or signals.is_match_active is not True
        or signals.chain_event is not None or signals.effect_gate_window_active is not False):
        return None
    if view.quiet is not True or not control._parts.T.valid_pair(view.next_pair):
        return None
    C.P.require(not state.origins and not state.debts and binding.scope == view.scope, 'private_exit_scope')
    C.P.require(state.counter == control.inventory.S.color_counts(binding.grid), 'private_exit_counter')
    proof = placement.committed(basis, control, binding)
    C.P.require(view.frame > proof['available_frame'], 'private_exit_clock')
    captured = N.W.observed(control, binding, signals, view)
    return captured if captured is not None and C.P.compatible(captured[0], binding.grid) else None


def install(stack: Any, patch: Any, basis: Any, placement: Any) -> None:
    old_eligible, old_make = N.eligible, C.make
    def selected(control: Any, binding: Any, signals: Any, call: Any) -> Any:
        if (getattr(binding, 'private_suffix_placement', None) is not None
            and getattr(binding, 'hidden_last_placement', None) is None):
            return eligible(basis, placement, control, binding, signals, call)
        return old_eligible(control, binding, signals, call)
    def make(control: Any, call: Any, result: Any, provisional: Any) -> Any:
        made = old_make(control, call, result, provisional)
        if (made is None or getattr(call['binding'], 'private_suffix_placement', None) is None
            or getattr(call['binding'], 'hidden_last_placement', None) is not None):
            return made
        certificate, side = made
        proof = json.loads(certificate.evidence_json)
        proof['private_suffix_committed_source'] = placement.committed(basis, control, call['binding'])
        return replace(certificate, evidence_json=C.encoded(proof)), side
    patch(stack, N, 'eligible', selected)
    patch(stack, C, 'make', make)
