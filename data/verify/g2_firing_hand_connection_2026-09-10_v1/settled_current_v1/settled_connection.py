"""通常currentの最終channel/カウンタ経路を保ち、精算改訂だけ分岐する。"""
from __future__ import annotations
from dataclasses import asdict
from types import FunctionType
from typing import Any
import settled_hold as H
import settled_publication as U

CURRENT_OPERATION = 3


def committed(p: Any, call: Any) -> Any:
    binding, ticket, view = call['binding'], call['settlement_current_ticket'], call['view']
    state = binding.owner.state
    assert ticket.binding is binding and ticket.scope == binding.scope == view.scope
    assert ticket.consumed and ticket.settled and not getattr(ticket, 'current_published', False)
    assert call['prepared'] is None and not call['consumed'], 'settled_current_no_native_pop'
    assert state.action == ticket.origin.action and binding.next_token is None
    assert ticket.origin in state.origins and ticket.origin.origin_id in state.consumed_ids and not state.debts
    assert binding.grid == binding.current == ticket.origin.predicted_final
    assert state.counter == p.S.color_counts(binding.grid)
    return state


def proof(p: Any, call: Any, **channels: Any) -> dict[str, Any]:
    state = committed(p, call)
    binding, view, ticket = call['binding'], call['view'], call['settlement_current_ticket']
    assert set(channels) == {'raw', 'sm', 'returned', 'probability'}
    for grid in channels.values():
        p.S.validate_grid(grid)
        assert grid == binding.grid, 'settled_current_channel_mismatch'
    return dict(kind=U.KIND, scope=asdict(state.scope), action=state.action,
        frame=view.frame, clock=view.clock, grid=binding.grid, settlement=ticket.settlement_proof,
        channels=channels, physical_certified=False)


def install_policy(stack: Any, control: Any, patch: Any, revision: Any, rows: list[Any]) -> None:
    v3 = type(control).prepared.__globals__
    policy = v3['V1'].P
    old_committed, old_proof, old_recover = policy.committed, policy.proof, policy.recover
    def checked(p: Any, call: Any) -> Any:
        return committed(p, call) if 'settlement_current_ticket' in call else old_committed(p, call)
    def made(p: Any, call: Any, **channels: Any) -> Any:
        return proof(p, call, **channels) if 'settlement_current_ticket' in call else old_proof(p, call, **channels)
    def recover(p: Any, call: Any, value: Any) -> Any:
        if 'settlement_current_ticket' not in call:
            return old_recover(p, call, value)
        assert value == proof(p, call, **value['channels']), 'settled_current_proof_binding'
        binding, view, ticket = call['binding'], call['view'], call['settlement_current_ticket']
        now = control._parts.C.H.clock(p, view.frame, view.clock, CURRENT_OPERATION)
        evidence = p.S.CurrentEvidence(binding.owner.state.scope, binding.owner.state.action,
            now, now, 'STABLE', binding.grid, p.digest(value))
        result = revision(p, binding, ticket.origin, evidence, now, value)
        ticket.current_published = True
        rows.append(dict(stage='settled_current_published', frame=view.frame, grid=binding.current,
                         action=result.action, consumed=False, proof=value, physical_certified=False))
        return result
    patch(stack, policy, 'committed', checked)
    patch(stack, policy, 'proof', made)
    patch(stack, policy, 'recover', recover)


def install(stack: Any, control: Any, state: Any, patch: Any, revision: Any,
            rows: list[Any], next_module: Any) -> None:
    cls, old = type(control), type(control).hold_transition
    module = old.__globals__
    from types import SimpleNamespace
    value = SimpleNamespace(**module)
    def hold(self: Any, sm: Any, signals: Any, binding: Any) -> Any:
        return H.installed(self, sm, signals, binding, value)
    patch(stack, cls, 'hold_transition', hold)
    install_policy(stack, control, patch, revision, rows)
    U.install(stack, state, patch)
    def ready(ticket: Any, view: Any) -> bool:
        return getattr(ticket, 'current_published', False) and next_module.ready(ticket, view)
    install_next = FunctionType(next_module.install.__code__, dict(vars(next_module), ready=ready))
    install_next(stack, control, patch, rows)
