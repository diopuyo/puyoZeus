"""整数ownerを作らず、既存reset leaseから確率ownerへの別型移行を監視する。"""
from __future__ import annotations
from typing import Any
import tracking_mode as M

B = M.B


def verify(mode: M.Mode, lease: Any) -> None:
    c, r = mode.connection, mode.connection.recovery
    B.require(lease.recovery is r and lease.used and not lease.active, 'probability_lease_identity')
    lease.archive.verify()
    value = c.registry.current(c.binding)
    actual = lease.evidence.scope(r.factory, r.pipe)
    old_scope = lease.archive.binding.scope
    B.require(actual == value.scope and actual[:5] == lease.new_scope[:5]
              and actual[-1] == lease.new_scope[-1] and actual[5] > old_scope[5], 'probability_lease_scope')
    B.require(lease.new_generation is None or lease.new_generation == actual[5], 'probability_lease_generation')
    B.require(r.pending is None and r.baseline_count == 0 and actual[-1] not in r.control.history,
              'probability_lease_not_integer')
    B.require(lease.guard.binding is None and c.binding.initial_call_token == mode.activation['source_call_token'],
              'probability_lease_binding')
    if lease.waiting:
        lease.waiting = False
        lease.new_generation = actual[5]
        lease.events.append(dict(kind='new_probabilistic_scope_owned', scope=actual,
            source_call_token=c.binding.initial_call_token, integer_owner=False,
            current_permission=False, physical_certified=False, quality_gate_clear=False))


def install(stack: Any, mode: M.Mode, lease: Any) -> None:
    original = lease.observe_wait
    def observe_wait() -> Any:
        if mode.native is None:
            return original()
        try:
            return verify(mode, lease)
        except BaseException as error:
            mode.fail(error)
            raise
    M.patch(stack, lease, 'observe_wait', observe_wait)
