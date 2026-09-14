"""終了済み確率scopeの凍結所有を検査する。実scopeを過去scopeに差し替えない。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import terminal_boundary as C


def verify(mode: Any, lease: Any, factory: Any) -> tuple:
    capture, connection = mode.arrival_capture, mode.connection
    recovery = connection.recovery
    C.require(capture.mode is mode and capture.recovery is recovery and capture.terminal_receipt is not None
        and capture.error is None and mode.error is None and recovery.error is None, 'terminal_owned_capture')
    ledger = capture.frozen_ledger
    C.require(mode.arrival_ledger is ledger and asdict(ledger) == capture.terminal_receipt['last_live_ledger'],
        'terminal_ledger_mutated')
    C.require(connection.registry.current(connection.binding) is capture.frozen_current
        and capture.frozen_current.scope == connection.binding.scope == ledger.scope, 'terminal_current_mutated')
    C.require(mode.native.last_frame == ledger.clock and not mode.native.pending
        and capture.pending is None and capture.terminal_rows > 0, 'terminal_unfinished_call')
    C.require(lease.recovery is recovery and lease.used and not lease.active and not lease.waiting
        and lease.guard.binding is None and recovery.pending is None and recovery.baseline_count == 0,
        'terminal_lease_identity')
    C.require(factory is recovery.factory and factory is connection.registry.factory
        and ledger.scope[-1] not in recovery.control.history, 'terminal_factory_owner')
    lease.archive.verify()  # 元整数退役の保存を検査し、全体reset権限は作らない。
    actual = recovery.evidence.scope(factory, recovery.pipe)
    C.require(all(actual[i] == ledger.scope[i] for i in (0, 1, 3, 4, 6))
        and all(type(actual[i]) is int and actual[i] >= ledger.scope[i] for i in (2, 5)), 'terminal_actual_scope')
    C.require(capture.terminal_receipt['further_old_scope_updates_allowed'] is False,
        'terminal_updates_reenabled')
    return actual


def install(stack: Any, mode: Any, lease: Any, patch: Any) -> None:
    original = lease.observe_wait
    def observe_wait() -> Any:
        capture = mode.arrival_capture
        if getattr(capture, 'terminal_receipt', None) is None:
            return original()
        try:
            return verify(mode, lease, mode.connection.recovery.factory)
        except BaseException as error:
            mode.fail(error)
            raise
    patch(stack, lease, 'observe_wait', observe_wait)
