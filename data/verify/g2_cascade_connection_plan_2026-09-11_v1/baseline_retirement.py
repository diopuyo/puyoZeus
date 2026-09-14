"""原Jで閉鎖済みの基準cascadeに限り、遅着baseline一件の再発火を防ぐ。"""
from __future__ import annotations
from dataclasses import asdict
import json
import math
from typing import Any
import tracker_input as T

SIDE, FPS = '1P', 60


class Retirement:
    def __init__(self, pipe: Any, state: Any) -> None:
        self.pipe, self.state = pipe, state
        self.mode = state['probabilistic_tracking_mode']
        self.connection, self.used, self.rows = self.mode.connection, False, []
        self.factory = self.connection.recovery.factory
        self.tracker = pipe._chain_tracker_1p
        assert type(self.tracker) is T.tracker_type(type(pipe)), 'baseline_tracker_type'
        self.original = self.tracker.update
        assert 'update' not in vars(self.tracker), 'baseline_update_instance_override'
        assert self.original.__func__ is type(self.tracker).update, 'baseline_update_not_original'

    def qualified(self, event: Any, clock: float) -> str | None:
        mode, connection = self.mode, self.connection
        if self.used: return 'already_retired'
        if self.state['probabilistic_tracking_mode'] is not mode: return 'mode_changed'
        if mode.connection is not connection or mode.error is not None: return 'connection_or_error_changed'
        if self.pipe._chain_tracker_1p is not self.tracker: return 'tracker_changed'
        if not mode.basis_cascade_closed or mode.basis_origin is None: return 'basis_not_closed'
        if mode.native is None or mode.native.pending or mode.native.seen_occurrences: return 'new_native_hand'
        if len(mode.applied) != 1 or mode.applied[0]['kind'] != 'basis_cascade': return 'transition_changed'
        receipt, origin = mode.applied[0], mode.basis_origin
        if event.before_board is None or tuple(map(tuple, event.before_board.to_dict()['grid'])) != origin['grid']:
            return 'before_grid_mismatch'
        trigger = event.trigger_sec
        if not isinstance(trigger, (int, float)) or not math.isfinite(trigger): return 'invalid_trigger'
        if not origin['first_observed_frame']/FPS <= trigger <= receipt['applied_frame']/FPS < clock:
            return 'outside_settlement_window'
        if connection.recovery.factory is not self.factory or connection.registry.factory is not self.factory:
            return 'factory_changed'
        current = connection.registry.current(connection.binding)
        if current.scope != connection.binding.scope or current.scope[-1] != SIDE: return 'scope_changed'
        if current.frame != receipt['applied_frame']: return 'current_changed'
        serializer = type(connection).__init__.__globals__['S']
        if serializer.encode(current) != receipt['state']: return 'settled_distribution_changed'
        return None

    def update(self, clock: float, board: Any) -> Any:
        before = (self.tracker._leftover, self.tracker._all_clear_pending)
        event = self.original(clock, board)
        if event is None or event.mechanism != 'baseline': return event
        reason = self.qualified(event, clock)
        self.rows.append(dict(clock=clock, tracker_id=id(self.tracker), event=asdict(event),
            reason=reason, suppressed=reason is None, original_update_calls=1,
            leftover_before=before[0], all_clear_before=before[1],
            leftover_after=self.tracker._leftover, all_clear_after=self.tracker._all_clear_pending,
            basis_origin=self.mode.basis_origin,
            settlement_token=None if not self.mode.applied else self.mode.applied[0]['source_call_token'],
            quality_gate_clear=False))
        if reason is not None: return event
        self.used = True
        return None

    def close(self) -> None:
        with (self.state['output'] / 'BASELINE_RETIREMENT.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(rows=self.rows, used=self.used,
                tracker_restored='update' not in vars(self.tracker), quality_gate_clear=False),
                stream, default=lambda value: value.to_dict(), indent=2)


def install(stack: Any, pipe: Any, state: Any) -> Retirement:
    value = Retirement(pipe, state)
    owned = value.update
    def restore() -> None:
        assert vars(value.tracker).get('update') is owned, 'baseline_retirement_foreign_restore'
        delattr(value.tracker, 'update')
        assert value.tracker.update == value.original, 'baseline_retirement_original_restore'
    def closing(error_type: Any, original_error: Any, traceback: Any) -> bool:
        failures = []
        for operation in (restore, value.close):
            try:
                operation()
            except BaseException as error:
                failures.append(error)
        if failures:
            state['baseline_cleanup_errors'] = [repr(error) for error in failures]
            if original_error is None: raise failures[0]
        return False
    stack.push(closing)
    value.tracker.update = owned
    state['basis_baseline_retirement'] = value
    return value


def derived(parent: Any) -> Any:
    class Context(parent):
        def perform(self, advisory: Any, frame: int, clock: float) -> None:
            super().perform(advisory, frame, clock)
            install(self.stack, self.pipe, self.state)
    return Context
