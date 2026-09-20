"""原完了の例外を保ち、所有hookの同一性を検査する同call観測版。"""
from __future__ import annotations
from dataclasses import asdict
import json
from types import MethodType
from typing import Any
import actual_connection as OLD
import gate_v3 as G


class Observer(OLD.Observer):
    def __init__(self, recovery: Any, reset_frame: int, deadline: int) -> None:
        super().__init__(recovery, reset_frame, deadline)
        self.gate = G.SettledBasisGate(self.gate.scope, reset_frame, deadline)
        self.audit: list[dict[str, Any]] = []
        self.calls = self.original_calls = self.original_returns = self.original_raises = 0
        self.propagated_observer_failure = False

    def capture(self, item: Any, result: Any, error: Any) -> None:
        if error is not None:
            self.gate._break('original_call_exception')
            self.audit.append(dict(reason='original_call_exception', token=item.get('token'), error=repr(error)))
            return
        if id(item['frame']) not in self.recovery.waits:
            pending = getattr(self.recovery, 'pending', None)
            side = item.get('scope', {}).get('side')
            missing = pending is not None and not pending['used'] and side == '1P'
            self.audit.append(dict(reason='missing_waiting_call' if missing else 'outside_waiting_scope',
                                   token=item.get('token')))
            G.V1.require(not missing, 'missing_waiting_call')
            return
        super().capture(item, result, error)


def completion(value: Observer, original: Any, item: Any, result: Any, error: Any) -> Any:
    value.calls += 1
    failure = None
    try:
        value.capture(item, result, error)
    except BaseException as caught:
        failure = caught
        value.error = repr(caught)
        value.gate._break('observer_failure')
    value.original_calls += 1
    try:
        returned = original(item, result, error)
    except BaseException:
        value.original_raises += 1
        raise  # 原例外オブジェクトをそのまま伝播し、観測エラーで置換しない。
    value.original_returns += 1
    if failure is not None:
        value.propagated_observer_failure = True
        raise failure
    return returned


def close(value: Observer, recovery: Any, state: dict[str, Any], bindings: tuple[Any, ...]) -> None:
    original, existed, stored, wrapper = bindings
    owned = vars(recovery).get('complete') is wrapper
    if owned:
        setattr(recovery, 'complete', stored) if existed else vars(recovery).pop('complete')
    restored = owned and recovery.complete == original
    candidate = value.gate.candidate
    result = dict(error=value.error, rows=value.gate.rows, audit=value.audit,
        candidate=None if candidate is None else asdict(candidate), restored=restored,
        restore_binding_owned=owned, actual_factory_observer=True, quality_gate_clear=False,
        calls=value.calls, original_calls=value.original_calls, original_returns=value.original_returns,
        original_raises=value.original_raises,
        original_integer_recovery_unchanged=(value.calls == value.original_calls
            and not value.propagated_observer_failure))
    with (state['output'] / 'SETTLED_BASIS_OBSERVER.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    G.V1.require(restored, 'restore_binding_changed')


def install(stack: Any, recovery: Any, state: dict[str, Any], reset_frame: int, deadline: int) -> Observer:
    G.V1.require('settled_basis_observer' not in state, 'duplicate_actual_basis')
    value = Observer(recovery, reset_frame, deadline)
    state['settled_basis_observer'] = value
    original = recovery.complete
    existed, stored = 'complete' in vars(recovery), vars(recovery).get('complete')
    def complete(self: Any, item: Any, result: Any, error: Any) -> Any:
        return completion(value, original, item, result, error)
    wrapper = MethodType(complete, recovery)
    stack.callback(close, value, recovery, state, (original, existed, stored, wrapper))
    recovery.complete = wrapper
    return value
