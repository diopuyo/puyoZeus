"""Session寿命に2P到来tap/状態を束縛する。評価数学と原1Pは保持する。"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def connected(base: type, ledger: Any, native: Any, tap: Any, binding: Any,
              services_factory: Any = None) -> type:
    class Session(base):
        def __init__(self, stack: Any, context: Any, policy: Any, physical: Any,
                     contract: Any, members: Any, frames: tuple) -> None:
            output = Path(context['state']['output'])
            self._arrival_bindings: dict[int, tuple] = {}
            self._arrival_output = output
            super().__init__(stack, context, policy, physical, contract, members, frames)
            self._arrival_services = None if services_factory is None else services_factory()
            ledger.require(self._arrival_services is None or hasattr(self, 'second_settled_notice_stream'),
                           'second_arrival_notice_stream_missing')
            ledger.require(not self.modes and self.mode is None, 'second_arrival_late_session')
            self._arrival_tap = tap.Tap(self.witness, output / 'SECOND_ENQUEUE_SOURCE.jsonl', stack)
            selected = mode_class(self._prefix_mode_type, self)
            self._prefix_mode_type = selected
            self.physical = SimpleNamespace(Mode=selected)
            stack.push(self.close_arrivals)

        def observe_arrivals(self, mode: Any, item: dict) -> None:
            if mode.prefix.lane is None:
                return
            key = id(mode)
            if key not in self._arrival_bindings:
                call = mode.connection.binding.initial_call_token
                name = sha256(call.encode()).hexdigest()[:16]
                value = binding.Binding(mode, self._arrival_tap, ledger, native,
                    self._arrival_output / ('SECOND_ARRIVAL_' + name + '.jsonl'))
                self._arrival_bindings[key] = (mode, value)
                value.terminal = None
                if self._arrival_services is not None:
                    services = self._arrival_services
                    value.notice_stream = self.second_settled_notice_stream
                    value.warning = services.warning.History(value, services.sampler,
                        self._arrival_output / ('SECOND_WARNING_' + name + '.jsonl'))
            owned, value = self._arrival_bindings[key]
            ledger.require(owned is mode, 'second_arrival_mode_reused')
            value.observe(item)
            if self._arrival_services is not None:
                value.warning.observe(item)

        def close_arrivals(self, kind: Any, body: Any, trace: Any) -> bool:
            failure = None
            for mode, value in self._arrival_bindings.values():
                for name in ('terminal', 'warning'):
                    component = getattr(value, name, None)
                    if component is not None:
                        try:
                            component.close()
                        except BaseException as error:
                            failure = failure or error
                try:
                    value.close(kind, body, trace)
                except BaseException as error:
                    failure = failure or error
            self._arrival_bindings.clear()
            if failure is not None and body is None:
                raise failure
            return False

        def terminal(self, mode: Any) -> Any:
            pair = self._arrival_bindings.get(id(mode))
            ledger.require(pair is None or pair[0] is mode, 'second_arrival_mode_reused')
            if pair is not None and pair[1].failure is not None:
                raise pair[1].failure
            return None if pair is None else getattr(pair[1], 'terminal', None)

        def recover(self, mode: Any, item: dict, result: Any, row: dict, error: Any) -> dict:
            if self._arrival_services is None:
                raise error
            ledger.require(id(mode) in self._arrival_bindings, 'second_recovery_unbound_mode')
            owned, value = self._arrival_bindings[id(mode)]
            ledger.require(owned is mode, 'second_arrival_mode_reused')
            name = sha256(mode.connection.binding.initial_call_token.encode()).hexdigest()[:16]
            return self._arrival_services.recovery.recover(value, self._arrival_services,
                item, result, row, error, self._arrival_output / ('SECOND_TERMINAL_' + name + '.jsonl'))

        def original_applied(self, mode: Any, row: dict) -> None:
            if mode.prefix.lane is None:
                return
            ledger.require(id(mode) in self._arrival_bindings, 'second_arrival_unbound_mode')
            owned, value = self._arrival_bindings[id(mode)]
            ledger.require(owned is mode, 'second_arrival_mode_reused')
            value.original_applied(row)
    return Session


def mode_class(base: type, session: Any) -> type:
    class Mode(base):
        def observe(self, item: Any, result: Any, error: Any) -> dict:
            try:
                return super().observe(item, result, error)
            finally:
                vars(self).pop('_second_stable_cache', None)

        def capture_origin(self, item: dict) -> None:
            terminal = session.terminal(self)
            if terminal is None:
                super().capture_origin(item)
            session.observe_arrivals(self, item)
            if terminal is not None:
                terminal.capture(item)

        def stable(self, item: Any, result: Any) -> tuple:
            observed, reason = super().stable(item, result)
            self._second_stable_cache = (item, result, observed, reason)
            return observed, reason

        def follow_hand(self, item: Any, result: Any, row: dict) -> dict:
            try:
                return self._second_follow_hand(item, result, row)
            finally:
                vars(self).pop('_second_stable_cache', None)

        def _second_follow_hand(self, item: Any, result: Any, row: dict) -> dict:
            terminal = session.terminal(self)
            if terminal is not None:
                return terminal.progress(item, result, row)
            try:
                following = super().follow_hand(item, result, row)
            except ValueError as error:
                if str(error) != 'probabilistic_scope:consumed_prefix_zero_support':
                    raise
                following = session.recover(self, item, result, row, error)
            session.original_applied(self, following)
            return following
    return Mode
