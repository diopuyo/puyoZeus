"""既存Controller内の履歴消費。現在公開を成功したと見なす経路は持たない。"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import sys
from typing import Any, Iterator
import transaction as T
import history_state as H

CLEAR_OBSERVATIONS = 2
LIVE_SIDES = frozenset(('1P', '2P'))


class Controller(T.Controller):
    """同call ticketと既存Sの参照を保持。OFF時は元Controllerへ委譲する。"""

    def __init__(self, provider: Any, legal: Any, *, enabled: bool = False,
                 inventory: Any = None) -> None:
        super().__init__(provider, legal, enabled=enabled)
        self.inventory = inventory
        self.history: dict[str, H.Binding] = {}
        self.calls: dict[int, dict[str, Any]] = {}

    def observed(self, sm: Any, signals: Any, view: Any, pipe: Any, side: str) -> Any:
        raw, proof = self.provider.raw(pipe, side, view)
        H.require(proof['captured_frame'] == view.frame and proof['raw_grid'] == raw, 'live_raw_binding')
        H.require(self.provider.no_origin(pipe, side, view), 'history_origin_unknown')
        if raw != T.board_key(signals.cnn_board):
            return None
        if any(color == 10 for row in raw for color in row):
            return None
        self.inventory.S.validate_grid(raw)
        return raw

    def baseline(self, sm: Any, view: Any, raw: Any, side: str) -> H.Binding | None:
        if not self.provider.baseline_selected(view):
            return None
        H.require(raw is not None and sm.context.state.value == 'stable'
                  and T.board_key(sm.context.confirmed_board) == raw, 'history_baseline_not_current')
        H.require(not view.refs and not view.tokens, 'baseline_FIFO_not_empty')
        proof = {'kind': 'live_history_baseline', 'frame': view.frame, 'time_sec': view.clock,
                 'grid': raw, 'live_scope': view.scope, 'prior_legacy_debt': 'UNKNOWN'}
        binding = H.establish(self.inventory, view, raw, proof)
        self.history[side] = binding
        return binding

    def hand(self, binding: H.Binding, view: Any) -> Any:
        H.require(binding.scope == view.scope and binding.phase == 'historical_owned', 'history_scope_changed')
        if not view.refs:
            H.require(binding.next_token is None, 'history_pending_slot_lost')
            return None
        if binding.next_token is None:
            H.require(len(view.refs) == 1 and view.added == (view.tokens[0],), 'first_token_not_fresh')
            H.start(self.inventory, binding, view.tokens[0], view.frame, view.clock)
        H.require(view.tokens[0] == binding.next_token and T.valid_pair(view.refs[0]), 'history_head_changed')
        if binding.candidate is None:
            H.require(T.valid_pair(view.next_pair) and T.valid_pair(view.dnext_pair) and view.quiet,
                      'history_NEXT_unknown')
            binding.candidate = T.Candidate(view.scope, view.queue, view.tokens[0], view.refs[0],
                binding.grid, view.next_pair, view.dnext_pair, binding.next_started[0])
        item = binding.candidate
        H.require(item.queue is view.queue and item.pair is view.refs[0], 'history_FIFO_reference_changed')
        return item

    def observe_clear(self, binding: H.Binding, item: Any, sm: Any, raw: Any,
                      signals: Any, view: Any) -> bool:
        exact = raw is not None and self.legal(binding.grid, raw, list(item.pair))
        if signals.effect_gate_window_active is not False:
            return bool(exact)
        if not exact:
            binding.clear_grid = binding.clear_first = binding.clear_last = None
            binding.clear_count = 0
            return False
        adjacent = binding.clear_last is not None and self.provider.adjacent(binding.clear_last, view)
        if binding.clear_grid != raw or not adjacent:
            binding.clear_grid, binding.clear_first, binding.clear_count = raw, (view.frame, view.clock), 0
        binding.clear_last = (view.frame, view.clock)
        binding.clear_count += 1
        return True

    def prepared(self, binding: H.Binding, item: Any, sm: Any, raw: Any,
                 signals: Any, view: Any) -> Any:
        exact = self.observe_clear(binding, item, sm, raw, signals, view)
        if not (exact and binding.clear_count >= CLEAR_OBSERVATIONS and binding.clear_grid == raw
                and self.advance(item, view)):
            return None
        proof = {'kind': 'live_historical_placement', 'token': item.token, 'pair': item.pair,
            'occurred': binding.clear_first, 'available_frame': view.frame, 'available_time': view.clock,
            'clear_last': binding.clear_last, 'clear_observations': binding.clear_count,
            'available_window': signals.effect_gate_window_active, 'new_token': view.added[0]}
        return H.prepare(self.inventory, binding, view, raw, proof)

    @contextmanager
    def hold_transition(self, sm: Any, signals: Any, binding: H.Binding) -> Iterator[None]:
        cls, original = type(sm), type(sm)._apply_transition
        H.require(T.original_code(original.__code__, '_apply_transition'), 'history_original_transition')
        preview = T.P.Controller(cls, original)
        def wrapped(current: Any, target: Any, observed: Any) -> None:
            caller = sys._getframe(1)
            H.require(current is sm and observed is signals and T.original_code(caller.f_code, 'update'),
                      'history_actual_SM_update')
            if target.value == 'stable' and current.context.state.value in T.ACTIVE_STATES:
                preview.within(current, observed)
                H.require(T.board_key(current.context.confirmed_board) == binding.current,
                          'history_hold_changed_confirmed')
                return
            original(current, target, observed)
        cls._apply_transition = wrapped
        try:
            yield
        finally:
            cls._apply_transition = original

    def update(self, sm: Any, frame: int, signals: Any, pipe: Any, side: str) -> Any:
        caller = sys._getframe(1)
        if not self.enabled:
            return sm.update(frame, signals)
        H.require(self.sticky_error is None, 'history_previous_failure')
        if side not in self.history and not self.provider.selected(pipe, side, frame, signals.time_sec):
            return sm.update(frame, signals)
        try:
            H.require(self.inventory is not None and side in LIVE_SIDES, 'history_inventory_unconnected')
            view = self.provider.view(pipe, side, frame, signals.time_sec, caller)
            raw = self.observed(sm, signals, view, pipe, side)
            binding = self.history.get(side) or self.baseline(sm, view, raw, side)
            H.require(binding is not None and id(caller) not in self.calls, 'history_missing_baseline_or_duplicate')
            H.require(T.board_key(sm.context.confirmed_board) == binding.current, 'history_current_changed')
            item = self.hand(binding, view)
            prepared = self.prepared(binding, item, sm, raw, signals, view) if item is not None else None
            self.calls[id(caller)] = self.call(caller, binding, view, pipe, side, prepared)
            with self.hold_transition(sm, signals, binding):
                result = sm.update(frame, signals)
            H.require(T.board_key(result.confirmed_board) == binding.current, 'history_SM_changed_current')
            return result
        except BaseException as exc:
            self.sticky_error = repr(exc)
            raise

    def call(self, caller: Any, binding: H.Binding, view: Any, pipe: Any,
             side: str, prepared: Any) -> dict[str, Any]:
        return {'frame': caller, 'binding': binding, 'view': view, 'prepared': prepared, 'stage': 0,
            'counter': Counter(getattr(pipe, '_tsumo_count_' + side.lower())),
            'first_move': getattr(pipe, '_first_move_sec_' + side.lower()),
            'new_color': getattr(pipe, '_last_consumed_color_' + side.lower()),
            'legacy_counter_gate_calls': 0, 'main_infer_gate_calls': 0, 'consumed': False}

    def gate(self, stage: str, original: bool) -> bool:
        caller = sys._getframe(1)
        call = self.bound_call(caller)
        if call is None:
            return original
        H.require(call['frame'] is caller and stage in ('consume', 'infer'), 'history_gate_caller')
        if stage == 'consume':
            H.require(call['stage'] == 0, 'history_duplicate_consume_gate')
            call['stage'] = 1
            if call['prepared'] is not None:
                self.provider.before_history_consume(call, caller)
                return True
            return False
        H.require(call['stage'] == 1, 'history_infer_gate_order')
        call['stage'], call['main_infer_gate_calls'] = 2, call['main_infer_gate_calls'] + 1
        if call['prepared'] is not None:
            self.consumed_history(call, caller)
        return False

    def legacy_accounting(self) -> bool:
        caller = sys._getframe(1)
        call = self.bound_call(caller)
        if call is None:
            return True
        H.require(call['prepared'] is not None and call['stage'] == 1, 'history_unprepared_pop')
        call['legacy_counter_gate_calls'] += 1
        H.require(call['legacy_counter_gate_calls'] == 1, 'history_duplicate_counter_gate')
        return False

    def side_effect_gate(self, kind: str, original: bool) -> bool:
        caller = sys._getframe(1)
        call = self.bound_call(caller)
        H.require(kind in ('secondary_infer', 'grace'), 'history_unknown_side_effect')
        return original if call is None else False

    def bound_call(self, caller: Any) -> Any:
        call = self.calls.get(id(caller))
        H.require(call is not None or not self.enabled or caller.f_locals.get('side') not in self.history,
                  'history_owned_scope_bypassed_update')
        return call

    def consumed_history(self, call: Any, caller: Any) -> None:
        binding, view, values = call['binding'], call['view'], caller.f_locals
        H.require(values.get('committed') is view.refs[0] and len(view.queue) + 1 == len(view.refs)
                  and all(a is b for a, b in zip(view.queue, view.refs[1:])), 'history_wrong_native_pop')
        self.unchanged(call, values['self'], values['side'])
        self.provider.after_history_consume(call, caller)
        H.commit(self.inventory, binding, call['prepared'], view.added[0], view)
        call['consumed'] = True

    def unchanged(self, call: Any, pipe: Any, side: str) -> None:
        suffix = side.lower()
        H.require(Counter(getattr(pipe, '_tsumo_count_' + suffix)) == call['counter'], 'legacy_counter_changed')
        H.require(getattr(pipe, '_first_move_sec_' + suffix) == call['first_move'], 'legacy_first_move_changed')
        H.require(getattr(pipe, '_last_consumed_color_' + suffix) is call['new_color'], 'new_hand_color_changed')

    def finish(self, caller: Any, error: BaseException | None = None) -> None:
        call = self.calls.pop(id(caller), None)
        if call is None:
            return
        try:
            if error is not None:
                self.sticky_error = repr(error)
                return
            H.require(call['stage'] == 2 and call['main_infer_gate_calls'] == 1, 'history_incomplete_step')
            H.require(call['consumed'] == (call['prepared'] is not None), 'history_unfinished_consumption')
            values, view, binding = caller.f_locals, call['view'], call['binding']
            self.unchanged(call, values['self'], values['side'])
            self.provider.after_history_step(call, caller)
            self.records.append({'frame': view.frame, 'side': view.scope[-1], 'mode': binding.phase,
                'history_consumed': call['consumed'], 'old_token': view.tokens[0] if view.tokens else None,
                'history_state': asdict(binding.owner.state), 'legacy_counter_changed': False,
                'current_permission': False, 'quality_gate_clear': False, 'infer_calls': 0})
        except BaseException as exc:
            self.sticky_error = repr(exc)
            raise
        finally:
            call['frame'] = None
