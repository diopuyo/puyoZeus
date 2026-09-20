"""既存historyのcommit後だけ原SM復帰し、実step返却までcurrent許可を遅延する。"""
from __future__ import annotations

from collections import Counter
import sys
from typing import Any
import history_state as H
import lifecycle as L
import transaction as T
import merge as M
import policy as P


def next_fall(sm: Any, signals: Any, binding: Any) -> None:
    """既に登録済みの次手がある場合、原transitionで前回currentを凍結する。"""
    if binding.owner.state.current is None or sm.context.state.value != 'stable' or binding.next_token is None:
        return
    original, module = type(sm)._apply_transition, sys.modules[type(sm).__module__]
    H.require(T.original_code(original.__code__, '_apply_transition'), 'current_next_original_transition')
    before = T.board_key(sm.context.confirmed_board)
    original(sm, module.BoardState.TSUMO_FALL, signals)
    H.require(T.board_key(sm.context.confirmed_board) == before == binding.current, 'current_next_changed_grid')


def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
    exact = self.observe_clear(binding, item, sm, raw, signals, view)
    if not (exact and binding.clear_count >= self._parts.C.CLEAR_OBSERVATIONS and binding.clear_grid == raw):
        return None
    directional = self.provider.advance_evidence(item, view)
    if directional is None:
        return None
    value = {'kind': 'live_historical_placement', 'token': item.token, 'pair': item.pair,
        'occurred': binding.clear_first, 'available_frame': view.frame, 'available_time': view.clock,
        'clear_last': binding.clear_last, 'clear_observations': binding.clear_count,
        'available_window': signals.effect_gate_window_active, 'new_token': view.added[0],
        'directional_commit': directional}
    return L.prepare(self.inventory, binding, view, raw, value)


def call(self: Any, caller: Any, binding: Any, view: Any, pipe: Any, side: str, proposal: Any) -> Any:
    value = self._history_current_base.call(self, caller, binding, view, pipe, side, proposal)
    values = caller.f_locals
    next_fall(values['sm'], values['signals'], binding)
    return value


def consumed_history(self: Any, value: Any, caller: Any) -> None:
    self._history_current_base.consumed_history(self, value, caller)
    P.committed(self.inventory, value)
    values, binding = caller.f_locals, value['binding']
    H.require(values['sm'].context is values['ctx'], 'current_context_identity')
    H.require(values['signals'].chain_event is None and values['signals'].is_match_active,
        'current_chain_or_inactive')
    value['current_merge'] = M.apply(values['sm'], values['signals'], binding.grid)
    binding.current = binding.grid


def current_counter(self: Any, original: Any) -> Any:
    value = self.bound_call(sys._getframe(1))
    if value is None or 'current_merge' not in value:
        return original
    H.require(Counter(original) == value['counter'], 'current_legacy_counter_input_changed')
    value['current_counter_calls'] = value.get('current_counter_calls', 0) + 1
    H.require(value['current_counter_calls'] == 1, 'current_counter_duplicate')
    state = P.committed(self.inventory, value)
    return Counter(dict(enumerate(state.counter, start=1)))


def current_recovered(self: Any) -> bool:
    value = self.bound_call(sys._getframe(1))
    return value is not None and 'current_merge' in value


def final_current(self: Any, value: Any, result: Any) -> Any:
    values, binding = value['frame'].f_locals, value['binding']
    H.require(result is not None and result.side == value['view'].scope[-1]
        and result.state.value == values['ctx'].state.value == 'stable', 'current_return_state')
    H.require(result.prob_board is not None and value.get('current_counter_calls') == 1,
        'current_probability_or_counter_missing')
    raw, _ = self.provider.raw(values['self'], values['side'], value['view'])
    proof = P.proof(self.inventory, value, raw=raw, sm=T.board_key(values['ctx'].confirmed_board),
        returned=T.board_key(result.confirmed_board), probability=T.board_key(result.prob_board.to_max_likelihood_board()))
    state = P.recover(self.inventory, value, proof)
    H.require(state.current.grid == binding.current, 'current_final_binding')
    return proof


def finish(self: Any, caller: Any, error: Any = None) -> None:
    value = self.calls.get(id(caller))
    if value is not None and error is None and 'current_merge' in value:
        try:
            self.provider.after_history_step(value, caller)
            value['current_proof'] = final_current(self, value, self._current_returns.get(id(caller)))
        except BaseException as exc:
            self.sticky_error = repr(exc)
            self._history_current_base.finish(self, caller, exc)
            raise
    self._history_current_base.finish(self, caller, error)
    if value is not None and 'current_proof' in value and error is None:
        self.records[-1].update(current_permission=True, current_proof=value['current_proof'],
            physical_current_certified=False)


def make_type(base: type) -> type:
    return type('CurrentHistoryController', (base,), {'_history_current_base': base,
        'prepared': prepared, 'call': call, 'consumed_history': consumed_history,
        'current_counter': current_counter, 'current_recovered': current_recovered, 'finish': finish})


def attach_returns(stack: Any, journal: Any, controller: Any) -> None:
    original = journal.complete_step
    controller._current_returns = {}
    def completed(item: Any, result: Any, error: Any, profile: Any) -> Any:
        frame = item['frame']
        if frame is not None:
            controller._current_returns[id(frame)] = result
        try:
            return original(item, result, error, profile)
        finally:
            if frame is not None:
                controller._current_returns.pop(id(frame), None)
    journal.complete_step = completed
    stack.callback(setattr, journal, 'complete_step', original)
