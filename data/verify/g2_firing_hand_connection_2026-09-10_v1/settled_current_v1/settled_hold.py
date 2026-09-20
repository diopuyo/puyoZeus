"""原SMのSTABLE選択だけに精算済みrawの限定出口を接続する。"""
from __future__ import annotations
from contextlib import contextmanager
import sys
from typing import Any, Iterator
import settled_merge as M


def eligible(control: Any, binding: Any, signals: Any, view: Any, pipe: Any) -> Any:
    ticket = getattr(binding, 'firing_ticket', None)
    if ticket is None or not getattr(ticket, 'settled', False) or getattr(ticket, 'current_published', False):
        return None
    if signals.chain_event is not None or signals.effect_gate_window_active is not False:
        return None
    state = binding.owner.state
    assert ticket.binding is binding and ticket.scope == binding.scope == view.scope
    assert ticket.origin in state.origins and ticket.origin.origin_id in state.consumed_ids and not state.debts
    assert state.action == ticket.origin.action and binding.next_token is None, 'settled_current_action_advanced'
    assert state.counter == control.inventory.S.color_counts(ticket.origin.predicted_final)
    raw, captured = control.provider.raw(pipe, view.scope[-1], view)
    assert captured['captured_frame'] == view.frame
    if raw != ticket.origin.predicted_final or raw != M.key(signals.cnn_board):
        return None
    return ticket


def wrappers(control: Any, sm: Any, signals: Any, binding: Any, module: Any,
             call: Any, apply: Any, within: Any) -> tuple[Any, Any]:
    pipe, view = call['frame'].f_locals['self'], call['view']
    env = sys.modules[type(sm).__module__]
    def check(current: Any, observed: Any, source: Any) -> None:
        assert current is sm and observed is signals
        assert (module.T.original_code(source.f_code, 'update') if source.f_code.co_name == 'update'
                else source.f_code is apply.__code__)
    def applied(current: Any, target: Any, observed: Any) -> None:
        check(current, observed, sys._getframe(1))
        ticket = eligible(control, binding, observed, view, pipe) if target.value == 'stable' else None
        if ticket is not None and current.context.state.value in module.T.ACTIVE_STATES:
            assert not call.get('current_merge') and not call['consumed'] and call['prepared'] is None
            call['current_merge'] = M.apply(current, observed, ticket.origin.predicted_final, apply)
            binding.current = M.key(current.context.confirmed_board)
            call['settlement_current_ticket'] = ticket
        elif target.value == 'stable' and module.should_fall(current, observed, binding):
            apply(current, env.BoardState.TSUMO_FALL, observed)
        elif target.value == 'stable' and current.context.state.value in module.T.ACTIVE_STATES:
            within(current, observed)
        else:
            apply(current, target, observed)
        assert M.key(current.context.confirmed_board) == binding.current
    def updated(current: Any, observed: Any) -> None:
        check(current, observed, sys._getframe(1))
        if module.should_fall(current, observed, binding):
            apply(current, env.BoardState.TSUMO_FALL, observed)
        else:
            within(current, observed)
    return applied, updated


@contextmanager
def installed(control: Any, sm: Any, signals: Any, binding: Any, module: Any) -> Iterator[None]:
    # contextlib経由なので実callは原Controllerの保有辞書で同signalsへ束縛する。
    calls = [v for v in control.calls.values() if v['binding'] is binding
             and v['frame'].f_locals.get('signals') is signals]
    assert len(calls) == 1
    call, cls = calls[0], type(sm)
    apply, within = cls._apply_transition, cls._update_within_current_state
    env = sys.modules[cls.__module__]
    assert module.T.original_code(apply.__code__, '_apply_transition') and module.original_within(within.__code__)
    assert apply.__globals__ is within.__globals__ is vars(env)
    cls._apply_transition, cls._update_within_current_state = wrappers(control, sm, signals, binding, module, call, apply, within)
    try:
        yield
    finally:
        cls._apply_transition, cls._update_within_current_state = apply, within
