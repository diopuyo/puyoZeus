"""次手遷移を元SM.update内で一度選択し、grace残留をpop前に拒否する。"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import sys
from types import CodeType
from typing import Any, Iterator
import history_state as H
import transaction as T
import _current_history_controller as V1


@lru_cache(maxsize=1)
def original_within(code: CodeType) -> bool:
    return T.original_code(code, '_update_within_current_state')


def should_fall(sm: Any, signals: Any, binding: Any) -> bool:
    return (signals.is_match_active and sm.context.state.value == 'stable'
        and binding.owner.state.current is not None and binding.next_token is not None)


@contextmanager
def hold_transition(self: Any, sm: Any, signals: Any, binding: Any) -> Iterator[None]:
    cls, module = type(sm), sys.modules[type(sm).__module__]
    apply, within = cls._apply_transition, cls._update_within_current_state
    H.require(T.original_code(apply.__code__, '_apply_transition') and original_within(within.__code__),
        'current_v2_original_SM')
    H.require(apply.__globals__ is vars(module) and within.__globals__ is vars(module),
        'current_v3_SM_globals')
    def check(current: Any, observed: Any, caller: Any) -> None:
        H.require(current is sm and observed is signals
            and (T.original_code(caller.f_code, 'update') if caller.f_code.co_name == 'update'
                 else caller.f_code is apply.__code__), 'current_v2_actual_SM_caller')
    def applied(current: Any, target: Any, observed: Any) -> None:
        check(current, observed, sys._getframe(1))
        if target.value == 'stable' and should_fall(current, observed, binding):
            apply(current, module.BoardState.TSUMO_FALL, observed)
        elif target.value == 'stable' and current.context.state.value in T.ACTIVE_STATES:
            within(current, observed)
        else:
            apply(current, target, observed)
        H.require(T.board_key(current.context.confirmed_board) == binding.current, 'current_v2_hold_changed_grid')
    def updated(current: Any, observed: Any) -> None:
        check(current, observed, sys._getframe(1))
        if should_fall(current, observed, binding):
            apply(current, module.BoardState.TSUMO_FALL, observed)
        else:
            within(current, observed)
    cls._apply_transition, cls._update_within_current_state = applied, updated
    try:
        yield
    finally:
        cls._apply_transition, cls._update_within_current_state = apply, within


def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
    value = V1.prepared(self, binding, item, sm, raw, signals, view)
    if value is not None:
        caller = sys._getframe(1)
        H.require(caller.f_code is self._parts.C.Controller.update.__code__
            and caller.f_locals['signals'] is signals, 'current_v2_prepare_caller')
        pipe, side = caller.f_locals['pipe'], view.scope[-1]
        key = '_landing_grace_' + side.lower()
        H.require(hasattr(pipe, key), 'current_grace_unobserved')
        grace = getattr(pipe, key)
        H.require(grace is None or view.clock >= grace[2], 'current_active_grace_before_pop')
    return value


def make_type(base: type) -> type:
    first = V1.make_type(base)
    return type('CurrentHistoryControllerV3', (first,), {'call': base.call,
        'hold_transition': hold_transition, 'prepared': prepared})
