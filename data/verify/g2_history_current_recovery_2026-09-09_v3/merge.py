"""原transitionの完成色追加だけを復元。履歴資格は呼出側が別に確認する。"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
from types import CodeType
from typing import Any
import transaction as T

ROWS, COLS = 13, 6
EMPTY, UNKNOWN = 0, 10
COLORS = frozenset(range(1, 6))
FILTER = '_filter_transition_new_cnn_for_burst_guard'


def fixed_filter(module: Any) -> Any:
    path = Path(module.__file__)
    data = path.read_bytes()
    T.require(hashlib.sha256(data).hexdigest() == T.P.SM_SHA, 'current_SM_source')
    code = compile(data, str(path), 'exec', dont_inherit=True)
    expected = next(c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == FILTER)
    actual = getattr(module, FILTER)
    T.require(actual.__code__ == expected and actual.__globals__ is vars(module), 'current_filter_source')
    return actual


def additions(sm: Any, signals: Any, target: Any) -> tuple[tuple[int, int, int], ...]:
    module = sys.modules[type(sm).__module__]
    old, raw = sm.context.confirmed_board, signals.cnn_board
    T.require(T.board_key(raw) == target, 'current_fresh_target')
    T.require(sm.context.state.value in T.ACTIVE_STATES, 'current_active_state')
    cells = []
    for r in range(ROWS):
        for c in range(COLS):
            a, b = old.get(r, c), raw.get(r, c)
            T.require(UNKNOWN not in (a, b), 'current_unknown_grid')
            if a != b:
                T.require(a == EMPTY and b in COLORS, 'current_not_color_addition')
                cells.append((r, c, b))
    T.require(bool(cells), 'current_no_addition')
    for board in (old, raw):
        private = board.copy()
        module._apply_gravity_filter(private)
        T.require(T.board_key(private) == T.board_key(board), 'current_unsupported_grid')
    return tuple(cells)


def transition(sm: Any, signals: Any, target: Any) -> dict[str, Any]:
    module, original = sys.modules[type(sm).__module__], type(sm)._apply_transition
    T.require(T.original_code(original.__code__, '_apply_transition'), 'current_original_transition')
    T.require(original.__globals__ is vars(module), 'current_transition_globals')
    expected_calls = int((sm._enable_transition_merge_guard and signals.effect_gate_window_active)
        or (sm._enable_ojama_column_stack_fix and sm.context.state is module.BoardState.OJAMA_FALL))
    cells, before = additions(sm, signals, target), T.board_key(sm.context.confirmed_board)
    original_filter, events = fixed_filter(module), []
    def filtered(old: Any, raw: Any, from_state: Any) -> Any:
        caller = sys._getframe(1)
        T.require(caller.f_code is original.__code__ and caller.f_locals['self'] is sm
            and caller.f_locals['signals'] is signals, 'current_filter_caller')
        T.require(not events and T.board_key(old) == before and T.board_key(raw) == target,
            'current_filter_binding')
        output = original_filter(old, raw, from_state).copy()
        for r, c, color in cells:
            T.require(output.get(r, c) in (UNKNOWN, color), 'current_filter_unexpected')
            output.set(r, c, color)
        events.append({'cells': cells})
        return output
    setattr(module, FILTER, filtered)
    try:
        original(sm, module.BoardState.STABLE, signals)
    finally:
        setattr(module, FILTER, original_filter)
    T.require(len(events) == expected_calls, 'current_filter_call_count')
    T.require(sm.context.state.value == 'stable' and T.board_key(sm.context.confirmed_board) == target,
        'current_original_merge_mismatch')
    return {'filter_calls': len(events), 'cells': cells, 'window': signals.effect_gate_window_active}


def apply(sm: Any, signals: Any, target: Any) -> dict[str, Any]:
    """元mergeを先にcopyで検証。実適用後の失敗はrollbackせず停止する。"""
    before = T.P.digest(sm.__dict__), T.P.digest(signals)
    shadow, observed = copy.deepcopy(sm), copy.deepcopy(signals)
    proposal = transition(shadow, observed, target)
    T.require(before == (T.P.digest(sm.__dict__), T.P.digest(signals)), 'current_preview_mutation')
    actual = transition(sm, signals, target)
    T.require(actual == proposal, 'current_preview_actual_disagree')
    return {'preview': proposal, 'actual': actual, 'physical_certified': False}
