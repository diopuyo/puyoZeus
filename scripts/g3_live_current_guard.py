"""G3現在出口に実SMの同frame資格を追加する。原結果は変更しない。"""
from __future__ import annotations

from typing import Any, Callable
from scripts import g3_current_exit as C

KEY = 'g3_live_current_guard'


def reasons(pipe: Any, result: Any, frame: int) -> list[str]:
    """reset後の旧STABLE結果を、実SMと可視盤面の照合で欠測にする。"""
    failures: list[str] = []
    contexts: list[Any] = []
    for side, name in C.SIDES:
        ctx = getattr(getattr(pipe, '_sm_' + side.lower(), None), 'context', None)
        contexts.append(ctx)
        if ctx is None:
            failures.append(side + '_LIVE_CONTEXT_MISSING')
            continue
        if ctx.frame_idx != frame:
            failures.append(side + '_LIVE_CONTEXT_CLOCK')
        if ctx.state.value != 'stable' or ctx.confirmed_board is None:
            failures.append(side + '_LIVE_NOT_CONFIRMED_STABLE')
            continue
        current = ctx.confirmed_board._grid
        C.A.require(current.shape == (C.ROWS, C.COLUMNS), 'live_current_board_shape')
        published = getattr(result, name).confirmed_board
        if published is not None and not (current[1:] == published._grid[1:]).all():
            failures.append(side + '_LIVE_VISIBLE_BOARD_MISMATCH')
    if contexts[0] is not None and contexts[0] is contexts[1]:
        failures.append('LIVE_CONTEXT_SIDE_ALIAS')
    return failures


def wrap(original: Callable) -> Callable:
    """元資格検査を一回実行し、不成立時は可視盤面も返さない。"""
    def selection(observer: Any, pipe: Any, result: Any, frame: int, clock: float) -> dict:
        row = original(observer, pipe, result, frame, clock)
        failures = reasons(pipe, result, frame)
        row['live_context_checked'] = True
        row['live_context_eligible'] = not failures
        if failures:
            row.update(status='HOLD', reasons=row['reasons'] + failures, visible_confirmed_boards=None)
        return row
    return selection


def install(stack: Any, state: dict, replace: Callable) -> None:
    """既存所有stackへ束縛し、元selectionの参照復元を委ねる。"""
    C.A.require(KEY not in state, 'live_current_duplicate')
    C.A.require(C.KEY not in state or state[C.KEY].rows == 0, 'live_current_late_install')
    state[KEY] = dict(quality_gate_clear=False, physical_identity_certified=False)
    replace(stack, C, 'selection', wrap(C.selection))
