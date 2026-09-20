"""CPU対象入力が実SMに届く値を、元profileを保って読むだけで保存。"""
from __future__ import annotations
import json
import sys
from typing import Any

NAMES = frozenset(('update', '_apply_transition', '_update_within_current_state', 'detect'))


def board(value: Any) -> Any:
    return None if value is None else value._grid.tolist()


def infer_call(frame: Any, pipe: Any) -> Any:
    chain, caller = [], frame
    selected = False
    for _ in range(8):
        if caller is None: break
        selected |= caller.f_locals.get('self') is pipe
        bound = caller.f_globals.get('infer_placement')
        chain.append(dict(name=caller.f_code.co_name, file=caller.f_code.co_filename,
            namespace=id(caller.f_globals), side=caller.f_locals.get('side'),
            bound_type=type(bound).__name__, bound_owner=type(getattr(bound, '__self__', None)).__name__))
        caller = caller.f_back
    return dict(method='infer_placement', pair=frame.f_locals.get('next_pair'), callers=chain) if selected else None


def install(stack: Any, pipe: Any, output: Any) -> None:
    original = sys.getprofile()
    machine, rows = pipe._sm_1p, []
    def profile(frame: Any, event: str, result: Any) -> None:
        if original is not None:
            original(frame, event, result)
        local, name = frame.f_locals, frame.f_code.co_name
        if name == 'infer_placement' and event == 'call':
            row = infer_call(frame, pipe)
            if row is not None: rows.append(row)
        if event not in ('call', 'return') or name not in NAMES:
            return
        selected = local.get('self') is machine or (name == 'detect' and local.get('ctx') is machine.context)
        if not selected:
            return
        signals = local.get('signals')
        rows.append(dict(method=name, event=event, frame=machine.context.frame_idx,
            state=machine.context.state.value, confirmed=board(machine.context.confirmed_board),
            cnn=None if signals is None else board(signals.cnn_board),
            hsv=None if signals is None else board(getattr(signals, 'hsv_board', None)),
            match_just_started=None if signals is None else signals.match_just_started,
            detector=type(local.get('self')).__name__, result_state=getattr(result, 'value', None)))
    def close() -> None:
        owned = sys.getprofile() is profile
        if owned:
            sys.setprofile(original)
        with (output / 'TARGET_SM_TRACE.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(rows=rows, restored=owned, artificial_input=True), stream, ensure_ascii=False)
        assert owned, 'target_profile_foreign_binding'
    stack.callback(close)
    sys.setprofile(profile)
