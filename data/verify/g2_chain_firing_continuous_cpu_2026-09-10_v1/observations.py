"""元 code identity に限定した読取と、原シミュレータ対照。"""
from __future__ import annotations
from contextlib import ExitStack
import sys
from typing import Any
import artificial_inputs as I


def board_value(board: Any) -> Any:
    return None if board is None else board.to_dict()['grid']


def event_value(event: Any) -> Any:
    if event is None:
        return None
    return dict(chain_count=event.chain_count, mechanism=event.mechanism,
        trigger_sec=event.trigger_sec, before_board=board_value(event.before_board),
        total_erased=event.total_erased, total_score=event.total_score)


def control() -> dict[str, Any]:
    from src.board import Board
    from src.chain import ChainSimulator
    from src.production_config import GHOST_CHAIN_RULE_ENABLED
    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    before, fired = I.board_at(Board(), I.PP_END), I.board_at(Board(), I.FIRE)
    old, new = sim.simulate(before), sim.simulate(fired)
    assert old.chain_count == 0 and new.chain_count == 1 and new.total_erased == 5
    assert new.final_board.count_puyos() == 3
    return dict(before=board_value(before), firing=board_value(fired), final=board_value(new.final_board),
        before_chain=old.chain_count, firing_chain=new.chain_count, erased=new.total_erased,
        original_simulator=True, artificial_input=True, full_positive=False)


def profiling(stack: ExitStack, pipe: Any, factory: Any, clock: Any) -> list[Any]:
    rows, previous = [], sys.getprofile()
    codes = {pipe._apply_chain_formula_early_fire.__func__.__code__: 'formula_fire',
        factory.provider.no_origin.__func__.__code__: 'no_origin',
        factory.controller.observed.__func__.__code__: 'observed'}
    def profile(frame: Any, event: str, result: Any) -> None:
        if previous is not None:
            previous(frame, event, result)
        if frame.f_code not in codes or event not in ('call', 'return'):
            return
        values = frame.f_locals
        value = dict(frame=clock['frame'], body=codes[frame.f_code], event=event,
            side=values.get('side'), active=event_value(pipe._active_chain_1p))
        if codes[frame.f_code] == 'no_origin' and event == 'return':
            value['returned'] = result
        if codes[frame.f_code] == 'formula_fire':
            value['argument_before'] = board_value(values.get('prev_confirmed'))
        rows.append(value)
    sys.setprofile(profile)
    stack.callback(sys.setprofile, previous)
    return rows


def snapshot(target: Any, state: Any, factory: Any, pipe: Any, frame: int) -> dict[str, Any]:
    row = target.snapshot(state, factory, pipe, frame)
    row.update(active=event_value(pipe._active_chain_1p), first_move=pipe._first_move_sec_1p,
        formula_consec=pipe._formula_consec_1p, prev_confirmed=board_value(pipe._prev_confirmed_1p))
    return row


def flags(pipe: Any) -> dict[str, Any]:
    names = ('_enable_chain_formula_detection', '_enable_formula_value_read',
        '_enable_chain_formula_read_verify', '_enable_chain_exit_next_signal',
        '_enable_game_event_chain_exit')
    result = {name: getattr(pipe, name) for name in names}
    assert result['_enable_chain_formula_detection'] and result['_enable_formula_value_read']
    assert result['_enable_chain_formula_read_verify']
    return result
