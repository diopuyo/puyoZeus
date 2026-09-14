"""同call scope修正版で原連続着地分岐を再検査する。J登録は人工。"""
from __future__ import annotations
from contextlib import ExitStack
from typing import Any
import pytest
import test_quarantine as OLD
import quarantine_v2 as Q


def test_original_landing_block_with_validated_call_scope() -> None:
    ns, recovery, raw = OLD.setup()
    code = OLD.block()
    recovery.journal.codes.add(code)
    scope = dict(frame_idx=ns['frame_idx'], time_sec=ns['time_sec'], side='1P')
    def scoped(pipe: Any, side: str, frame: int, clock: float) -> dict[str, Any]:
        assert pipe is recovery.pipe and (side, frame, clock) == ('1P', 35172, 35172 / 60)
        return scope
    recovery.journal.scope = scoped
    recovery.journal.epoch = lambda pipe, side: recovery.pending['epoch']
    def register(frame: Any) -> None:
        recovery.journal.active = dict(frame=frame, pipe=recovery.pipe, token='fixture-J',
            scope=scope, epoch=recovery.pending['epoch'])
    ns['REGISTER'] = register
    original = ns['infer_placement']
    counter = dict(recovery.pipe._tsumo_count_1p)
    other = recovery.pipe._sm_2p.context
    with ExitStack() as stack:
        guard = Q.install(stack, recovery, ns)
        exec(code, ns)
        assert ns['inferred_landing'] is None and ns['side_prob_board'] is None
        assert ns['ctx'].confirmed_board.to_dict()['grid'] == raw
        assert guard.rows[0]['original_pair'] == (4, 5)
    assert ns['infer_placement'] is original
    assert dict(recovery.pipe._tsumo_count_1p) == counter
    assert recovery.pipe._sm_2p.context is other
