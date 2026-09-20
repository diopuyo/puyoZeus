"""左右の実Native/物理数学を同一型で確認。J/frame/画像供給は人工。"""
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import test_normal_basis as F
parts = F.parts
ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope='module')
def modules(parts: Any) -> Any:
    imports = dict(journal_context=parts.context, serialization=parts.original.binding.S,
                   native_consumption=parts.original.mode.CORE.V1.N)
    retired = parts.loader('_normal_test_retired', ROOT / 'second_retirement.py', imports)
    tracking = parts.loader('_normal_test_tracking', ROOT / 'second_tracking.py', imports |
                            dict(second_basis=parts.basis, second_retirement=retired))
    physical = parts.loader('_normal_test_physical', ROOT / 'second_physical.py', imports |
                            dict(second_tracking=tracking))
    return N(tracking=tracking, physical=physical)


def setup(parts: Any, modules: Any) -> Any:
    f = F.fixture(parts)
    f.B, f.modes = parts.original.mode.B, []
    journal = f.witness.journal
    journal.errors, journal.active, journal.codes = [], None, {step.__code__}
    journal.scope = lambda pipe, side, frame, clock: dict(source_id='normal', run_id='fixture',
        side=side, frame_idx=frame, time_sec=clock, pipe_object_id=id(pipe), generation=dict(reset_epoch=0))
    journal.epoch = lambda pipe, side: 0
    provider = N(journal=journal, no_origin=lambda pipe, side, view: True,
                 raw=lambda pipe, side, view: (f.B.grid(getattr(pipe, '_sm_' + side.lower()).context.confirmed_board), None))
    for side, evidence in zip(('1P', '2P'), f.evidence):
        setattr(f.pipe, '_landing_grace_' + side.lower(), None)
        evidence.state = dict(postcommit_current_receiver=N(rec=N(
            side_value=lambda result: dict(state_value=result.state.value))))
        mode = modules.physical.Mode(evidence, f.witness, f.pipe, f.registry, f.factory, 100,
            parts.policy, N(Mode=parts.original.mode.BASE.Mode), provider, side=side)
        f.modes.append(mode)
    return f


def step(f: Any, mode: Any, side: str, frame_idx: int, events: list) -> Any:
    time_sec = frame_idx / 60
    sm = getattr(f.pipe, '_sm_' + side.lower())
    sm.context.frame_idx = frame_idx
    board = sm.context.confirmed_board
    signals = N(is_match_active=True, effect_gate_window_active=False, cnn_board=board)
    result = N(state=N(value='stable'), confirmed_board=board)
    journal = f.witness.journal
    item = dict(frame=sys._getframe(), pipe=f.pipe, scope=journal.scope(f.pipe, side, frame_idx, time_sec),
                epoch=0, token='step:' + side + ':' + str(frame_idx), events=events)
    return mode.observe(item, None, result)


def test_both_normal_sides_native_transition(parts: Any, modules: Any) -> None:
    f = setup(parts, modules)
    assert type(f.modes[0]) is type(f.modes[1]) is modules.physical.Mode
    for side, mode in zip(('1P', '2P'), f.modes):
        board = getattr(f.pipe, '_sm_' + side.lower()).context.confirmed_board
        board.set(12, 3, 1)
        board.set(11, 3, 2)
        token = 'normal:fixture:reset:0:' + side + ':enqueue:hand1'
        events = [dict(stage='fifo_before', accounting=dict(pending_tsumo=[[1, 2]]),
                       fifo_occurrence_tokens=[token]),
                  dict(stage='fifo_after', accounting=dict(pending_tsumo=[]),
                       committed=[1, 2], enqueue_occurrence_token=token)]
        row = step(f, mode, side, 12, events)
        assert row['physical_transition_applied'] and not mode.native.pending
        assert len(mode.applied) == 1 and mode.connection.registry.current(mode.connection.binding).frame == 12
    assert f.modes[0].connection.binding is not f.modes[1].connection.binding
