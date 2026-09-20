"""原J完了直前のflag捕捉→completed→元captureを結合。pipe/SM/保存ACKは人工。"""
from __future__ import annotations

from contextlib import ExitStack
import sys
from types import SimpleNamespace as N
from typing import Any

import pytest
from test_session_capture_connection import saved, live, session
import test_journal_witness as T
import live_binding_v3 as V
import serialization as S
import evaluation_flags as F


def emit_step(journal: Any, step: dict, *, effect: bool = False) -> None:
    pipe, side, frame_idx = journal.pipe, step['side'], step['frame_idx']
    sm = getattr(pipe, '_sm_' + side.lower())
    signals = N(is_match_active=True, effect_gate_window_active=effect)
    scope = {k: step[k] for k in ('source_id', 'run_id', 'frame_idx', 'time_sec', 'side', 'pipe_object_id', 'generation')}
    board = sm.context.confirmed_board
    result = N(confirmed_board=board, inferred_board=board, state=N(name='STABLE', value='stable'))
    item = dict(scope=scope, token=step['token'], epoch=step['software_reset'], pipe=pipe,
                frame=sys._getframe(), events=[], return_line=None)
    journal.complete_step(item, result, None, sys.getprofile())
    assert item['frame'] is None


@pytest.fixture
def connected(live: Any, request: Any) -> Any:
    journal = T.recorder()
    vars(journal).update(vars(live.journal))
    journal.pipe, journal.codes = live.pipe, {emit_step.__code__}
    generations = {r['side']: T.Generation(**r['generation']) for r in live.steps}
    journal.tracker = N(generation=lambda side: generations[side])
    modes = []
    for binding, step in zip(live.bindings, live.steps, strict=True):
        value = live.registry.current(binding)
        connection = N(registry=live.registry, binding=binding)
        native = N(connection=connection, pending=[], last_frame=value.frame, seen_calls={step['token']})
        modes.append(N(connection=connection, native=native, error=None, activation={'frame': value.frame},
                       applied=[dict(applied_frame=value.frame, state=S.encode(value))]))
        setattr(live.pipe, '_landing_grace_' + step['side'].lower(), None)
    with ExitStack() as stack:
        witness = V.V2.W.install(stack, journal)
        flags = F.install(stack, journal)
        for step in live.steps:
            emit_step(journal, step, effect=request.param == 'effect' and step['side'] == '2P')
        assert journal.count == 2 and set(flags.latest) == {'1P', '2P'}
        yield N(base=live, journal=journal, witness=witness, modes=modes, flags=flags)
    assert flags.closed and flags.error is None and witness.closed


@pytest.mark.parametrize('connected', ['ready', 'effect'], indirect=True)
def test_flags_to_capture_without_manual_flags(session: Any, connected: Any) -> None:
    value = session.value
    value.evaluation_flags = connected.flags
    value.completed(35370)
    expected = not connected.flags.latest['2P']['effect_window']
    assert len(session.calls) == len(value.saved) == int(expected)
    assert value.schedule.accepted == ((35370,) if expected else ())
