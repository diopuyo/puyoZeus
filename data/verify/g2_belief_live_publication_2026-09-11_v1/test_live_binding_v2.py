"""原complete発行→修復binder。生pipe/両side反映機構は人工fixture。"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
from test_live_binding import saved, live
import test_journal_witness as T
import live_binding_v2 as V
import serialization as S


@pytest.fixture
def connected(live: Any, saved: Any) -> Any:
    journal = T.recorder()
    for key, value in vars(live.journal).items():
        setattr(journal, key, value)
    generations = {step['side']: T.Generation(**step['generation']) for step in live.steps}
    journal.tracker = N(generation=lambda side: generations[side])
    modes = []
    for binding, step in zip(live.bindings, live.steps, strict=True):
        value = live.registry.current(binding)
        connection = N(registry=live.registry, binding=binding)
        native = N(connection=connection, pending=[], last_frame=value.frame, seen_calls={step['token']})
        modes.append(N(connection=connection, native=native, error=None, activation={'frame':value.frame},
                       applied=[dict(applied_frame=value.frame, state=S.encode(value))]))
    with ExitStack() as stack:
        witness = V.W.install(stack, journal)
        for step in live.steps:
            scope = {key:step[key] for key in ('frame_idx','time_sec','side','source_id','run_id','pipe_object_id','generation')}
            item = dict(scope=scope, token=step['token'], epoch=step['software_reset'], events=[], return_line=None, frame=None)
            journal.complete_step(item, None, None, sys.getprofile())
        capture = lambda: V.current(live.rec, live.factory, live.pipe, journal, live.registry,
            live.bindings, saved[3], witness, tuple(modes))
        yield N(base=live, capture=capture, journal=journal, witness=witness, modes=modes)


def test_original_emission_to_bound(connected: Any) -> None:
    bound = connected.capture()
    assert bound.tokens == ('step:598', 'step:599')
    assert len(bound.values[0].worlds) == 49
    # callerの旧stepsコピーを改変しても、原発行内容が評価へ使われる。
    connected.base.steps[0]['exception'] = 'caller_fake_error'
    assert connected.capture().tokens == bound.tokens


def test_unapplied_registry_now_rejected(connected: Any) -> None:
    live = connected.base
    value = live.registry.current(live.bindings[0])
    live.registry._states[value.scope] = replace(value, frame=value.frame-2)
    with pytest.raises(ValueError, match='reflection_frame'):
        connected.capture()


def test_unprocessed_second_side_rejected(connected: Any) -> None:
    connected.modes[1].native.seen_calls.clear()
    with pytest.raises(ValueError, match='reflection_J_not_observed'):
        connected.capture()


def test_other_witness_rejected(connected: Any) -> None:
    connected.witness.journal = T.recorder()
    with pytest.raises(ValueError, match='witness_owner'):
        connected.capture()


def test_original_error_cannot_be_hidden_by_caller_copy(connected: Any) -> None:
    live = connected.base
    step = live.steps[0]
    scope = {key:step[key] for key in ('frame_idx','time_sec','side','source_id','run_id','pipe_object_id','generation')}
    item = dict(scope=scope, token=step['token'], epoch=step['software_reset'], events=[], return_line=None, frame=None)
    connected.journal.complete_step(item, None, ValueError('actual_J_failed'), sys.getprofile())
    assert live.capture().tokens == ('step:598', 'step:599')  # 旧入口はcallerコピーだけを読む。
    with pytest.raises(ValueError, match='J_completion'):
        connected.capture()
