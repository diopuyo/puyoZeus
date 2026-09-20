"""借用Receiverを再登録せず、原O.complete_stepだけを解除して通常Oへ交代する。"""
from contextlib import ExitStack
from copy import deepcopy
from typing import Any
import second_observation as O
from test_second_observation import context, generated, live, saved


def test_original_observation_reinstalls_after_early_scope(context: Any) -> None:
    value = context
    original = value.journal.complete_step
    state_keys = set(value.state)
    receiver = value.state['postcommit_current_receiver']
    with ExitStack() as first:
        early = O.install(first, value.journal, value.state)
        generated(value.journal, value.pipe, value.result, value.row)
        assert early.latest is not None and value.journal.count == 1
    assert early.closed and value.journal.complete_step == original
    assert set(value.state) == state_keys and value.state['postcommit_current_receiver'] is receiver
    frame, clock = value.row['frame_idx'] + 2, value.row['time_sec'] + 2 / 60
    scope = value.journal.scope()
    scope.update(frame_idx=frame, time_sec=clock)
    value.journal.scope = lambda *args: deepcopy(scope)
    value.pipe._sm_2p.context.frame_idx = frame
    pb = value.state['hidden_probability_observer'].rows[-1]
    pb.update(frame_idx=frame, time_sec=clock)
    pb['probability']['capture_scope'].update(frame_idx=frame, time_sec=clock)
    with ExitStack() as second:
        late = O.install(second, value.journal, value.state)
        generated(value.journal, value.pipe, value.result, dict(value.row, frame_idx=frame, time_sec=clock))
        assert late.latest['frame'] == frame and value.journal.count == 2
    assert late.closed and value.journal.complete_step == original
    assert set(value.state) == state_keys and value.state['postcommit_current_receiver'] is receiver
