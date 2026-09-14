"""原prepare/原geometryで世代退役・初期化・最初のseq1と非干渉を検査する。"""
from __future__ import annotations
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
from typing import Any
import pytest
import retirement as R
import test_original_epoch as F


def objects() -> tuple[Any, Any, Any, list[Any]]:
    adapter, runtime, view = F.objects(0, 'artificial-hidden-segment:0')
    runtime.histories['1P'] = F.native.History(0)
    rows: list[Any] = []
    adapter.controller.active = None
    adapter.controller.instances = {id(runtime.pipe): runtime}
    adapter.controller.rec = N(emit=rows.append)
    adapter.states[(id(runtime.pipe), '2P')] = F.O.empty_state(0, 'other-segment')
    context = N(pipe=runtime.pipe, factory=N(provider=N(journal=N(controller=adapter.controller))),
                state={'directional_next_runtime': {'adapter': adapter}})
    return context, adapter, runtime, rows


def test_retirement_preserves_other_and_uses_original_provider() -> None:
    context, adapter, runtime, rows = objects()
    old_candidate = F.O.MotionCandidate(5, 34924, 34924, 'artificial-desync:5')
    adapter.states[(id(runtime.pipe), '1P')].update(pending=old_candidate, last_candidate=old_candidate,
                                                  baseline_dnext=(4, 4), armed_at=34922)
    proof = R.Retirement(context)
    runtime.histories['1P'] = F.native.History(1)
    new = runtime.histories['1P']
    report = proof.finish(F.FRAME)
    assert runtime.histories['1P'] is new and report['other_sides_unchanged']
    assert rows == [] and report['emitted'] is False
    path = Path(__file__).resolve().parent.parent / 'g2_reset_recovery_candidate_2026-09-10_v1/reset_inputs.py'
    spec = importlib.util.spec_from_file_location('_g2_retirement_original_inputs', path)
    inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inputs)
    shift = F.FRAME - inputs.RESET
    values = dict(vars(inputs), FIRST=inputs.FIRST + shift, SECOND=inputs.SECOND + shift,
                  QUIET=tuple(frame + shift for frame in inputs.QUIET))
    geometry = FunctionType(inputs.geometry.__code__, values)
    key = (id(runtime.pipe), '1P')
    commits = []
    for frame in (F.FRAME, 34936, 34950, 34954):
        invocation = N(frame=frame, time_sec=frame / F.FPS, runtime=runtime,
                        main=N(p1=N(dnext_pair=(3, 3) if frame < 34950 else (2, 2))))
        view = geometry(F.O, invocation, '1P')
        assert view.segment_id == 'artificial-reset-segment:1'
        if view.candidate is not None: assert view.candidate.sequence_number == 1
        pair = (4, 5) if frame < 34950 else (3, 3)
        accepted, state, commit, reason = adapter.prepare(runtime, '1P', frame, pair, view, True)
        if commit: commits.append((frame, reason))
        adapter.states[key] = state
        runtime.histories['1P'] = F.native.History(1, accepted, frame)
    assert commits == [(34950, 'occurrence_committed')]
    assert report['old']['epoch'] == 0 and report['new']['epoch'] == 1


@pytest.mark.parametrize('fault', ['blocked', 'active', 'other_state', 'other_history', 'started'])
def test_failure_conditions(fault: str) -> None:
    context, adapter, runtime, _ = objects()
    key = (id(runtime.pipe), '1P')
    if fault == 'blocked': adapter.states[key]['blocked'] = 'frame_gap'
    if fault == 'active': adapter.controller.active = object()
    if fault in ('blocked', 'active'):
        with pytest.raises(ValueError): R.Retirement(context)
        assert runtime.histories['1P'].epoch == 0
        return
    proof = R.Retirement(context)
    runtime.histories['1P'] = F.native.History(1)
    if fault == 'other_state': adapter.states[(id(runtime.pipe), '2P')]['epoch'] = 1
    if fault == 'other_history': runtime.histories['2P'] = F.native.History(1)
    if fault == 'started': runtime.histories['1P'] = F.native.History(1, None, F.FRAME)
    with pytest.raises(ValueError): proof.finish(F.FRAME)


def test_retired_segment_rejected_and_absent_state_noop() -> None:
    context, adapter, runtime, rows = objects()
    proof = R.Retirement(context)
    runtime.histories['1P'] = F.native.History(1)
    proof.finish(F.FRAME)
    view = F.O.MotionObservation(N(main=N(p1=N(dnext_pair=(3, 3)))), '1P', 1,
        'artificial-hidden-segment:0', F.FRAME, F.FRAME / F.FPS, None, None)
    with pytest.raises(ValueError, match='retired_segment_reused'):
        adapter.prepare(runtime, '1P', F.FRAME, (4, 5), view, True)
    context, adapter, runtime, rows = objects()
    del adapter.states[(id(runtime.pipe), '1P')]
    proof = R.Retirement(context)
    runtime.histories['1P'] = F.native.History(1)
    assert proof.finish(F.FRAME)['absent_noop'] and rows == []
