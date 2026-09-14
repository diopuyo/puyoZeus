"""原要求の拒否分類と元Scheduledの保存/終了条件の限定CPU対照。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import compatibility as C

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1'
sys.path.insert(0, str(BASE))
from test_scheduled_session import module, context, update
import capture_schedule as S
import selection as SELECT


def replace(stack: Any, owner: Any, key: str, value: Any) -> None:
    before = getattr(owner, key)
    stack.callback(setattr, owner, key, before)
    setattr(owner, key, value)


def producer(frame: int, current: bool = True) -> dict:
    return dict(identity=dict(source_id='cpu', run_id='cpu'), last_frame=frame,
        metadata=[dict(frame_idx=frame, side=s) for s in C.SIDES],
        stable_snapshots=[dict(frame_idx=frame if current else frame-2, side=s) for s in C.SIDES])


def test_actual_request_missing_and_unchanged() -> None:
    path = ROOT.parent / 'video38_second_prefix_candidate_v37/JOINT_EVENTS.jsonl.request0.json'
    raw = path.read_bytes()
    value = json.loads(raw)['producer']
    assert not C.producer_ready(value, 36832)
    assert path.read_bytes() == raw


@pytest.mark.parametrize('case', ['ready', 'old', 'empty', 'future', 'side', 'metadata', 'clock'])
def test_producer_contract(case: str) -> None:
    value = producer(100, case != 'old')
    if case == 'empty': value['stable_snapshots'] = []
    if case == 'future': value['stable_snapshots'][0]['frame_idx'] = 102
    if case == 'side': value['stable_snapshots'][-1]['side'] = '1P'
    if case == 'metadata': value['metadata'][0]['frame_idx'] = 98
    if case == 'clock': value['last_frame'] = 98
    if case in ('future', 'side', 'metadata', 'clock'):
        with pytest.raises(ValueError): C.producer_ready(value, 100)
    else:
        assert C.producer_ready(value, 100) is (case == 'ready')


@pytest.mark.parametrize('ready', [True, False])
def test_original_scheduled_save_close_and_restore(module: Any, monkeypatch: Any, tmp_path: Path, ready: bool) -> None:
    monkeypatch.setattr(S, 'END', 35374)
    monkeypatch.setattr(S, 'EARLIEST', (35370, 35372))
    monkeypatch.setattr(S, 'MIN_GAP', 2)
    monkeypatch.setattr(module.E, 'reasons', lambda owner, frame: ())
    old_completed, old_close = module.Session.completed, module.Session.close
    with ExitStack() as outer:
        SELECT.bind(module, outer, replace)
        with ExitStack() as inner:
            ctx = context(tmp_path)
            ctx['state']['joint_producer_capture'] = N(identity=dict(source_id='cpu', run_id='cpu'),
                snapshot=lambda: producer(value.capture()['frame'], ready))
            value = module.Session(inner, ctx, None, None, None, 'ok', S.EARLIEST)
            for frame in range(35368, S.END+1, 2): update(value, frame)
    assert module.Session.completed is old_completed and module.Session.close is old_close
    result = json.loads((tmp_path/'LEGACY_M1_COMPATIBILITY.json').read_bytes())
    assert result['legacy_m1_complete'] is ready and result['quality_gate_clear'] is False
    assert len(value.saved) == (2 if ready else 0)
    assert value.evaluation_flags.closed and value.schedule_stream.closed
    assert result['old_finish_error'] == (None if ready else C.INCOMPLETE)


@pytest.mark.parametrize('case', ['pending', 'early', 'wrong_error'])
def test_noncompatibility_failure_not_suppressed(tmp_path: Path, case: str) -> None:
    state = S.Schedule(last=S.END if case != 'early' else S.END-2,
        pending=S.END if case == 'pending' else None)
    schedule = N(**vars(S))
    if case == 'wrong_error':
        schedule.finish = lambda value: (_ for _ in ()).throw(ValueError('different_failure'))
    with pytest.raises(ValueError): C.finished(schedule, state, tmp_path)
    assert not (tmp_path/'LEGACY_M1_COMPATIBILITY.json').exists()
