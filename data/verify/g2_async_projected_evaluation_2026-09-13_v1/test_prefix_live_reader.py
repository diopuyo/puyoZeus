"""実ledger/型を使う人工所有fixture。実原J/動画接続の合格には流用しない。"""
from contextlib import ExitStack
from types import SimpleNamespace as N
from typing import Any
import pytest
import prefix_live_adapter as A
import prefix_live_reader as R
import test_stable_evidence as E

source = E.source


def fixture(source: Any, output: Any, stack: Any) -> tuple:
    mode, item, qualification = E.synthetic_owner(source, output)
    live = A.Live(stack, source[0].mode)
    live.recorder.record(mode, item, qualification)
    c, ledger = mode.connection, mode.arrival_ledger
    c.recovery.error = None
    c.recovery.pipe._sm_1p = item['frame'].f_locals['sm']
    lane = A.N.Lane.__new__(A.N.Lane)  # 所有検査専用。物理初期化済みとはしない。
    lane.parts, lane.current = live.parts, object()
    lane.observations = [dict(frame=ledger.clock, evidence_key=item['token'])]
    live.lanes[id(mode)] = (mode, lane)
    c.registry = N(current=lambda binding: lane.current)
    session = N(journal=c.recovery.journal, pipe=c.recovery.pipe,
                factory=c.recovery.factory, state=mode.state)
    step = item['scope'] | dict(software_reset=item['epoch'], generation_after=item['scope']['generation'])
    return live, mode, session, step, lane


def test_last_qualified_frame_can_be_older_than_cutoff(source: Any, tmp_path: Any) -> None:
    with ExitStack() as stack:
        live, mode, session, step, lane = fixture(source, tmp_path, stack)
        before = (dict(live.recorder.last), dict(live.recorder.owners), mode.arrival_ledger)
        assert R.owner(live, A, mode, session, step) is lane
        next_frame = step['frame_idx'] + 2
        mode.arrival_ledger = live.module.L.advance_clock(mode.arrival_ledger, next_frame)
        step = step | dict(frame_idx=next_frame, time_sec=next_frame / 60)
        assert R.owner(live, A, mode, session, step) is lane
        assert live.recorder.last == before[0] and live.recorder.owners == before[1]
        assert lane.observations[-1]['frame'] < next_frame


@pytest.mark.parametrize('fault', ['factory', 'binding', 'scope', 'clock', 'qualified_owner',
    'current', 'token', 'active', 'recovery_error'])
def test_owner_fault_is_not_a_silent_hold(source: Any, tmp_path: Any, fault: str) -> None:
    with ExitStack() as stack:
        live, mode, session, step, lane = fixture(source, tmp_path, stack)
        if fault == 'factory': session.factory = object()
        elif fault == 'binding': mode.connection.binding = N(scope=mode.arrival_ledger.scope)
        elif fault == 'scope': step = step | dict(software_reset=step['software_reset'] + 1)
        elif fault == 'clock': step = step | dict(frame_idx=step['frame_idx'] + 2)
        elif fault == 'qualified_owner': live.recorder.owners[id(mode)] = (mode,)
        elif fault == 'current': mode.connection.registry.current = lambda binding: object()
        elif fault == 'token': live.recorder.last[id(mode)] = (step['frame_idx'], 'foreign')
        elif fault == 'active': live.active[id(mode)] = (mode,)
        elif fault == 'recovery_error': mode.connection.recovery.error = 'original failure'
        with pytest.raises(ValueError):
            R.owner(live, A, mode, session, step)
