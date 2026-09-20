"""元J/Registry/captureと新completedを結合。初期物理/資格flag/保存ACKは人工。"""
from __future__ import annotations

import importlib
import io
import json
from types import SimpleNamespace as N
from typing import Any

import pytest
from test_live_binding_v3 import saved, live, connected
import live_session as ORIGINAL
import capture_schedule as S


@pytest.fixture
def session(connected: Any, saved: Any, monkeypatch: Any, tmp_path: Any) -> Any:
    monkeypatch.setitem(__import__('sys').modules, 'old_session', ORIGINAL)
    module = importlib.import_module('scheduled_session')
    c, b = connected, connected.base
    value = module.Session.__new__(module.Session)
    first, second = c.modes
    first.closed = second.closed = False
    first.arrival_ledger = N(applied=[], arrivals=[], acknowledgements=[])
    value.state = dict(provisional_context_observer=b.rec, probabilistic_tracking_mode=first,
                       probabilistic_basis_connection=first.connection, output=tmp_path)
    value.mode, value.modes = second, c.modes
    value.factory, value.pipe, value.journal = b.factory, b.pipe, c.journal
    value.witness, value.contract, value.members = c.witness, saved[3], None
    value.error, value.saved = None, []
    value.schedule, value.schedule_rows, value.schedule_stream = S.Schedule(last=35368), 0, io.StringIO()
    flags = {}
    for step in c.witness.pair(35370):
        scope = {k: step[k] for k in ('source_id', 'run_id', 'frame_idx', 'time_sec', 'side', 'pipe_object_id', 'generation')}
        flags[step['side']] = dict(scope=scope, token=step['token'], software_reset=step['software_reset'],
            hold_reason=None, state='stable', match_active=True, effect_window=False,
            origin_present=False, grace_end=None)
    value.evaluation_flags = N(latest=flags, closed=False, error=None, journal=c.journal)
    calls = []

    def save(capture: Any, members: Any, path: Any, *, seed: int) -> dict:
        bound = capture()  # 元Session.capture→元V3→原J/実Registry照合。保存ACKだけ人工。
        calls.append(bound)
        return dict(frame=bound.frame, artificial_save_ack=True)

    monkeypatch.setattr(module, 'SAVE', N(run=save))
    return N(value=value, calls=calls, connected=c)


@pytest.mark.parametrize('case', ['ready', 'pending', 'origin', 'unseen'])
def test_completed_uses_original_capture(session: Any, case: str) -> None:
    s = session.value
    if case == 'pending': s.mode.native.pending.append(object())
    elif case == 'origin': s.pipe._active_chain_2p = object()
    elif case == 'unseen': s.mode.native.seen_calls.clear()
    if case in ('origin', 'unseen'):
        with pytest.raises(ValueError): s.completed(35370)
        assert not s.saved and not session.calls
    else:
        s.completed(35370)
        assert len(s.saved) == len(session.calls) == int(case == 'ready')
        assert s.schedule.accepted == ((35370,) if case == 'ready' else ())
    packet = json.loads(s.schedule_stream.getvalue())
    assert packet['saved'] is (case == 'ready')
