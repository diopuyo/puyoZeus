"""実旧所有fixture＋人工終了認証。原J実動画の代わりにはしない。"""
from contextlib import ExitStack
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import test_original_terminal_reader as O
import terminal_prefix_reader as R

source, F = O.source, O.F
STRIDE, FPS = 2, 60


def fixture(source: Any, path: Path, stack: Any) -> tuple:
    live, mode, session, step, lane = F.fixture(source, path, stack)
    frame = step['frame_idx'] + STRIDE
    step = step | dict(frame_idx=frame, time_sec=frame/FPS, status='returned',
        exception=None, returned=dict(state='MENU', active_origin=None))
    ledger = mode.arrival_ledger
    session.journal.errors = []
    session.error, session.restored = None, False
    session.state[R.MODE_KEY] = mode
    session.witness = object()
    mode.arrival_capture = N(mode=mode, recovery=mode.connection.recovery,
        journal=session.journal, terminal_receipt=dict(kind='arrival_scope_terminal/v1',
            frame=frame, last_live_ledger=asdict(ledger)), frozen_ledger=ledger,
        frozen_current=lane.current, terminal_last=frame, pending=None, error=None,
        terminal_stream=io.StringIO())
    lease = N(closed=False, live=live, current=lambda:(live,F.A), mode_open=lambda value:None)
    reader = N(read_pair=lambda *args:N(holds=(), rows_json=json.dumps([step])))
    return live, mode, session, step, lane, lease, reader


def test_terminal_read_holds_without_mutation(source: Any, tmp_path: Path) -> None:
    with ExitStack() as stack:
        live, mode, session, step, lane, lease, reader = fixture(source, tmp_path, stack)
        frozen = mode.arrival_ledger
        assert R.owner(live, F.A, mode, session, step) is lane
        result = R.read(session, lease, reader, step['frame_idx'])
        assert result == (None, step, 'match_ended_scope_frozen')
        assert mode.arrival_ledger is frozen and mode.arrival_capture.frozen_current is lane.current


@pytest.mark.parametrize('fault', ['receipt', 'last_frame', 'ledger', 'current', 'pending',
    'source', 'generation', 'active_origin', 'active_state', 'stream', 'owner', 'journal_error'])
def test_terminal_fault_not_silently_held(source: Any, tmp_path: Path, fault: str) -> None:
    with ExitStack() as stack:
        live, mode, session, step, lane, lease, reader = fixture(source, tmp_path, stack)
        cap = mode.arrival_capture
        if fault == 'receipt': cap.terminal_receipt = None
        elif fault == 'last_frame': cap.terminal_last -= STRIDE
        elif fault == 'ledger': cap.terminal_receipt['last_live_ledger']['clock'] -= STRIDE
        elif fault == 'current': cap.frozen_current = object()
        elif fault == 'pending': cap.pending = object()
        elif fault == 'source': step['source_id'] = 'foreign'
        elif fault == 'generation': step['generation_after'] = dict(step['generation_after'], reset_epoch=-1)
        elif fault == 'active_origin': step['returned']['active_origin'] = {}
        elif fault == 'active_state': step['returned']['state'] = 'STABLE'
        elif fault == 'stream': cap.terminal_stream.close()
        elif fault == 'owner': cap.mode = object()
        elif fault == 'journal_error': session.journal.errors.append('original')
        with pytest.raises(ValueError): R.owner(live, F.A, mode, session, step)


def test_normal_clock_contract_unchanged(source: Any, tmp_path: Path) -> None:
    with ExitStack() as stack:
        live, mode, session, step, lane = F.fixture(source, tmp_path, stack)
        assert R.owner(live, F.A, mode, session, step) is lane
        with pytest.raises(ValueError, match='projection_live_finished_clock'):
            R.owner(live, F.A, mode, session, step | dict(frame_idx=step['frame_idx']+STRIDE))
