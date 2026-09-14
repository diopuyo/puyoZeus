"""レビュー補足の終端時計・同一性・未完了負例。"""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any
import pytest
import test_terminal_prefix_reader as T

source = T.source


@pytest.mark.parametrize('fault', ['receipt_clock', 'deadline', 'time', 'ledger_identity', 'active_journal'])
def test_boundary_faults_remain_strict(source: Any, tmp_path: Path, fault: str) -> None:
    with ExitStack() as stack:
        live, mode, session, step, lane, lease, reader = T.fixture(source, tmp_path, stack)
        cap = mode.arrival_capture
        if fault == 'receipt_clock': cap.terminal_receipt['frame'] = mode.arrival_ledger.clock
        elif fault == 'deadline':
            step['frame_idx'] = mode.arrival_ledger.deadline + T.STRIDE
            cap.terminal_last = step['frame_idx']
        elif fault == 'time': step['time_sec'] += 1
        elif fault == 'ledger_identity': cap.frozen_ledger = replace(mode.arrival_ledger)
        elif fault == 'active_journal': session.journal.active = object()
        with pytest.raises(ValueError): T.R.owner(live, T.F.A, mode, session, step)
