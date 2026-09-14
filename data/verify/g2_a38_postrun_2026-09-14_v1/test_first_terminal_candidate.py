"""原enqueue→候補Mode→保存/復元。初期盤面・終了資格・step通知は人工。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import first_terminal_candidate as C
import terminal_ownership as O

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'g2_arrival_ack_candidate_2026-09-12_v1'))
import test_arrival_mode as F


def setup(path: Path) -> tuple:
    mode, journal, pipe, _ = F.fixture(path)
    selected, capture = C.derive(F.A)
    mode.__class__ = selected  # 人工fixtureに最終候補型を使用。実loader接続ではない。
    mode.basis_origin, mode.basis_cascade_closed = None, True
    scope = mode.arrival_ledger.scope
    context = dict(source_id=scope[0], run_id=scope[1], frame_idx=10, time_sec=10/60,
        available_frame=10, capture_status='CAPTURED', failures=[], upstream_failures={},
        capture_token='synthetic-end-context', generation={'after': {'value': {'1P': {'reset_epoch':scope[5], 'side':'1P'}}}},
        update=dict(returned=True, exception=None, returned_frame_idx=10, returned_time_sec=10/60,
            match_end_locked_observed=True, match_end_locked=True,
            post_match_lockdown_active_observed=True, post_match_lockdown_active=True))
    mode.state['provisional_context_observer'] = N(rows=[context])
    journal.codes.add(notify.__code__)
    return mode, journal, pipe, capture


def notify(mode: Any, journal: Any, pipe: Any, frame: int) -> dict:
    item = dict(frame=N(f_code=notify.__code__, f_locals=dict(is_active=False, frame_idx=frame, time_sec=frame/60)),
        pipe=pipe, epoch=3, scope=journal.scope(pipe, '1P', frame, frame/60), token=f'step:{frame}', events=[])
    row = mode.observe(item, None, None)
    mode.save(row)
    return row


def send(journal: Any, pipe: Any, frame: int, failed: bool = False) -> None:
    journal.update_before['1P'] = dict(accounting=F.F.J.account(pipe, '1P'))
    def original(p: Any, s: str, f: int, t: float, active: bool, pair: Any) -> None:
        if failed: raise RuntimeError('original-noop-failure')
    journal.wrap_enqueue(original)(pipe, '1P', frame, frame/60, False, None)


def test_inactive_then_next_tick_no_old_ledger_update(tmp_path: Path) -> None:
    mode, journal, pipe, capture = setup(tmp_path)
    before = mode.arrival_ledger
    with ExitStack() as stack:
        mode.arrival_capture = capture(mode, tmp_path)
        stack.push(mode.arrival_capture.close)
        send(journal, pipe, 12)
        assert mode.arrival_capture.terminal_receipt is None
        first = notify(mode, journal, pipe, 12)
        send(journal, pipe, 14)
        notify(mode, journal, pipe, 14)
        assert first['terminal_scope_observation'] and mode.arrival_ledger is before
        assert mode.native.last_frame == 10 and mode.rows == 0
    assert 'appended' not in vars(journal.fifo)
    rows = [json.loads(line) for line in (tmp_path/'FIRST_TERMINAL.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and rows[0]['terminal_receipt'] and rows[1]['terminal_receipt'] is None
    assert json.loads((tmp_path/'ARRIVAL_SOURCE_STATUS.json').read_bytes())['rows'] == 0


def test_original_failure_is_not_terminal_success(tmp_path: Path) -> None:
    mode, journal, pipe, capture = setup(tmp_path)
    with ExitStack() as stack, pytest.raises(RuntimeError, match='original-noop-failure'):
        mode.arrival_capture = capture(mode, tmp_path)
        stack.push(mode.arrival_capture.close)
        send(journal, pipe, 12, failed=True)
    assert not (tmp_path/'FIRST_TERMINAL.jsonl').exists()
    assert json.loads((tmp_path/'FIRST_TERMINAL_STATUS.json').read_bytes())['receipt'] is None


def test_active_path_keeps_original_arrival_math(tmp_path: Path) -> None:
    mode, journal, pipe, _ = F.fixture(tmp_path)
    selected, capture = C.derive(F.A)
    mode.__class__ = selected
    with ExitStack() as stack:
        mode.arrival_capture = capture(mode, tmp_path)
        stack.push(mode.arrival_capture.close)
        F.F.send(journal, pipe)
        row = F.step(mode, journal, pipe, 12)
        assert row['transition']['kind'] == 'basis_cascade_with_arrival'
        assert len(mode.arrival_ledger.applied) == 1
        assert mode.arrival_capture.terminal_receipt is None
    assert not (tmp_path/'FIRST_TERMINAL.jsonl').exists()


def test_original_step_error_identity_preserved(tmp_path: Path) -> None:
    mode, journal, pipe, capture = setup(tmp_path)
    failure = RuntimeError('original-step-failure')
    with pytest.raises(RuntimeError) as caught, ExitStack() as stack:
        mode.arrival_capture = capture(mode, tmp_path)
        stack.push(mode.arrival_capture.close)
        send(journal, pipe, 12)
        mode.observe({}, None, failure)
    assert caught.value is failure
    assert not (tmp_path/'FIRST_TERMINAL.jsonl').exists()
    assert json.loads((tmp_path/'FIRST_TERMINAL_STATUS.json').read_bytes())['pending']


def test_original_enqueue_reentry_keeps_first_prepared(tmp_path: Path) -> None:
    mode, journal, pipe, capture = setup(tmp_path)
    with pytest.raises(ValueError, match='enqueue_owner_or_unfinished'), ExitStack() as stack:
        mode.arrival_capture = capture(mode, tmp_path)
        stack.push(mode.arrival_capture.close)
        send(journal, pipe, 12)
        prepared = mode.arrival_capture.terminal_prepared
        send(journal, pipe, 12)
    assert mode.arrival_capture.terminal_prepared is prepared
    assert mode.arrival_capture.terminal_receipt is None


def test_ended_lease_preserves_archive_and_checks_actual_scope(tmp_path: Path) -> None:
    mode, journal, pipe, capture = setup(tmp_path)
    recovery, connection = mode.connection.recovery, mode.connection
    recovery.factory = connection.registry.factory = object()
    recovery.pending, recovery.baseline_count, recovery.control = None, 0, N(history={})
    checked = []
    lease = N(recovery=recovery, used=True, active=False, waiting=False, guard=N(binding=None),
        archive=N(verify=lambda: checked.append(True)), observe_wait=lambda: 'original')
    with ExitStack() as stack:
        mode.arrival_capture = capture(mode, tmp_path)
        stack.push(mode.arrival_capture.close)
        O.install(stack, mode, lease, F.A.CORE.patch)
        assert lease.observe_wait() == 'original'
        send(journal, pipe, 12)
        notify(mode, journal, pipe, 12)
        scope = mode.arrival_ledger.scope
        actual = tuple(value + 1 if index == 5 else value for index, value in enumerate(scope))
        recovery.evidence.scope = lambda f, p: actual
        assert lease.observe_wait() == actual and checked == [True]
        recovery.evidence.scope = lambda f, p: ('foreign', *actual[1:])
        with pytest.raises(ValueError, match='terminal_actual_scope'):
            lease.observe_wait()
