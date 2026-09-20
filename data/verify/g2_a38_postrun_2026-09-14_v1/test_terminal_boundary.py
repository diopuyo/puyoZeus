"""実A38終了台帳＋人工の未到達inactive原票で契約を検査。実終了合格ではない。"""
from __future__ import annotations
from copy import deepcopy
from collections import deque
from dataclasses import asdict, replace
from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any
import pytest
import terminal_boundary as C

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'video38_second_prefix_candidate_v38'
sys.path.insert(0, str(ROOT.parent / 'g2_arrival_ack_candidate_2026-09-12_v1'))
import ledger as L
import test_enqueue_capture as F


@lru_cache(maxsize=1)
def actual_end_context() -> dict:
    with (RUN/'provisional_context.jsonl').open() as stream:
        rows = deque(stream, maxlen=2)
    value = json.loads(rows[0])
    assert value['frame_idx'] == 36884
    return value


def fixture(side: str) -> tuple[Any, dict, dict]:
    if side == '1P':
        value = json.loads((RUN/'ARRIVAL_LEDGER_STATUS.json').read_bytes())['state']
    else:
        paths = tuple(RUN.glob('SECOND_TERMINAL_*.jsonl'))
        assert len(paths) == 1
        value = json.loads(paths[0].read_text().splitlines()[-1])['ledger']
    scope = tuple(value['scope'])
    arrivals = tuple(L.Arrival(scope, a['token'], tuple(a['pair']), a['frame'], a['call_token']) for a in value['arrivals'])
    acks = tuple(L.Ack(a['token'], a['frame'], a['call_token']) for a in value['acknowledgements'])
    ledger = L.Ledger(scope, value['start'], value['deadline'], value['clock'], arrivals, tuple(value['applied']), acks)
    unacked = arrivals[len(acks):]
    row = dict(source_id=scope[0], run_id=scope[1], software_reset=scope[2], pipe_object_id=scope[3],
        generation=dict(reset_epoch=scope[5]), side=side, frame_idx=ledger.clock+2,
        time_sec=(ledger.clock+2)/60, kind='enqueue', status='returned', active=False,
        returned_none=True, token='synthetic-inactive-not-observed-J', added_occurrence_tokens=[],
        fifo_occurrence_tokens=[], before=dict(pending_tsumo=[]), after=dict(pending_tsumo=[], tsumo_count={}),
        discarded_tokens=[a.token for a in unacked], update_begin_accounting=dict(pending_tsumo=[list(a.pair) for a in unacked]))
    end = dict(source_id=scope[0], run_id=scope[1], frame=36840,
        match_end_locked=True, post_match_lockdown_active=True)
    return ledger, row, end


@pytest.mark.parametrize('side', ['1P', '2P'])
def test_seal_preserves_real_ledger_and_unacked(side: str) -> None:
    ledger, row, end = fixture(side)
    before = deepcopy(asdict(ledger))
    receipt = C.prepare(L, ledger, row, end)
    assert asdict(ledger) == before
    assert len(receipt['unacknowledged_terminal_tokens']) == (0 if side == '1P' else 1)
    assert not receipt['acknowledged_by_this_event'] and not receipt['physical_applied_by_this_event']
    assert not receipt['further_old_scope_updates_allowed'] and not receipt['quality_gate_clear']


@pytest.mark.parametrize('mutation', ['active', 'missing_end', 'foreign', 'clock', 'added', 'discard', 'begin', 'failed'])
def test_reject_unqualified_terminal(mutation: str) -> None:
    ledger, row, end = fixture('2P')
    if mutation == 'active': row['active'] = True
    elif mutation == 'missing_end': end['match_end_locked'] = False
    elif mutation == 'foreign': end['run_id'] = 'foreign'
    elif mutation == 'clock': row['frame_idx'] += 2
    elif mutation == 'added': row['added_occurrence_tokens'] = ['fake']
    elif mutation == 'discard': row['discarded_tokens'] = []
    elif mutation == 'begin': row['update_begin_accounting']['pending_tsumo'] = []
    elif mutation == 'failed': row['status'] = 'exception'
    with pytest.raises(ValueError):
        C.prepare(L, ledger, row, end)


@pytest.mark.parametrize('side', ['1P', '2P'])
def test_original_journal_inactive_clear_then_terminal(side: str) -> None:
    ledger, _, end = fixture(side)
    _, journal, pipe = F.setup()
    scope = tuple(id(pipe) if i == 3 else value for i, value in enumerate(ledger.scope))
    ledger = replace(ledger, scope=scope, arrivals=tuple(replace(a, scope=scope) for a in ledger.arrivals))
    frame = ledger.clock + 2
    journal.selected = {(frame, side)}
    journal.source_id, journal.run_id = scope[:2]
    journal.epoch = lambda p, s: scope[2]
    journal.scope = lambda p, s, f, t: dict(source_id=scope[0], run_id=scope[1], frame_idx=f,
        time_sec=t, pipe_object_id=id(p), side=s, generation=dict(reset_epoch=scope[5]))
    queue = getattr(pipe, '_pending_tsumo_' + side.lower())
    unacked = ledger.arrivals[len(ledger.acknowledgements):]
    queue.extend(a.pair for a in unacked)
    old = journal.fifo.sync(pipe, side, scope[2])
    old['tokens'] = [a.token for a in unacked]  # 人工初期履歴。過去原appendの再現ではない。
    journal.update_before[side] = dict(queue=queue, refs=tuple(queue), accounting=F.J.account(pipe, side))
    queue.clear()  # 原pipelineのinactive clearを模擬し、J自身の破棄追跡を通す。
    calls = []
    def original(p: Any, s: str, f: int, t: float, active: bool, pair: Any) -> None:
        calls.append((s, active))
    journal.wrap_enqueue(original)(pipe, side, frame, frame/60, False, None)
    rows = [json.loads(line) for line in journal.stream.getvalue().splitlines()]
    assert calls == [(side, False)] and len(rows) == 1 and not journal.errors
    receipt = C.prepare(L, ledger, rows[0], end)
    assert receipt['unacknowledged_terminal_tokens'] == [a.token for a in unacked]
    assert len(ledger.acknowledgements) == (7 if side == '1P' else 10)
    assert not queue and not receipt['acknowledged_by_this_event']


@pytest.mark.parametrize('side', ['1P', '2P'])
def test_actual_completed_context_qualifies_end(side: str) -> None:
    ledger, row, _ = fixture(side)
    end = C.end_from_context(ledger, actual_end_context())
    receipt = C.prepare(L, ledger, row, end)
    assert end['frame'] == 36884 and end['qualification'] == 'original_completed_context'
    assert not receipt['quality_gate_clear']  # inactiveイベントだけは未到達の人工対照。


@pytest.mark.parametrize('mutation', ['missing_flag', 'failed', 'clock', 'generation'])
def test_end_context_rejects_missing_failed_stale(mutation: str) -> None:
    ledger, _, _ = fixture('1P')
    row = deepcopy(actual_end_context())
    if mutation == 'missing_flag': row['update']['match_end_locked_observed'] = False
    elif mutation == 'failed': row['capture_status'] = 'FAILED'
    elif mutation == 'clock': row['frame_idx'] -= 2
    elif mutation == 'generation': row['generation']['after']['value']['1P']['reset_epoch'] += 1
    with pytest.raises(ValueError):
        C.end_from_context(ledger, row)
