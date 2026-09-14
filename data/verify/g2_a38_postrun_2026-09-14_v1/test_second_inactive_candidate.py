"""実A38台帳/終了context＋人工step/tap。実Witness接続・実動画合格ではない。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import second_inactive_candidate as C
import test_terminal_boundary as F


def setup(path: Path, stack: ExitStack) -> tuple:
    ledger, row, _ = F.fixture('2P')
    rows = [row]
    pipe = object()
    journal = N(errors=[], codes={setup.__code__})
    journal.scope = lambda *args: {'generation': {'reset_epoch':ledger.scope[5]}}
    current = N(scope=ledger.scope)
    connection = N(binding=N(scope=ledger.scope), registry=N(current=lambda _: current),
        recovery=N(journal=journal, pipe=pipe, factory=object(), evidence=N(scope=lambda *a: ledger.scope)))
    mode = N(connection=connection, native=N(last_frame=ledger.clock, pending=[]), error=None)
    value = N(mode=mode, ledger=ledger, failure=None, closed=False, L=F.L,
        N=N(extract=lambda _: None), terminal=N(committed=True, closed=False),
        tap=N(journal=journal, between=lambda start, end, scope: [r for r in rows if start < r['frame_idx'] <= end]))
    session = N(_arrival_bindings={id(mode):(mode, value)}, _arrival_output=path, stack=stack,
        state={'provisional_context_observer':N(rows=[F.actual_end_context()])})
    item = dict(pipe=pipe, token='synthetic-step:36886', scope=dict(frame_idx=36886, side='2P'),
        frame=N(f_code=setup.__code__, f_locals=dict(is_active=False, frame_idx=36886, time_sec=36886/60)))
    return C.End(value, session), item, rows


def test_two_ticks_keep_unacked_and_current(tmp_path: Path) -> None:
    with ExitStack() as stack:
        end, item, rows = setup(tmp_path, stack)
        stack.push(end.close)
        before = end.value.ledger
        end.observe(item, None)
        current = end.current
        following = deepcopy(rows[0])
        following.update(frame_idx=36888, time_sec=36888/60, token='synthetic-enqueue:36888',
            update_begin_accounting={'pending_tsumo':[]})
        rows.append(following)
        item['scope']['frame_idx'] = item['frame'].f_locals['frame_idx'] = 36888
        item['frame'].f_locals['time_sec'] = 36888/60
        item['token'] = 'synthetic-step:36888'
        end.observe(item, None)
        assert end.value.ledger is before and end.current is current
        assert len(before.arrivals) == len(before.applied) == 11 and len(before.acknowledgements) == 10
        assert end.value.mode.native.last_frame == 36884 and not end.value.mode.native.pending
    saved = [json.loads(line) for line in (tmp_path/'SECOND_INACTIVE_END.jsonl').read_text().splitlines()]
    assert len(saved) == 2 and len(saved[0]['terminal_receipt']['unacknowledged_terminal_tokens']) == 1
    assert saved[1]['terminal_receipt'] is None


@pytest.mark.parametrize('mutation', ['active', 'scope', 'consumed', 'uncommitted', 'source_missing', 'step_error'])
def test_reject_invalid_end(tmp_path: Path, mutation: str) -> None:
    with pytest.raises((ValueError, RuntimeError)), ExitStack() as stack:
        end, item, rows = setup(tmp_path, stack)
        stack.push(end.close)
        error = None
        if mutation == 'active': item['frame'].f_locals['is_active'] = True
        elif mutation == 'scope': end.value.mode.connection.recovery.evidence.scope = lambda *a: ('foreign',)
        elif mutation == 'consumed': end.value.N.extract = lambda _: object()
        elif mutation == 'uncommitted': end.value.terminal.committed = False
        elif mutation == 'source_missing': rows.clear()
        elif mutation == 'step_error': error = RuntimeError('original-step-failure')
        end.observe(item, error)
    assert not (tmp_path/'SECOND_INACTIVE_END.jsonl').exists()


def test_factory_normal_active_delegates(tmp_path: Path) -> None:
    calls = []
    class Base:
        def observe(self, item: dict, result: Any, error: Any) -> dict:
            calls.append((item, result, error))
            return {'normal': True}
    session = N(_arrival_bindings={})
    selected = C.mode_factory(lambda base, session: base)(Base, session)
    mode = selected()
    item = {'frame':N(f_locals={'is_active':True})}
    assert mode.observe(item, 1, None) == {'normal':True}
    session._arrival_bindings[id(mode)] = (mode, N())
    assert mode.observe(item, 2, None) == {'normal':True}
    assert len(calls) == 2
