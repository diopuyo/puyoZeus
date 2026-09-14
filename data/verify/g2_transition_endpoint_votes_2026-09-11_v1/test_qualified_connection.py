"""資格predicateの人工反例。原J到達成功は別の実factory試験が必須。"""
from __future__ import annotations
from types import SimpleNamespace as N
from typing import Any
import pytest
import qualified_connection as Q


def objects() -> tuple[Any, Any, Any]:
    scope = ('source', 'run', 1, 2, 3, 1, '1P')
    old = ('source', 'run', 0, 2, 3, 0, '1P')
    pipe, caller = N(_pending_tsumo_1p=[]), N(f_code=objects.__code__)
    item = dict(frame=caller, pipe=pipe, epoch=1, token='same-call', scope={'frame_idx': 12})
    journal = N(active=item, errors=[], codes={caller.f_code}, scope=lambda *_: item['scope'], epoch=lambda *_: 1)
    r = N(pipe=pipe, journal=journal, error=None, reset_count=1, factory=object(),
        pending=dict(epoch=1, used=False, old_scope=old), evidence=N(scope=lambda *_: scope),
        control=N(tickets={}), provider=N(no_origin=lambda *_: True),
        waits={id(caller): dict(item=item, caller=caller, scope=scope, frame=12, clock=12 / 60, view=object())})
    return Q.Qualification(r, 10, 24), N(time_sec=12 / 60), caller


def test_saved_same_call_is_eligible() -> None:
    value, signals, caller = objects()
    assert value.check(12, signals, caller)
    assert value.rows[0]['same_call'] and value.rows[0]['token'] == 'same-call'


@pytest.mark.parametrize('fault', ['wait_missing', 'foreign_item', 'epoch', 'fifo', 'ticket', 'clock'])
def test_qualification_rejects_missing_or_foreign_evidence(fault: str) -> None:
    value, signals, caller = objects()
    r = value.recovery
    if fault == 'wait_missing': r.waits.clear()
    if fault == 'foreign_item': r.waits[id(caller)]['item'] = dict(r.journal.active)
    if fault == 'epoch': r.journal.active['epoch'] = 2
    if fault == 'fifo': r.pipe._pending_tsumo_1p.append((1, 2))
    if fault == 'ticket': r.control.tickets[id(caller)] = object()
    if fault == 'clock': signals.time_sec = 1
    with pytest.raises(RuntimeError):
        value.check(12, signals, caller)
    assert not value.rows


@pytest.mark.parametrize('fault', ['no_view', 'origin', 'deadline'])
def test_unqualified_window_does_not_supply_votes(fault: str) -> None:
    value, signals, caller = objects()
    if fault == 'no_view': value.recovery.waits[id(caller)]['view'] = None
    if fault == 'origin': value.recovery.provider.no_origin = lambda *_: False
    if fault == 'deadline': value.deadline = 10
    assert value.check(12, signals, caller) is False


def test_caller_bypass_is_rejected() -> None:
    value, _, _ = objects()
    with pytest.raises(RuntimeError, match='vote_update_caller'):
        value()
    assert value.error is not None
