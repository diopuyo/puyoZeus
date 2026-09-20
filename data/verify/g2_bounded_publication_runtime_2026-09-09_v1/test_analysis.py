"""公開差の限定判定器が余分な変更を通さないことを確認する。"""
from __future__ import annotations
import copy
from types import SimpleNamespace
from typing import Any
import pytest
import analyze_metadata as A


def fixture() -> tuple[Any, dict, dict]:
    old = {'frame_idx': 35806, 'time_sec': 35806 / 60, 'returned': {
        side: dict(confirmed=None, probability=None, board_none_reason=A.L.P.PENDING, state='STABLE')
        for side in ('1P', '2P')}}
    new = copy.deepcopy(old)
    new['returned']['2P'].update(confirmed={'sha256': 'synthetic'}, probability={'present': True}, board_none_reason=None)
    return SimpleNamespace(rows=lambda p: iter([old if p.parent == A.OLD else new])), old, new


def test_publication_delta_exact_pending_and_unchanged() -> None:
    x, old, new = fixture()
    assert A.publication_delta(x, [(35806, '2P')])['changed_frame_sides'] == [(35806, '2P')]
    new['returned'] = copy.deepcopy(old['returned'])
    assert A.publication_delta(x, [])['changed_frame_sides'] == []


@pytest.mark.parametrize('fault', ('foreign', 'unknown_reason', 'missing_status', 'clock'))
def test_publication_delta_rejects_extra_change(fault: str) -> None:
    x, old, new = fixture()
    accepted = [(35806, '2P')]
    if fault == 'foreign':
        new['returned']['2P']['state'] = 'CHAIN'
    elif fault == 'unknown_reason':
        old['returned']['2P']['board_none_reason'] = 'unknown'
    elif fault == 'missing_status':
        accepted = []
    else:
        new['time_sec'] += 1
    with pytest.raises((ValueError, RuntimeError)):
        A.publication_delta(x, accepted)
