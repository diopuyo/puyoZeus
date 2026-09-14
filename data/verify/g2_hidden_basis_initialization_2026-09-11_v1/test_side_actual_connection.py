"""片側接続の保存分類/原例外保持/2P値検査。原factory成功は別runで必須。"""
from __future__ import annotations
import contextlib
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_reset_inflight_quarantine_2026-09-11_v1'))
import side_actual_connection as S


def context(output: Path) -> Any:
    return N(stack=contextlib.ExitStack(), recovery=None, rows=[], error=None,
        state=dict(output=output, repeat_scope_guard=N(reset_lease=N(events=[]))))


def test_install_failure_is_recorded_without_replacing_exception(tmp_path: Any, monkeypatch: Any) -> None:
    value, error = context(tmp_path), ValueError('source_guard_failure')
    def fail() -> None:
        raise error
    monkeypatch.setattr(S, 'selected', fail)
    with pytest.raises(ValueError) as caught:
        S.run(value, lambda *_: pytest.fail('原performへ入らない'), None, 10, 10 / 60)
    assert caught.value is error
    record = json.loads((tmp_path / 'SIDE_ACTUAL_CONNECTION.json').read_bytes())
    assert record['stage'] == 'side_install_failed' and record['recovery_rows'] is None


def test_save_failure_preserves_original_error(tmp_path: Any, monkeypatch: Any) -> None:
    value, error = context(tmp_path), ValueError('元エラー')
    def fail() -> None:
        raise error
    monkeypatch.setattr(S, 'selected', fail)
    (tmp_path / 'SIDE_ACTUAL_CONNECTION.json').mkdir()
    with pytest.raises(ValueError) as caught:
        S.run(value, lambda *_: None, None, 10, 10 / 60)
    assert caught.value is error and value.rows[-1]['stage'] == 'side_failure_save_failed'


def test_preflight_rejection_has_reason_and_details(tmp_path: Any, monkeypatch: Any) -> None:
    class Rejected(ValueError):
        reason, report = 'fifo_not_empty', {'side': '1P'}
    monkeypatch.setitem(sys.modules, '_g2_selected_side_join', N(SideJoinRejected=Rejected))
    monkeypatch.setattr(S, 'selected', lambda: N(install=lambda *_: None))
    monkeypatch.setattr(S, 'snapshot', lambda *_: {})
    error = Rejected('fifo_not_empty')
    def fail(*args: Any) -> None:
        raise error
    with pytest.raises(Rejected) as caught:
        S.run(context(tmp_path), fail, None, 10, 10 / 60)
    assert caught.value is error
    record = json.loads((tmp_path / 'SIDE_ACTUAL_CONNECTION.json').read_bytes())
    assert record['stage'] == 'side_preflight_rejected' and record['details'] == {'side': '1P'}


@pytest.mark.parametrize('fault', ['none', 'other_epoch', 'other_generation', 'clock', 'full_reset'])
def test_side_value_checks(fault: str) -> None:
    side = dict(epoch=0, history={}, history_id=1, generation=dict(reset_epoch=0))
    before = dict(sides={'1P': deepcopy(side), '2P': deepcopy(side)}, pipe_id=1,
                  history_dict_id=2, true_clocks={'clock': 10}, shared={})
    after = deepcopy(before)
    after['sides']['1P']['epoch'] = after['sides']['1P']['generation']['reset_epoch'] = 1
    lease = N(outer_calls=0, native_calls=0, depth=0)
    if fault == 'other_epoch': after['sides']['2P']['epoch'] = 1
    if fault == 'other_generation': after['sides']['2P']['generation']['reset_epoch'] = 1
    if fault == 'clock': after['true_clocks']['clock'] = 11
    if fault == 'full_reset': lease.outer_calls = 1
    if fault == 'none':
        S.verify(before, after, lease)
    else:
        with pytest.raises(AssertionError):
            S.verify(before, after, lease)
