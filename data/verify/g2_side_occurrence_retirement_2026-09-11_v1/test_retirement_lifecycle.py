"""追加wrapperの開始→保存→終了票と元失敗保持。原reset資格は別統合が必須。"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import retirement as R
import test_retirement as F


@pytest.mark.parametrize('fault', ['none', 'before', 'after', 'save', 'emit'])
def test_wrapper_save_and_original_error(tmp_path: Path, fault: str) -> None:
    context, adapter, runtime, _ = F.objects()
    error, calls = ValueError('元操作の例外'), []
    def broken_emit(row: Any) -> None:
        raise error
    if fault == 'emit': adapter.controller.rec.emit = broken_emit
    class Parent:
        def perform(self, advisory: Any, frame: int, clock: float) -> None:
            calls.append(frame)
            if fault in ('before', 'save'): raise error
            runtime.histories['1P'] = F.F.native.History(1)
            if fault == 'after': adapter.controller.active = object()
    child = R.derived(Parent)()
    child.__dict__.update(vars(context), rows=[], error=None)
    child.state['output'] = tmp_path
    if fault == 'save': (tmp_path / 'SIDE_OCCURRENCE_RETIREMENT.json').mkdir()
    if fault in ('none', 'emit'):
        child.perform(None, F.F.FRAME, F.F.FRAME / 60)
    else:
        with pytest.raises(ValueError) as caught:
            child.perform(None, F.F.FRAME, F.F.FRAME / 60)
        if fault != 'after': assert caught.value is error
    assert calls == [F.F.FRAME]
    if fault == 'save':
        assert child.rows[-1]['stage'] == 'retirement_save_failed'
    else:
        result = json.loads((tmp_path / 'SIDE_OCCURRENCE_RETIREMENT.json').read_bytes())
        expected = {'none': 'side_occurrence_retired', 'before': 'original_side_reset',
                    'after': 'after_reset_retirement', 'emit': 'side_occurrence_retired'}
        assert result['stage'] == expected[fault]
        if fault == 'emit': assert result['new']['epoch'] == 1 and result['emitted'] is False
