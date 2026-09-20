"""原identity本文の保持とbinding参照だけの交換を確認。実retired認証は別統合。"""
from __future__ import annotations
from typing import Any
import pytest
import stage_retired_identity as S
import test_stage_identity_repro as R


def repaired() -> tuple[Any, Any, Any, Any, Any, Any]:
    control, factory, rows, module, old = R.sample()
    del control.history['1P']
    original = R.actual()
    def binding(given: Any, supplied: Any) -> Any:
        assert given is factory and supplied is rows
        return old
    function = S.rewrite(original, binding)
    assert original.__globals__['live_identity'] is original
    return function, control, factory, rows, module, old


def test_repaired_normal_and_original_retains_refusal() -> None:
    function, control, factory, rows, module, _ = repaired()
    assert function(control, factory, rows, module) is True
    assert control.history == {}
    with pytest.raises(KeyError, match='1P'):
        R.actual()(control, factory, rows, module)


@pytest.mark.parametrize('defect,reason', [('controller', 'live_controller_identity'),
    ('provider', 'live_controller_identity'), ('state', 'live_final_state'), ('journal', 'live_J_missing')])
def test_repaired_keeps_original_rejection(defect: str, reason: str) -> None:
    function, control, factory, rows, module, _ = repaired()
    if defect == 'controller':
        factory.controller = object()
    elif defect == 'provider':
        factory.provider = object()
    elif defect == 'state':
        rows[-1]['decision']['history_state']['frame'] += 2
    else:
        factory.provider.journal.controller = None
    with pytest.raises(AssertionError, match='^' + reason + '$'):
        function(control, factory, rows, module)
    assert control.history == {}


def test_source_code_replacement_rejected() -> None:
    function = R.actual()
    function.__code__ = (lambda *args: True).__code__
    with pytest.raises(AssertionError, match='^stage_identity_code$'):
        S.rewrite(function, lambda *args: object())


def test_mixed_missing_parts_kept() -> None:
    function, control, _, rows, module, _ = repaired()
    with pytest.raises(AssertionError, match='^live_parts_missing$'):
        function(control, None, rows, module)
    assert function(None, None, rows, module) is False
