"""同scope結果を正常集計へ渡し、停止理由を成功へ変換しない。"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import pytest
import common as K
import repeated_connection as R


def sample() -> dict[str, Any]:
    binding = object()
    guard = SimpleNamespace(error=None, record=None, frame=K.FRAMES[-1], binding=binding,
        factory=SimpleNamespace(controller=SimpleNamespace(history={'1P':binding})))
    hook = dict(attempts=1, installed=True, closed=True, references_restored=True,
        qualification={'qualified_restored':True}, rows=[])
    return dict(repeat_scope_guard=guard, repeated_firing_constructor=hook)


def test_normal_scope_completion() -> None:
    value = R.verify(sample())
    assert value['same_scope_guard']['same_binding'] and not value['quality_gate_clear']


@pytest.mark.parametrize('kind', ('missing', 'error', 'record', 'frame', 'binding', 'empty'))
def test_unclosed_scope_is_rejected(kind: str) -> None:
    state = sample()
    guard = state['repeat_scope_guard']
    if kind == 'missing': del state['repeat_scope_guard']
    if kind == 'error': guard.error = RuntimeError('same_scope_stop:reset_requested')
    if kind == 'record': guard.record = {'reason':'reset_requested'}
    if kind == 'frame': guard.frame = K.FRAMES[-1]-K.STRIDE
    if kind == 'binding': guard.factory.controller.history['1P'] = object()
    if kind == 'empty': guard.binding = None; guard.factory.controller.history.clear()
    with pytest.raises(ValueError, match='same_scope_not_closed'):
        R.verify(state)


def test_stop_record_retained_without_row_save() -> None:
    state = sample()
    guard = state['repeat_scope_guard']
    guard.error, guard.record = RuntimeError('first_stop'), {'reason':'inactive_after_baseline'}
    value = R.scope_status(state)
    assert value['record'] is guard.record and value['error'] == repr(guard.error)
    assert not state['repeated_firing_constructor']['rows']
    assert not value['quality_gate_clear'] and not value['new_scope_continuation']
