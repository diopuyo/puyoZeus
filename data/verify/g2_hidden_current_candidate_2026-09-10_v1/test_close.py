"""接続wrapperの失敗閉鎖境界。人工journalであり原Jの到達証拠ではない。"""
from __future__ import annotations
from contextlib import ExitStack
from types import FunctionType, SimpleNamespace as N
from typing import Any
import pytest
from test_conditional import current
import hidden_current_connection as H


def unused() -> None:
    pass


def patched(stack: Any, obj: Any, name: str, value: Any) -> None:
    old = getattr(obj,name)
    stack.callback(setattr,obj,name,old)
    setattr(obj,name,value)


@pytest.mark.parametrize('case',('success','capture_error','original_error','close_error','anchor_error'))
def test_complete_exactly_once_and_no_early_candidate(monkeypatch: Any, case: str) -> None:
    value,state,binding = current()
    binding.owner,binding.hidden_current = N(state=state),None
    lifecycle = N(compatible_current=lambda *args:False)
    prepared = FunctionType(unused.__code__,{'V1':N(L=lifecycle)})
    control = type('ArtificialControl',(),{'prepared':prepared})()
    control.sticky_error = None
    calls,rows,outputs = [],[],[]
    def complete(*args: Any) -> str:
        assert binding.hidden_current is None and not outputs
        calls.append(args)
        if case=='close_error': journal.errors.append('artificial-close-error')
        if case=='anchor_error': value_state.current = 'invalid-slot-fixture'
        return 'closed'
    value_state = state
    journal = N(complete_step=complete,errors=[])
    factory = N(controller=control,provider=N(journal=journal))
    def capture(*args: Any) -> Any:
        if case=='capture_error': raise ValueError('artificial-capture-error')
        return {'binding':binding},(value,object())
    monkeypatch.setattr(H,'fall_connection',lambda *args:None)
    monkeypatch.setattr(H.H,'install',lambda *args:None)
    monkeypatch.setattr(H,'capture',capture)
    error = ValueError('original') if case=='original_error' else None
    with ExitStack() as stack:
        H.install(stack,factory,patched,None,rows,outputs)
        if case in ('capture_error','close_error','anchor_error'):
            with pytest.raises((ValueError,TypeError)): journal.complete_step({},None,error,None)
            assert control.sticky_error
        else:
            assert journal.complete_step({},None,error,None)=='closed'
    assert len(calls)==1 and journal.complete_step is complete
    assert bool(outputs)==bool(rows)==(case=='success')
    assert (binding.hidden_current is value)==(case=='success')
