"""原installを通す分岐/復元検査。後段evaluateはstubで実完走ではない。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
from typing import Any
import pytest
import probability_finish as F


@pytest.mark.parametrize('kind', ['missing', 'inactive', 'active', 'partial'])
def test_finish_selection_and_restore(kind: str, monkeypatch: Any) -> None:
    calls: list[Any] = []
    def previous(*args: Any) -> dict[str, Any]:
        calls.append('original')
        return dict(original=True)
    def evaluate(*args: Any) -> dict[str, Any]:
        calls.append('probability')
        if kind == 'partial':
            raise AssertionError('partial_authority')
        return dict(probability=True)
    target = N(evaluate=previous)
    main = FunctionType((lambda: None).__code__, dict(Q=N(FINAL=target),
        K=N(write=lambda path, result: calls.append(path.name))))
    state: dict[str, Any] = {}
    if kind != 'missing':
        state['probabilistic_tracking_mode'] = N(native=None if kind == 'inactive' else object(),
            activation=None, connection=N(binding=None), error=None)
    monkeypatch.setattr(F, 'evaluate', evaluate)
    with ExitStack() as stack:
        F.install(stack, main, None, None)
        if kind == 'partial':
            with pytest.raises(AssertionError, match='^partial_authority$'):
                target.evaluate(None, None, None, Path('.'), state)
            assert calls == ['probability', 'LIVE_PROBABILITY_FINAL_FAILURE.json']
            assert state[F.KEY]['error'] == "AssertionError('partial_authority')"
        else:
            target.evaluate(None, None, None, Path('.'), state)
            if kind in ('missing', 'inactive'):
                assert calls == ['original'] and F.KEY not in state
            else:
                assert calls == ['probability', 'LIVE_PROBABILITY_FINAL.json']
    assert target.evaluate is previous
