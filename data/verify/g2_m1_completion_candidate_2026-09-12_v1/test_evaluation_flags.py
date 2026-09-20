"""人工原stepで局所flagを捕捉し、元例外・一回転送・hook解除を検査する。"""
from __future__ import annotations

from contextlib import ExitStack
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import evaluation_flags as F


def step(journal: Any, frame_idx: int, side: str = '1P', error: Any = None) -> None:
    sm = getattr(journal.pipe, '_sm_' + side.lower())
    sm.context.frame_idx = frame_idx
    signals = N(is_match_active=True, effect_gate_window_active=False)
    scope = dict(frame_idx=frame_idx, time_sec=frame_idx/60, side=side)
    item = dict(frame=sys._getframe(), pipe=journal.pipe, scope=scope, token=f'{side}:{frame_idx}', epoch=3)
    journal.complete_step(item, N(state=N(value='stable')), error, None)
    assert item['frame'] is None


def journal() -> Any:
    class Journal:
        def __init__(self) -> None:
            self.pipe = N(_sm_1p=N(context=N()), _sm_2p=N(context=N()),
                          _landing_grace_1p=None, _landing_grace_2p=None,
                          _active_chain_1p=None, _active_chain_2p=None)
            self.codes, self.calls, self.failure = {step.__code__}, 0, None
        def complete_step(self, item: dict, result: Any, error: Any, profile: Any) -> None:
            self.calls += 1
            item['frame'] = None
            if self.failure is not None: raise self.failure
    return Journal()


def test_both_sides_original_once_and_restore() -> None:
    j = journal()
    original = j.complete_step
    with ExitStack() as stack:
        evidence = F.install(stack, j)
        step(j, 10)
        step(j, 10, '2P')
        assert set(evidence.latest) == {'1P', '2P'} and j.calls == 2
        assert all(r['match_active'] is True and r['effect_window'] is False for r in evidence.latest.values())
    assert evidence.closed and evidence.error is None and j.complete_step == original


def test_original_exception_identity() -> None:
    j = journal()
    j.failure = RuntimeError('原書込失敗')
    with ExitStack() as stack:
        evidence = F.install(stack, j)
        with pytest.raises(RuntimeError) as caught: step(j, 10)
        assert caught.value is j.failure and evidence.error is j.failure
        assert evidence.latest == {} and j.calls == 1


def test_unknown_original_code_is_not_silent() -> None:
    j = journal()
    j.codes = set()
    with ExitStack() as stack:
        evidence = F.install(stack, j)
        with pytest.raises(ValueError, match='original_J_owner'): step(j, 10)
        assert evidence.error is not None and j.calls == 1 and not evidence.latest
