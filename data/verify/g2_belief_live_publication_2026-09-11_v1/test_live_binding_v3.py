"""v2原J/同update検査を保ったv3入口の反映・寿命検査。"""
from __future__ import annotations
from typing import Any
import pytest
from test_live_binding_v2 import saved,live,connected
import live_binding_v3 as V


def capture(c: Any,saved: Any) -> Any:
    b=c.base
    return V.current(b.rec,b.factory,b.pipe,c.journal,b.registry,b.bindings,saved[3],c.witness,tuple(c.modes))


def test_same_bound_through_new_entry(connected: Any,saved: Any) -> None:
    before=connected.capture()
    after=capture(connected,saved)
    assert before.digest==after.digest and before.tokens==after.tokens
    assert all(a is b for a,b in zip(before.values,after.values,strict=True))


@pytest.mark.parametrize('case',['closed','unseen','witness'])
def test_new_entry_keeps_old_rejections(connected: Any,saved: Any,case: str) -> None:
    if case=='closed': connected.modes[1].closed=True
    elif case=='unseen': connected.modes[1].native.seen_calls.clear()
    else: connected.witness.closed=True
    with pytest.raises(ValueError):
        capture(connected,saved)
