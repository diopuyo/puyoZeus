"""旧binderの未反映受理を再現し、反映証拠との結合で拒否する。"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace as N
from typing import Any
import pytest
from test_live_binding import saved, live
import reflection as R


def mode(live: Any) -> Any:
    value = live.registry.current(live.bindings[0])
    connection = N(registry=live.registry, binding=live.bindings[0])
    native = N(connection=connection, pending=[], last_frame=value.frame, seen_calls={'step:598'})
    return N(connection=connection, native=native, error=None, activation={'frame':34944},
             applied=[dict(applied_frame=value.frame, state=R.S.encode(value))])


def test_original_accepts_unapplied_registry(live: Any) -> None:
    original = live.registry.current(live.bindings[0])
    receipt = mode(live)
    stale = replace(original, frame=original.frame-2)
    # Registry更新漏れのfault injection。旧binderが古stateを返す事実を保存する試験。
    live.registry._states[original.scope] = stale
    assert live.capture().values[0] is stale
    with pytest.raises(ValueError, match='reflection_frame'):
        R.verify(receipt, stale, 'step:598', original.frame)


def test_reflected_and_unchanged_following(live: Any) -> None:
    receipt = mode(live)
    value = live.registry.current(live.bindings[0])
    R.verify(receipt, value, 'step:598', value.frame)
    receipt.native.last_frame += 2
    receipt.native.seen_calls.add('step:600')
    R.verify(receipt, value, 'step:600', value.frame+2)


@pytest.mark.parametrize('case', ['pending', 'unseen', 'lagging', 'state'])
def test_incomplete_reflection(live: Any, case: str) -> None:
    receipt = mode(live)
    value = live.registry.current(live.bindings[0])
    if case == 'pending': receipt.native.pending.append(object())
    elif case == 'unseen': receipt.native.seen_calls.clear()
    elif case == 'lagging': receipt.native.last_frame -= 2
    elif case == 'state': receipt.applied[-1]['state']['tokens'].append('foreign')
    with pytest.raises(ValueError):
        R.verify(receipt, value, 'step:598', value.frame)
