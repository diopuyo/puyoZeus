"""原owner境界の人工対照を再用。原world全体は後続の統合で検査する。"""
from typing import Any
from types import SimpleNamespace as N
import pytest
import probability_world as W
from test_probability_owner import R, sample


def test_old_scope_uses_retained_owner() -> None:
    s = sample()
    counts = dict(retired=0, active=0)
    W.scope_authority(N(R=R), s.runtime, s.lease, counts, s.factory, s.old.scope, s.key[0])
    assert counts == dict(retired=1, active=0) and s.checked == [s.lease.archive]


@pytest.mark.parametrize('case', ('after_reset', 'foreign_scope', 'integer_owner', 'bool_frame'))
def test_new_probability_not_counted_as_integer_world(case: str) -> None:
    s = sample()
    counts = dict(retired=0, active=0)
    frame: Any = s.key[0]
    scope = s.old.scope
    if case == 'after_reset': frame = s.lease.empty_evidence.frame + 2
    elif case == 'foreign_scope': scope = s.lease.new_scope
    elif case == 'integer_owner': s.factory.controller.history['1P'] = s.old
    elif case == 'bool_frame': frame = True
    with pytest.raises(AssertionError):
        W.scope_authority(N(R=R), s.runtime, s.lease, counts, s.factory, scope, frame)
    assert counts == dict(retired=0, active=0)
