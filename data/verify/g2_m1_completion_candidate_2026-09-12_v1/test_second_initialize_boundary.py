"""元initializeで未発行HOLDと後続正常登録を検査。世代差は人工票の負例。"""
from __future__ import annotations

import json
from typing import Any

import pytest
from test_second_basis import saved, live, context, policy, observed
import second_basis as B
import second_basis_boundary as NEW


@pytest.mark.parametrize('case', ['action', 'reset', 'decrease', 'boolean', 'pipe', 'grid'])
def test_original_initialize_no_boundary_publication(observed: Any, policy: Any,
                                                     monkeypatch: Any, case: str) -> None:
    o = observed
    old = o.witness.rows['2P']
    step = json.loads(old)
    step['generation']['action_revision'] = 0
    step['generation_after']['action_revision'] = 1
    if case == 'reset': step['generation']['reset_epoch'] -= 1
    elif case == 'decrease': step['generation']['action_revision'] = 2
    elif case == 'boolean': step['generation']['action_revision'] = False
    elif case == 'pipe': step['pipe_object_id'] += 1
    elif case == 'grid': step['returned']['confirmed']['grid'][12][0] = 99
    o.witness.rows['2P'] = json.dumps(step)
    monkeypatch.setattr(B, 'qualify', NEW.wrapped(B.qualify, B))
    with pytest.raises(ValueError) as error:
        B.initialize(o.evidence, o.witness, o.pipe, o.registry, o.factory, 35410, policy)
    assert isinstance(error.value, B.BasisHold) is (case == 'action')
    assert not o.registry._bindings
    # 同じ人工fixtureの正常対照へ戻す。実動画の次frameへの復帰とは呼ばない。
    o.witness.rows['2P'] = old
    binding, receipt = B.initialize(o.evidence, o.witness, o.pipe, o.registry, o.factory, 35410, policy)
    assert o.registry.current(binding).frame == receipt['state']['frame']
    assert len(o.registry._bindings) == 1
