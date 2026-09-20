"""元Lease型の人工carrierで片側契約と旧全体reset契約を分離する。"""
from typing import Any
from types import SimpleNamespace as N
import pytest
from test_probability_owner import R, sample, invoke


def test_same_side_carrier_old_rejects_new_accepts() -> None:
    s = sample()
    with pytest.raises(AssertionError):
        R.retired_owner(s.lease, s.factory, s.old, s.value, s.value, s.key)
    assert invoke(s)


@pytest.mark.parametrize('case', ('whole_reset', 'missing', 'duplicate', 'scope', 'depth',
    'zero_reset', 'double_reset', 'generation_skip', 'mixed', 'order', 'qualified_archive'))
def test_bad_side_contract_rejected(case: str) -> None:
    s = sample()
    if case == 'whole_reset': s.lease.outer_calls = s.lease.native_calls = 1
    elif case == 'missing': s.lease.events.pop()
    elif case == 'duplicate': s.lease.events.append(dict(s.lease.events[-1]))
    elif case == 'scope': s.lease.events[-1]['scope'] = ('foreign',)
    elif case == 'depth': s.lease.depth = 1
    elif case == 'zero_reset': s.lease.recovery.reset_count = 0
    elif case == 'double_reset': s.lease.recovery.reset_count = 2
    elif case == 'mixed': s.lease.events.append(dict(kind='reset_waiting'))
    elif case == 'order': s.lease.events.reverse()
    elif case == 'qualified_archive': s.lease.events[0]['old'] = dict(forged=True)
    elif case == 'generation_skip':
        scope = (*s.lease.new_scope[:5], s.old.scope[5] + 2, '1P')
        s.lease.new_scope, s.lease.new_generation = scope, scope[5]
        s.lease.events[-1]['scope'] = scope
        current = s.connection.registry.current(s.connection.binding)
        current.scope = s.connection.binding.scope = s.connection.observer.gate.candidate.scope = scope
        s.lease.recovery.evidence = N(scope=lambda *args: scope)
    with pytest.raises(AssertionError):
        invoke(s)
