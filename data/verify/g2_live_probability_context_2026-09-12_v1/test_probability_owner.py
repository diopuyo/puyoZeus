"""原Lease型＋人工archive/Registryで終了権限の境界を検査。実finalize未到達。"""
from __future__ import annotations
from copy import copy
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import probability_owner as P

BASE = Path(__file__).resolve().parent.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1'
sys.path.insert(0, str(BASE / 'qa_retired_v1'))
from test_retired import carrier
import retired_completion as R


def sample() -> Any:
    lease, factory, old, value, key = carrier()
    recovery = lease.recovery
    recovery.baseline_count, recovery.error, recovery.failure = 0, None, None
    factory.controller.history.clear()
    lease.new_generation, lease.guard = lease.new_scope[5], N(binding=None)
    lease.empty_new_binding = lease.empty_new_owner = None
    lease.outer_calls = lease.native_calls = lease.depth = 0
    lease.events = [dict(kind='reset_qualified', frame=102, clock=102 / 60, old=lease.archive.receipt()),
                    dict(kind='side_reset_waiting', scope=lease.new_scope, outer_calls=0, native_calls=0)]
    binding = N(scope=lease.new_scope, initial_call_token='step:102')
    belief = N(scope=lease.new_scope, frame=102, deadline=120)
    registry = N(factory=factory, current=lambda b: belief if b is binding else None)
    connection = N(registry=registry, binding=binding, recovery=recovery, deadline=120,
                   observer=N(recovery=recovery, error=None, gate=N(candidate=N(
                       scope=lease.new_scope, frame=102, source_call_token='step:102'))))
    mode = N(connection=connection, native=N(connection=connection, last_frame=104), error=None,
             activation=dict(frame=102, source_call_token='step:102', tracking_deadline=120,
                 prior_pending=dict(epoch=1, used=False, empty_retirement=lease.retirement)))
    checked: list[Any] = []
    runtime = {P.REGISTRY_KEY: registry, P.ARCHIVE_KEY: N(verify=checked.append,
               original=lease.archive, private=None),
               'probabilistic_basis_connection': connection, 'probabilistic_tracking_mode': mode}
    return N(lease=lease, factory=factory, old=old, value=value, key=key,
             runtime=runtime, connection=connection, mode=mode, checked=checked)


def invoke(s: Any) -> bool:
    return P.retired_owner(R, s.runtime, s.lease, s.factory, s.old, s.value, s.value, s.key)


def test_probability_owner_keeps_archive_and_does_not_reenter_integer_wait() -> None:
    s = sample()
    s.lease.observe_wait = lambda: pytest.fail('復元後の整数待機へ再突入')
    assert invoke(s) and s.checked == [s.lease.archive]


@pytest.mark.parametrize('case', ('waiting', 'pending', 'integer', 'owner', 'receipt', 'retirement',
    'registry', 'binding', 'generation', 'activation', 'consumed_pending', 'future', 'error', 'deadline',
    'qualified_frame', 'qualified_clock', 'candidate_frame', 'stale_integer_binding'))
def test_wrong_owner_or_transition_rejected(case: str) -> None:
    s = sample()
    if case == 'waiting': s.lease.waiting = True
    elif case == 'pending': s.lease.recovery.pending = {}
    elif case == 'integer': s.factory.controller.history['1P'] = s.old
    elif case == 'owner': s.old.owner.state = copy(s.value)
    elif case == 'receipt': s.lease.empty_evidence.receipt_sha = 'wrong'
    elif case == 'retirement': s.lease.retirement['missing_count'] = 'KNOWN'
    elif case == 'registry': s.runtime[P.REGISTRY_KEY] = object()
    elif case == 'binding': s.connection.binding = copy(s.connection.binding)
    elif case == 'generation': s.lease.new_generation += 1
    elif case == 'activation': s.mode.activation['source_call_token'] = 'other'
    elif case == 'consumed_pending': s.mode.activation['prior_pending']['used'] = True
    elif case == 'future': s.mode.native.last_frame = 101
    elif case == 'error': s.connection.observer.error = 'failed'
    elif case == 'deadline': s.connection.deadline = 999
    elif case == 'qualified_frame': s.lease.events[0]['frame'] += 2
    elif case == 'qualified_clock': s.lease.events[0]['clock'] += 1
    elif case == 'candidate_frame': s.connection.observer.gate.candidate.frame += 2
    elif case == 'stale_integer_binding': s.lease.empty_new_binding = s.old
    with pytest.raises(AssertionError):
        invoke(s)


def test_original_integer_normal_control_unchanged() -> None:
    lease, factory, old, value, key = carrier()
    assert R.retired_owner(lease, factory, old, value, value, key)
