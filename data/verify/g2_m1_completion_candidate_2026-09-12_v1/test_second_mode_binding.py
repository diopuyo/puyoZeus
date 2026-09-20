"""型選択の正常/拒否と既存2P原基準→正常手→遅延反映を結合する。"""
from __future__ import annotations

from types import SimpleNamespace as N
from typing import Any

import pytest
import second_mode_binding as NEW
import test_second_physical as OLD
import test_second_retirement as RETIRE
import test_second_physical_lifetime as LIFETIME
from test_second_physical import saved, live, context, policy, observed, physical


def arrival(physical: Any) -> tuple[Any, Any]:
    class Mode(physical.Mode):
        pass
    first = Mode.__new__(Mode)
    first.connection = N(binding=N(scope=('1P',)))
    first.arrival_ledger = object()
    return first, N(Mode=Mode, BASE=physical, B=physical.B)


@pytest.mark.parametrize('delay', [False, True])
def test_selected_second_original_normal_hand(observed: Any, policy: Any, physical: Any, delay: bool) -> None:
    first, module = arrival(physical)
    ledger = first.arrival_ledger
    selected = NEW.select(first, N(Mode=module.Mode), module)
    assert selected.Mode is physical.Mode and first.arrival_ledger is ledger
    OLD.test_normal_hand_once(observed, policy, selected, delay)


@pytest.mark.parametrize('change', ['first_type', 'supplied', 'side', 'belief'])
def test_foreign_route_rejected(physical: Any, change: str) -> None:
    first, module = arrival(physical)
    supplied = N(Mode=module.Mode)
    if change == 'first_type': first = N(connection=first.connection)
    elif change == 'supplied': supplied.Mode = physical.Mode
    elif change == 'side': first.connection.binding.scope = ('2P',)
    elif change == 'belief': module.BASE = N(Mode=physical.Mode, B=object())
    with pytest.raises(ValueError, match='second_physical'):
        NEW.select(first, supplied, module)


def test_session_constructor_preserves_context_and_frames(physical: Any) -> None:
    first, module = arrival(physical)
    class Session:
        def __init__(self, stack: Any, context: Any, policy: Any, physical: Any,
                     contract: Any, members: Any, frames: tuple[int, ...]) -> None:
            self.received = (stack, context, policy, physical, contract, members, frames)
    selected = NEW.session_class(N(Session=Session), module)
    context = {'state': {'probabilistic_tracking_mode': first}}
    values = tuple(object() for _ in range(4))
    instance = selected(values[0], context, values[1], N(Mode=module.Mode), values[2], values[3], (10, 20))
    assert instance.received == (values[0], context, values[1], instance.received[3], values[2], values[3], (10, 20))
    assert instance.received[3].Mode is physical.Mode
    assert context['state']['probabilistic_tracking_mode'] is first


@pytest.mark.parametrize('case,flag', [('retire', False), ('retire', True),
    ('boundary', False), ('boundary', True), ('consumption', None), ('writer', None), ('body', None)])
def test_selected_second_lifetime(observed: Any, policy: Any, physical: Any, case: str, flag: Any) -> None:
    first, module = arrival(physical)
    selected = NEW.select(first, N(Mode=module.Mode), module)
    if case == 'retire': RETIRE.test_retire_preserve_and_rebind(observed, policy, selected, flag)
    elif case == 'boundary': RETIRE.test_reset_inside_original_call_holds_boundary(observed, policy, selected, flag)
    elif case == 'consumption': RETIRE.test_retirement_call_consumption_is_preserved(observed, policy, selected)
    elif case == 'writer': LIFETIME.test_original_writer_failure_blocks_publication(observed, policy, selected)
    else: LIFETIME.test_body_exception_and_hook_restoration(observed, policy, selected)


@pytest.mark.parametrize('change', [None, 'owner', 'first_type', 'second_type', 'side'])
def test_session_rechecks_binding_before_basis(physical: Any, monkeypatch: Any, change: Any) -> None:
    first, module = arrival(physical)
    class Session:
        def __init__(self, *args: Any) -> None:
            self.basis_calls = 0
        def basis(self) -> None:
            self.basis_calls += 1
    context = {'state': {'probabilistic_tracking_mode': first}}
    selected = NEW.session_class(N(Session=Session), module)
    instance = selected(None, context, None, N(Mode=module.Mode), None, None, (10, 20))
    if change == 'owner': context['state']['probabilistic_tracking_mode'] = object()
    elif change == 'first_type': module.Mode = object
    elif change == 'second_type': monkeypatch.setattr(module.BASE, 'Mode', object)
    elif change == 'side': first.connection.binding.scope = ('2P',)
    if change is None:
        instance.basis()
        assert instance.basis_calls == 1
    else:
        with pytest.raises(ValueError, match='second_physical'): instance.basis()
        assert instance.basis_calls == 0
