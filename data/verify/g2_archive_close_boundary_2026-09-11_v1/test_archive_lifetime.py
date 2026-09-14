"""元Archive全検査を登録解除後にも通し、値/参照/新型の改変を拒否する。"""
from __future__ import annotations
import importlib.util
import sys
from typing import Any
import pytest
import archive_lifetime as L
import test_original_close as F

modules, fixture, actual = F.modules, F.fixture, F.actual
REASONS = dict(basis_value='empty_archive_basis_changed', column_value='empty_archive_column_changed',
               reference='empty_archive_reference_entry', new_type='empty_archive_basis_type')


@pytest.mark.parametrize('fault', ['none', 'basis_value', 'column_value', 'reference', 'new_type'])
def test_all_checks_after_module_unregistration(actual: Any, fault: str) -> None:
    control, key = F.F.registry(actual)
    archive = F.F.X.Archive(control, actual.binding)
    verifier = L.Verifier(archive)
    name = type(actual.basis).__module__
    original = sys.modules.pop(name)
    try:
        if fault == 'basis_value': actual.basis.frame += 2
        if fault == 'column_value': control.empty_tail_prepop_checks[0]['saved_only'] = False
        if fault == 'reference': control.empty_completion_refs[key] = tuple(list(control.empty_completion_refs[key]))
        if fault == 'new_type':
            spec = importlib.util.spec_from_file_location(name, original.__file__)
            replacement = importlib.util.module_from_spec(spec)
            sys.modules[name] = replacement
            spec.loader.exec_module(replacement)
            assert replacement.Basis is not original.Basis
            actual.basis.__class__ = replacement.Basis
        expected = sys.modules.get(name)
        if fault == 'none':
            verifier.verify(archive)
        else:
            with pytest.raises(AssertionError, match=REASONS[fault]):
                verifier.verify(archive)
        assert sys.modules.get(name) is expected  # verifierによる再登録・書換えは禁止。
    finally:
        sys.modules[name] = original


def test_foreign_archive_is_rejected(actual: Any) -> None:
    archive = F.F.X.Archive(actual.local.controller, actual.binding)
    verifier = L.Verifier(archive)
    other = F.F.X.Archive(actual.local.controller, actual.binding)
    with pytest.raises(AssertionError):
        verifier.verify(other)


def test_all_private_modules_unregistered(actual: Any) -> None:
    F.F.registry(actual)
    archive = F.F.X.Archive(actual.local.controller, actual.binding)
    verifier = L.Verifier(archive)
    removed = {name: module for name, module in list(sys.modules.items())
               if '/data/verify/' in str(getattr(module, '__file__', '')).replace('\\', '/')}
    assert type(actual.basis).__module__ in removed and type(archive).__module__ in removed
    try:
        for name in removed: sys.modules.pop(name)
        verifier.verify(archive)
        assert not any(name in sys.modules for name in removed)
    finally:
        sys.modules.update(removed)
