"""独立レビューのforeign alias反例と正規cacheを検査する。"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

import dependencies as D


def test_own_cached_module_is_reused() -> None:
    first = D.load('candidate')
    assert D.load('candidate') is first


def test_foreign_existing_alias_rejected(monkeypatch: Any) -> None:
    original = D.load('candidate')
    alien = ModuleType('_parent_motion_provider_candidate')
    alien.__file__ = original.__file__
    monkeypatch.setitem(sys.modules, '_parent_motion_provider_candidate', alien)
    with pytest.raises(ValueError, match='foreign_dependency_alias'):
        D.load('candidate')
    assert D.OWNED['candidate'] is original


def test_changed_cached_origin_rejected(monkeypatch: Any) -> None:
    original = D.load('candidate')
    monkeypatch.setattr(original, '__file__', '/not/the/fixed/module.py')
    with pytest.raises(ValueError, match='cached_dependency_origin'):
        D.load('candidate')


def test_removed_owned_alias_is_not_silently_recreated(monkeypatch: Any) -> None:
    original = D.load('candidate')
    monkeypatch.delitem(sys.modules, '_parent_motion_provider_candidate')
    with pytest.raises(ValueError, match='owned_dependency_alias_removed'):
        D.load('candidate')
    assert D.OWNED['candidate'] is original
