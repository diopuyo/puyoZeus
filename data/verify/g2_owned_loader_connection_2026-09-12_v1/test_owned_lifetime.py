"""実ownedと同じ所有契約で正常/失敗/foreignを限定検査する。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace as N
from typing import Any
import pytest
import owned_adapter as R


@dataclass(frozen=True)
class Dependencies:
    side: Any


def fixture(monkeypatch: Any, failure: bool = False) -> tuple:
    original = lambda: Dependencies(side=None)
    module = ModuleType(R.SIDE_ALIAS)
    module.__file__ = str(R.SIDE / 'side_actual_connection.py')
    error = LookupError('original_partial_import')
    def load(*args: Any) -> Any:
        sys.modules[R.SIDE_ALIAS] = module
        if failure:
            raise error
        return module
    owner = N(__file__=str(R.PRIOR / 'loader.py'), dependencies=original,
              bootstrap=lambda: N(load=load))
    monkeypatch.setitem(sys.modules, R.OWNED_ALIAS, owner)
    return owner, original, module, error


@pytest.mark.parametrize('failure', [False, True])
def test_owned_restore(monkeypatch: Any, failure: bool) -> None:
    owner, original, module, error = fixture(monkeypatch, failure)
    generic = R.A.L.dependencies
    if failure:
        with pytest.raises(LookupError) as caught, ExitStack() as stack:
            R.install_side(stack, owner)
            owner.dependencies()
        assert caught.value is error
    else:
        with ExitStack() as stack:
            R.install_side(stack, owner)
            assert owner.dependencies().side is module
            assert owner.dependencies().side is module
    assert R.SIDE_ALIAS not in sys.modules and owner.dependencies is original
    assert R.A.L.dependencies is generic


def test_foreign_side_preserved(monkeypatch: Any) -> None:
    owner, *_ = fixture(monkeypatch)
    foreign = ModuleType(R.SIDE_ALIAS)
    monkeypatch.setitem(sys.modules, R.SIDE_ALIAS, foreign)
    with pytest.raises(RuntimeError, match='foreign_alias'), ExitStack() as stack:
        R.install_side(stack, owner)
    assert sys.modules[R.SIDE_ALIAS] is foreign


def test_replaced_owner_rejected_before_import(monkeypatch: Any) -> None:
    owner, original, *_ = fixture(monkeypatch)
    foreign = N()
    with ExitStack() as stack:
        R.install_side(stack, owner)
        monkeypatch.setitem(sys.modules, R.OWNED_ALIAS, foreign)
        with pytest.raises(RuntimeError, match='owned_loader_identity'):
            owner.dependencies()
    assert owner.dependencies is original and sys.modules[R.OWNED_ALIAS] is foreign
    assert R.SIDE_ALIAS not in sys.modules


def test_foreign_dependency_preserved_on_close(monkeypatch: Any) -> None:
    owner, *_ = fixture(monkeypatch)
    foreign = lambda: None
    with pytest.raises(RuntimeError, match='repair_dependency_foreign'), ExitStack() as stack:
        R.install_side(stack, owner)
        owner.dependencies = foreign
    assert owner.dependencies is foreign
