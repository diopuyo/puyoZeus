"""接続置換の属性範囲と設定衝突拒否を、長い採録前に確認する。"""
from __future__ import annotations
from contextlib import ExitStack
from types import FunctionType
from typing import Any
import sys
import pytest
import collector_connector_v3 as C


def test_only_connect_replaced_and_copied_globals_dispatch(monkeypatch: Any) -> None:
    original = C.INTEGRATED.OLD
    captured = []
    marker = object()
    def target(stack: Any, state: dict) -> Any:
        captured.append((stack, state))
        return marker
    monkeypatch.setattr(C, 'connect', target)
    with ExitStack() as stack:
        C.install(stack)
        facade = C.INTEGRATED.OLD
        assert set(vars(facade)) == set(vars(original))
        assert all(getattr(facade, k) is v for k, v in vars(original).items() if k != 'connect')
        bound = FunctionType(facade.connect.__code__, dict(vars(facade)))
        state = {'test': True}
        assert bound(stack, state) is marker and captured == [(stack, state)]
    assert C.INTEGRATED.OLD is original and C.DISPATCH not in sys.modules
    assert 'anchor_v2' not in sys.modules


@pytest.mark.parametrize('key', ['enable_phantom_board_guard', 'enable_event_accounting_sidecar'])
def test_explicit_live_conflict_rejected(key: str) -> None:
    with pytest.raises(ValueError, match='live_setting_conflict'):
        C.RAW.merged({key: True}, {key: False})


def test_equal_setting_is_preserved() -> None:
    assert C.RAW.merged({'a': True}, {'a': True, 'b': 2}) == {'a': True, 'b': 2}
