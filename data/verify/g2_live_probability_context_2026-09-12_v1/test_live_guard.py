"""未完成finalizeで実動画へ進まず、cold前の型解決も拒否する。"""
from __future__ import annotations
from contextlib import ExitStack
import sys
from typing import Any
import pytest
import live_adapter as A
import loader as L


@pytest.mark.parametrize('args', [['--mode', 'live'], ['--mode=live']])
def test_live_entry_is_rejected_before_imports(monkeypatch: Any, args: list[str]) -> None:
    monkeypatch.setattr(sys, 'argv', ['entry.py', *args])
    before = dict(sys.modules)
    with ExitStack() as stack, pytest.raises(RuntimeError, match='finalize_not_connected'):
        A.configured(stack)
    assert dict(sys.modules) == before


def test_before_cold_board_rejected(monkeypatch: Any) -> None:
    monkeypatch.delitem(sys.modules, 'src.board', raising=False)
    with pytest.raises(RuntimeError, match='requires_frozen_cold_board'):
        L.bootstrap()
