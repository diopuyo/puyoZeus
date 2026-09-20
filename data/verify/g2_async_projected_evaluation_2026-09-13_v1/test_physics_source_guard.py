"""実ソースから読んだ型で非凍結・原pipeline共有・私有凍結を区別する。"""
import importlib.util
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import check_saved_inputs as INPUT
import session_runtime_binding as B
import src.chain  # 人工fixtureにも原pipeline側の型を明示して対照にする。


def test_source_guard_rejects_current_and_shared_physics(monkeypatch: Any) -> None:
    current = INPUT.S.B
    with pytest.raises(ValueError, match='nonfrozen_board'):
        B.physics(N(B=N(B=current)))
    spec = importlib.util.spec_from_file_location('_guard_actual_frozen_board', B.FROZEN / 'src/board.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    supplied = N(B=N(B=N(Board=module.Board, ChainSimulator=current.ChainSimulator)))
    assert B.physics(supplied) == (module.Board, current.ChainSimulator)
    monkeypatch.setattr(sys.modules['src.chain'], 'ChainSimulator', current.ChainSimulator)
    with pytest.raises(ValueError, match='pipeline_simulator_shared'):
        B.physics(supplied)
