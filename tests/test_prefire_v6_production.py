"""選択後の本番評価器の経路を確認する。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src import prefire_v6_production as prod
from src.prefire_v5_search import Position, Exchange


def test_static_terminal_only_gfe() -> None:
    state = Position(bytes(78), (1, 2, 3, 4))
    evaluator = object.__new__(prod.ProductionValue)
    evaluator.gfe = lambda *args: .6
    exchange = Exchange((state, state), (0, 0), (0, 0), (False, False))
    assert evaluator(exchange, 0) == .6


def test_firing_uses_s3_and_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    fire = Position(bytes(78), (1, 2, 3, 4), score=40, chains=1)
    wait = Position(bytes(78), (1, 2, 3, 4))
    evaluator = object.__new__(prod.ProductionValue)
    evaluator.gfe = lambda *args: .6
    evaluator.context = SimpleNamespace()
    evaluator.s3_calls = 0
    evaluator.certainties = {}
    calls = []
    def proof(side: int, state: Position, elapsed: float) -> bool:
        calls.append(side)
        return side == 0
    evaluator.proven = proof
    monkeypatch.setattr(prod.layer, 's3_value', lambda *args: .8)
    exchange = Exchange((fire, wait), (0, 0), (0, 0), (False, False))
    assert evaluator(exchange, 0) == pytest.approx(.98)
    assert evaluator.s3_calls == 1 and calls == [0, 1]
