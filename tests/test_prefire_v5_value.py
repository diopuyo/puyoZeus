"""局面価値・台帳の入力契約と最善応手の選択。"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src import prefire_v5_search as search
from src import prefire_v5_value as value
from tests.test_prefire_v5_search import position


@pytest.mark.parametrize('flag', [False, True])
def test_observed_ledger_even_if_display_flag_off(flag: bool) -> None:
    safety = SimpleNamespace(ledger_enabled=flag, ledger=SimpleNamespace(pending=(13, 27)))
    overlay = SimpleNamespace(_landing_projection=SimpleNamespace(safety=safety))
    assert value.pending_counts(overlay) == (13, 27)


def test_missing_pending_is_missing() -> None:
    overlay = SimpleNamespace(_landing_projection=SimpleNamespace(safety=None),
                              _snapshots=[(0.0, SimpleNamespace(net_balance_capped=0))])
    assert value.pending_counts(overlay) is None


@pytest.mark.parametrize('attacker, expected', [(0, 0.4), (1, 0.7)])
def test_minimax_independent_matrix(monkeypatch: pytest.MonkeyPatch, attacker: int, expected: float) -> None:
    options = tuple(replace(position(), path=((idx, 0),), consumed=1) for idx in (0, 1))
    monkeypatch.setattr(search, 'candidates', lambda *args: options)
    matrix = np.array([[0.1, 0.8], [0.7, 0.4]])
    def evaluator(exchange: search.Exchange, elapsed: float) -> float:
        return float(matrix[exchange.sides[0].path[0][0], exchange.sides[1].path[0][0]])
    result = value.choose((position(), position()), attacker, 0.0, evaluator)
    assert result[0] == expected


def test_missing_response_does_not_become_zero() -> None:
    def unreachable(exchange: search.Exchange, elapsed: float) -> float:
        raise AssertionError('欠測で評価器を呼ばない')
    assert value.choose((position(), position(())), 0, 0.0, unreachable) is None


def test_wait_value_reads_placed_board() -> None:
    def evaluator(exchange: search.Exchange, elapsed: float) -> float:
        return np.count_nonzero(np.frombuffer(exchange.sides[0].board, np.int8)) / search.sim.CELLS
    result = value.choose((position(), position()), 0, 0.0, evaluator, depth=1)
    assert result[0] == 2 / search.sim.CELLS
