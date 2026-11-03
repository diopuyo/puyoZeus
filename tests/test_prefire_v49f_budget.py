"""速い時計・遅い時計でも決定的経路の探索量が変わらないことを検査する。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from src import prefire_branches, prefire_chain_board, prefire_work_budget as budget
from src.prefire_growth import SideInput
from scripts.prefire_v49e_speed_verify import stable

def test_branch_limit_ignores_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(budget.FLAG, '1')
    calls = []
    def native(board: list, pairs: list, horizon: int, colors: list, **kwargs: object) -> list:
        calls.append((horizon, kwargs['budget_ms']))
        return [(k, k*40, k, k*40, k) for k in range(horizon+1)]
    request = SideInput(1, 0, 2., tuple([0]*78), ((1, 2), (3, 4)), (1, 2, 3, 4), 0, 2., (1.,), 0, 10)
    outputs = []
    for tick in (.00001, 100.):
        times = iter(i*tick for i in range(100))
        monkeypatch.setattr(prefire_branches, 'perf_counter', lambda: next(times))
        with ThreadPoolExecutor(1) as pool:
            outputs.append(stable(prefire_branches.calculate(request, SimpleNamespace(ama_branches_py=native), pool, True)))
    assert outputs[0] == outputs[1]
    assert calls == [(2, None)]*8

def test_strict_repair_limit_inside_round_robin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(budget.FLAG, '1')
    monkeypatch.setattr(budget, 'REPAIR_SIMULATIONS', 3)
    monkeypatch.setattr(prefire_chain_board, 'REPAIR_LIMIT', 3)
    counter = []
    def neighbours(value: dict, *args: object) -> object:
        return iter([([value['origin'][0]+100*k], 0., []) for k in range(1, 10)])
    def simulate(board: list, native: object) -> dict:
        counter.append(board)
        return dict(origin=board, scores=[40])
    monkeypatch.setattr(prefire_chain_board, 'neighbours', neighbours)
    monkeypatch.setattr(prefire_chain_board, 'simulate', simulate)
    monkeypatch.setattr(prefire_chain_board, 'perf_counter', lambda: 1e10)
    prefire_chain_board.repair([dict(origin=[i], scores=[40], log_weight=0) for i in range(8)],
        [dict(count=1, score=80)], [1, 2, 3, 4], [], dict(prob_floor=.01), None)
    assert len(counter) == 3

def test_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(budget.FLAG, raising=False)
    assert not budget.enabled()
    from src.prefire_v49f import PREFIRE_V49F_ENABLED_DEFAULT
    assert not PREFIRE_V49F_ENABLED_DEFAULT
