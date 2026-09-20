"""基準に既に存在する連鎖を、次手を追加せず条件付けする私有演算。"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any
import belief as B


@dataclass(frozen=True)
class Report:
    retained_mass: float
    chain_probabilities: tuple[tuple[int, float], ...]
    physical_certified: bool = field(default=False, init=False)
    next_consumed: bool = field(default=False, init=False)
    quality_gate_clear: bool = field(default=False, init=False)


def settle(value: B.Belief, scope: tuple[Any, ...], frame: int, token: str,
           origin: B.Board, observed: B.Board) -> tuple[B.Belief, Report]:
    B.validate(value)
    before = B.grid(origin)[B.HIDDEN_ROWS:]
    B.require(all(w.grid[B.HIDDEN_ROWS:] == before for w in value.worlds), 'basis_origin_visible_mismatch')
    expected = B.grid(observed)[B.HIDDEN_ROWS:]
    simulator = B.ChainSimulator(exclude_hidden_row_from_pop=True)
    results: dict[B.Grid, Any] = {}
    chains: dict[int, list[float]] = defaultdict(list)
    for world in value.worlds:
        result = simulator.simulate(B.Board.from_dict({'grid': world.grid}))
        grid = B.grid(result.final_board)
        # 発火した原originを対象とする。推定連鎖数では支持を絞らない。
        accepted = result.chain_count > 0 and grid[B.HIDDEN_ROWS:] == expected
        results[world.grid] = grid if accepted else None
        if accepted:
            chains[result.chain_count].append(world.weight)
    following, mass = B.advance(value, scope, frame, token, observed, results.__getitem__)
    distribution = tuple((count, math.fsum(weights) / mass) for count, weights in sorted(chains.items()))
    return following, Report(mass, distribution)
