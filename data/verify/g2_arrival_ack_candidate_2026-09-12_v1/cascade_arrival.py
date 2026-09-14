"""未観測の連鎖後状態を公開せず、到来済み一手と合成して現可視で条件付けする。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import math
from typing import Any

import belief as B
import transition_candidates as T
import ledger as L


@dataclass(frozen=True)
class Report:
    chain_selected_mass: float
    combined_retained_mass: float
    next_hand_report: Any
    intermediate_published: bool = field(default=False, init=False)
    physical_certified: bool = field(default=False, init=False)
    quality_gate_clear: bool = field(default=False, init=False)


def latent(value: B.Belief, origin: B.Board) -> tuple[B.Belief, float]:
    B.validate(value)
    visible = B.grid(origin)[B.HIDDEN_ROWS:]
    B.require(all(w.grid[B.HIDDEN_ROWS:] == visible for w in value.worlds), 'basis_origin_visible_mismatch')
    simulator = B.ChainSimulator(exclude_hidden_row_from_pop=True)
    merged: dict[B.Grid, list[float]] = defaultdict(list)
    for world in value.worlds:
        result = simulator.simulate(B.Board.from_dict({'grid': world.grid}))
        if result.chain_count > 0:
            merged[B.grid(result.final_board)].append(world.weight)
    B.require(bool(merged), 'basis_origin_without_existing_chain')
    weights = {grid: math.fsum(parts) for grid, parts in merged.items()}
    mass = math.fsum(weights.values())
    temporary = replace(value, worlds=tuple(B.World(grid, weight / mass) for grid, weight in sorted(weights.items())))
    B.validate(temporary)
    return temporary, mass


def run(value: B.Belief, ledger: L.Ledger, frame: int, arrival: L.Arrival,
        origin: B.Board, observed: B.Board, prior: Any) -> tuple[B.Belief, Report]:
    L.check(ledger)
    B.require(ledger.scope == value.scope and ledger.start == value.frame and type(frame) is int
              and ledger.clock <= frame <= ledger.deadline == value.deadline, 'arrival_composite_clock_scope')
    B.require(not ledger.applied and ledger.arrivals and ledger.arrivals[0] == arrival,
              'basis_composite_first_arrival')
    B.require(arrival.frame <= frame and arrival.token not in value.tokens, 'arrival_composite_future_or_reuse')
    temporary, mass = latent(value, origin)
    # latentは局所変数だけ。モデル/保存/Registryへ単独で公開しない。
    following, report = T.run(temporary, value.scope, frame, arrival.token, arrival.pair,
                              observed, 0, prior)
    return following, Report(mass, mass * report.retained_mass, report)
