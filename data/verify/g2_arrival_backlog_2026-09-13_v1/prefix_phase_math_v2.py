"""手数・消去前後ごとの条件付き分布。異なる手数間の確率や公開権限は作らない。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any
import split_landing as S

SETTLED = 'settled'
PREPOP = 'prepop'


@dataclass(frozen=True)
class Family:
    prefix: int
    phase: str
    value: Any
    mixture_weight: None = field(default=None, init=False)
    source_qualification_connected: bool = field(default=False, init=False)
    publication_permission: bool = field(default=False, init=False)


def retain(engine: Any, buckets: dict, prefix: int, phase: str,
           grid: tuple, weight: float, expected: tuple) -> None:
    if grid[engine.B.HIDDEN_ROWS:] != expected[engine.B.HIDDEN_ROWS:]:
        return
    buckets[(prefix, phase)][grid].append(weight)
    engine.B.require(sum(len(grids) for grids in buckets.values()) <= engine.B.MAX_WORLDS,
        'prefix_phase_support_limit')


def walk(engine: Any, grid: tuple, weight: float, prefix: int, arrivals: tuple,
         frame: int, expected: tuple, fired: frozenset, prior: Any, buckets: dict) -> None:
    """未観測中間集合は保存せず深さ優先列挙。未来の到来や未観測発火を使わない。"""
    if not any(arrival.token in fired for arrival in arrivals[prefix:]):
        retain(engine, buckets, prefix, SETTLED, grid, weight, expected)
    if prefix == len(arrivals) or arrivals[prefix].frame > frame:
        return
    arrival = arrivals[prefix]
    simulator = engine.B.ChainSimulator(exclude_hidden_row_from_pop=True)
    world = engine.B.World(grid, weight)
    for candidate, share in S.weighted_candidates(engine.H, world, arrival.pair, prior):
        placed = S.apply_candidate(engine.H, grid, candidate)
        result = simulator.simulate(engine.B.Board.from_dict({'grid': placed}))
        if result.chain_count > 0 and arrival.token not in fired:
            if not any(later.token in fired for later in arrivals[prefix + 1:]):
                retain(engine, buckets, prefix + 1, PREPOP, placed, share, expected)
            continue
        if arrival.token in fired and result.chain_count == 0:
            continue
        walk(engine, engine.B.grid(result.final_board), share, prefix + 1, arrivals,
            frame, expected, fired, prior, buckets)


def normalized(engine: Any, family: Family, arrivals: tuple, frame: int, buckets: dict) -> tuple:
    result = []
    for (prefix, phase), grids in sorted(buckets.items()):
        weights = {grid: math.fsum(values) for grid, values in grids.items()}
        total = math.fsum(weights.values())
        engine.B.require(math.isfinite(total) and total > 0, 'prefix_phase_mass')
        worlds = tuple(engine.B.World(grid, weight / total) for grid, weight in sorted(weights.items()))
        old = family.value
        tokens = old.tokens + tuple(arrival.token for arrival in arrivals[family.prefix:prefix])
        value = engine.B.Belief(old.scope, frame, old.deadline, worlds, tokens)
        engine.B.validate(value)
        result.append(Family(prefix, phase, value))
    return tuple(result)


def condition(engine: Any, family: Family, arrivals: tuple, frame: int,
              observed: Any, fired: frozenset, prior: Any) -> tuple:
    """一つの条件付きfamilyだけを更新。family間の質量を合算してはいけない。"""
    value = family.value
    engine.B.validate(value)
    engine.B.require(type(family) is Family and family.phase in (SETTLED, PREPOP)
        and type(family.prefix) is int and 0 <= family.prefix <= len(arrivals), 'prefix_phase_family')
    engine.B.require(type(frame) is int and value.frame < frame <= value.deadline, 'prefix_phase_clock')
    engine.B.require(type(arrivals) is tuple and type(fired) is frozenset, 'prefix_phase_inputs')
    engine.B.require(all(arrival.scope == value.scope for arrival in arrivals), 'prefix_phase_scope')
    tokens = tuple(arrival.token for arrival in arrivals)
    engine.B.require(len(tokens) == len(set(tokens)) and fired <= frozenset(tokens), 'prefix_phase_tokens')
    engine.B.require(all(arrival.frame <= frame for arrival in arrivals if arrival.token in fired),
        'prefix_phase_future_fire')
    engine.B.require(family.prefix == 0 or value.tokens[-family.prefix:] == tokens[:family.prefix],
        'prefix_phase_applied_identity')
    engine.B.require(all(arrival.frame <= value.frame for arrival in arrivals[:family.prefix]),
        'prefix_phase_applied_clock')
    engine.B.require(all(arrivals[i].frame < arrivals[i + 1].frame for i in range(len(arrivals) - 1)),
        'prefix_phase_arrival_order')
    expected = engine.B.grid(observed)
    buckets = defaultdict(lambda: defaultdict(list))
    simulator = engine.B.ChainSimulator(exclude_hidden_row_from_pop=True)
    for world in value.worlds:
        grid = world.grid
        groups = simulator.find_erasable_groups(engine.B.Board.from_dict({'grid': grid}))
        engine.B.require(bool(groups) == (family.phase == PREPOP), 'prefix_phase_input_physics')
        if family.phase == PREPOP:
            engine.B.require(family.prefix > 0, 'prefix_phase_prepop_without_hand')
            if arrivals[family.prefix - 1].token not in fired:
                if not any(later.token in fired for later in arrivals[family.prefix:]):
                    retain(engine, buckets, family.prefix, PREPOP, grid, world.weight, expected)
                continue
            result = simulator.simulate(engine.B.Board.from_dict({'grid': grid}))
            engine.B.require(result.chain_count > 0, 'prefix_phase_prepop_without_group')
            grid = engine.B.grid(result.final_board)
        walk(engine, grid, world.weight, family.prefix, arrivals, frame, expected, fired, prior, buckets)
    return normalized(engine, family, arrivals, frame, buckets)
