"""一手の着地と連鎖を一つの確率遷移にする。原列挙・原連鎖物理を再用する。"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field, replace
import math
from typing import Any
import belief as B
import hidden_landing as H


@dataclass(frozen=True)
class Result:
    retained_mass: float
    hypotheses: int
    source_support: int
    final_support: int
    chain_count: int | None
    prior_assumption: str
    chain_probabilities: tuple[tuple[int, float], ...]
    placement_prior_calibrated: bool = field(default=False, init=False)
    physical_certified: bool = field(default=False, init=False)
    quality_gate_clear: bool = field(default=False, init=False)


def hypotheses(world: B.World, pair: tuple[int, int], prior: H.PlacementPrior) -> tuple[Any, ...]:
    values = H.enumerate_hypotheses(world.grid, pair)
    B.require(bool(values), 'transition_no_placement')
    weights = tuple(prior.weight(value) for value in values)
    B.require(all(type(w) is float and math.isfinite(w) and w > 0 for w in weights), 'transition_prior')
    total = math.fsum(weights)
    B.require(math.isfinite(total) and total > 0, 'transition_prior_total')
    return tuple((value, world.weight * weight / total) for value, weight in zip(values, weights))


def simulate(simulator: Any, placed: B.Grid, origin: B.Grid | None,
             observed: B.Grid, chain_count: int | None) -> tuple[B.Grid, int] | None:
    if origin is not None and placed[B.HIDDEN_ROWS:] != origin[B.HIDDEN_ROWS:]:
        return None
    result = simulator.simulate(B.Board.from_dict({'grid': placed}))
    if chain_count is not None and result.chain_count != chain_count:
        return None
    final = B.grid(result.final_board)
    return (final, result.chain_count) if final[B.HIDDEN_ROWS:] == observed[B.HIDDEN_ROWS:] else None


def run(value: B.Belief, scope: tuple[Any, ...], frame: int, occurrence_token: str,
        pair: tuple[int, int], observed: B.Board, chain_count: int | None,
        prior: H.PlacementPrior, *, origin_observed: B.Board | None = None) -> tuple[B.Belief, Result]:
    B.validate(value)
    B.require(scope == value.scope and type(frame) is int and value.frame < frame <= value.deadline,
              'transition_scope_clock')
    B.require(type(occurrence_token) is str and bool(occurrence_token)
              and occurrence_token not in value.tokens, 'transition_token')
    B.require(type(pair) is tuple and len(pair) == 2
              and all(type(c) is int and c in B.PIECE_COLORS for c in pair), 'transition_pair')
    B.require(chain_count is None or type(chain_count) is int and chain_count >= 0, 'transition_chain_count')
    B.require(type(prior) is H.PlacementPrior and prior.calibrated is False
              and type(prior.assumption) is str and bool(prior.assumption), 'transition_prior_type')
    expected = B.grid(observed)
    origin = None if origin_observed is None else B.grid(origin_observed)
    B.require(chain_count == 0 or origin is not None, 'transition_chain_origin_missing')
    simulator = B.ChainSimulator(exclude_hidden_row_from_pop=True)
    merged: dict[B.Grid, list[float]] = defaultdict(list)
    chains: dict[int, list[float]] = defaultdict(list)
    count = 0
    for world in value.worlds:
        for hypothesis, weight in hypotheses(world, pair, prior):
            count += 1
            placed = H.apply_hypothesis(world.grid, hypothesis)
            final = simulate(simulator, placed, origin, expected, chain_count)
            if final is not None:
                merged[final[0]].append(weight)
                chains[final[1]].append(weight)
    return finish(value, frame, occurrence_token, merged, count, chain_count, prior, chains)


def finish(value: B.Belief, frame: int, token: str, merged: dict[B.Grid, list[float]],
           count: int, chain_count: int | None, prior: H.PlacementPrior,
           chains: dict[int, list[float]]) -> tuple[B.Belief, Result]:
    B.require(bool(merged), 'transition_zero_support')
    B.require(len(merged) <= B.MAX_WORLDS, 'transition_support_limit')
    weights = {key: math.fsum(parts) for key, parts in merged.items()}
    mass = math.fsum(weights.values())
    B.require(math.isfinite(mass) and mass > 0, 'transition_mass')
    worlds = tuple(B.World(key, weight / mass) for key, weight in sorted(weights.items()))
    result = replace(value, frame=frame, worlds=worlds, tokens=value.tokens + (token,))
    B.validate(result)
    probabilities = tuple((key, math.fsum(parts) / mass) for key, parts in sorted(chains.items()))
    return result, Result(mass, count, len(value.worlds), len(worlds), chain_count, prior.assumption, probabilities)
