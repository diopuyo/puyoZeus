"""起点Beliefの全候補を読み取り専用で連鎖後へ伝播する。current更新ではない。"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any
import belief as B
from src.scoring import calculate_chain_score

FPS = 60


@dataclass(frozen=True)
class Outcome:
    grid: B.Grid
    chain_count: int
    raw_chain_score: int
    weight: float


@dataclass(frozen=True)
class ProjectedView:
    scope: tuple[Any, ...]
    source_frame: int
    origin_frame: int  # 起点を利用可能になった採録frame。発火発生frameとは別。
    cutoff_frame: int
    origin_token: str
    source_digest: str
    outcomes: tuple[Outcome, ...]
    origin_trigger_sec: float | None = None
    provenance: str = field(default='physics_projected', init=False)
    hidden_joint_support_preserved: bool = field(default=True, init=False)
    native_consumption_applied: bool = field(default=False, init=False)
    accounting_connected: bool = field(default=False, init=False)
    source_producer_authorized: bool = field(default=False, init=False)
    production_permission: bool = field(default=False, init=False)
    quality_gate_clear: bool = field(default=False, init=False)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(asdict(value), sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def project(value: B.Belief, scope: tuple[Any, ...], origin: B.Board, *,
            origin_frame: int, cutoff_frame: int, origin_token: str,
            origin_trigger_sec: float | None = None) -> ProjectedView:
    B.validate(value)
    B.require(scope == value.scope, 'projected_scope')
    B.require(type(origin_frame) is int and type(cutoff_frame) is int
        and value.frame <= origin_frame <= cutoff_frame <= value.deadline, 'projected_clock')
    B.require(type(origin_token) is str and bool(origin_token)
        and origin_token not in value.tokens, 'projected_origin_token')
    B.require(origin_trigger_sec is None or type(origin_trigger_sec) in (float, int)
        and math.isfinite(origin_trigger_sec) and 0 <= origin_trigger_sec <= origin_frame / FPS,
        'projected_trigger_clock')
    visible = B.grid(origin)[B.HIDDEN_ROWS:]
    B.require(all(world.grid[B.HIDDEN_ROWS:] == visible for world in value.worlds),
        'projected_anchor_visible_mismatch')
    before = digest(value)
    simulator = B.ChainSimulator(exclude_hidden_row_from_pop=True)
    merged: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for world in value.worlds:
        result = simulator.simulate(B.Board.from_dict({'grid': world.grid}))
        key = (B.grid(result.final_board), result.chain_count, calculate_chain_score(result).total_score)
        merged[key].append(world.weight)
    B.require(0 < len(merged) <= B.MAX_WORLDS, 'projected_support_limit')
    outcomes = tuple(Outcome(*key, math.fsum(weights)) for key, weights in sorted(merged.items()))
    B.require(abs(math.fsum(row.weight for row in outcomes) - 1.0) <= B.SUM_TOLERANCE,
        'projected_mass')
    B.require(before == digest(value), 'projected_source_mutated')
    return ProjectedView(scope, value.frame, origin_frame, cutoff_frame, origin_token, before, outcomes,
        None if origin_trigger_sec is None else float(origin_trigger_sec))
