"""返しの能力を未知テープで平均し、1往復の両側盤面を評価する。"""
from __future__ import annotations

from typing import Any, Callable
from dataclasses import replace
from time import perf_counter
import numpy as np
from src.phase_j._live_ex_frozen.src.indicators_v2 import SEC_PER_HAND
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import Context
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Budget, Plan, plan, tape, SearchBudgetExpired
from src.phase_j._live_ex_frozen.src.prefire_world_v410 import World
from src.phase_j._live_ex_frozen.src.prefire_worker import LOW_SPEC_RATIO

POLICIES = ('main', 'counter')
CONTINUOUS_EPS = 1e-6
MIN_TAPES = 2
EVALUATION_RESERVE_MS = 20.
MS = 1000.


def branch(native: Any, ctx: Context, budget: Budget, sample: int, policy: str) -> tuple[World, dict]:
    """Yを仮定した世界へ戻り、Xの終了直後からZを組む。Y終了後からではない。"""
    receiver, attacker = ctx.receiver, ctx.attacker
    y = Plan(receiver, policy, ctx.input_sec, ctx.ends[attacker]+SEC_PER_HAND)
    if ctx.triggers[receiver] is None:
        pairs = tape(ctx.queues[receiver], ctx.colors, sample, receiver)
        y = plan(native, ctx, ctx.world, receiver, ctx.input_sec, y.deadline, pairs, policy, budget, 'Y')
    else:
        y.fire, y.end = ctx.triggers[receiver], ctx.ends[receiver]
        y.reason = 'already_fired_observed'
    world = ctx.world.copy()
    world.add(y.events)
    start = max(ctx.input_sec, ctx.ends[attacker])
    deadline = y.end if y.end is not None else start
    z = Plan(attacker, 'main', start, deadline, reason='no_second_window')
    if deadline > start:
        pairs = tape(ctx.queues[attacker], ctx.colors, sample, attacker)
        z = plan(native, ctx, world, attacker, start, deadline, pairs, 'main', budget, 'Z')
        world.add(z.events)
    world.finish()
    return world, dict(sample=sample, policy=policy, Y=y.audit(), Z=z.audit(),
                       accounting=world.audit(), observed_reply_fire_sec=ctx.triggers[receiver],
                       ability_reply_fire_sec=y.fire, factual_reply_used=ctx.triggers[receiver] is not None)


def evaluate_branches(worlds: list[World], evaluate_many: Callable[[list], list[float]]) -> list[float]:
    """仮想死亡は連続値の内点まで。死亡証明と0/100%を作らない。"""
    requests, indices, values = [], [], [None]*len(worlds)
    for index, world in enumerate(worlds):
        if any(world.dead):
            values[index] = .5 if all(world.dead) else CONTINUOUS_EPS if world.dead[0] else 1-CONTINUOUS_EPS
        else:
            requests.append(([list(b) for b in world.boards], [world.ledger.pending(i) for i in (0, 1)]))
            indices.append(index)
    predicted = evaluate_many(requests) if requests else []
    for index, probability in zip(indices, predicted, strict=True):
        values[index] = float(np.clip(probability, CONTINUOUS_EPS, 1-CONTINUOUS_EPS))
    return values


def balanced(native: Any, ctx: Context, budget: Budget, policies: tuple) -> tuple:
    """同一テープの全方策が終了した単位だけ採用する。未知1本だけなら棄権する。"""
    by_policy = {p: [] for p in policies}
    completed, cutoff = 0, False
    for sample in range(budget.samples):
        try:
            unit = []
            for policy in policies:
                budget.check()
                unit.append(branch(native, ctx, budget, sample, policy))
            budget.check()
        except SearchBudgetExpired:
            cutoff = True
            break
        for policy, result in zip(policies, unit, strict=True):
            by_policy[policy].append(result)
        completed += 1
    ordered = [item for policy in policies for item in by_policy[policy]]
    return [w for w, _ in ordered], [a for _, a in ordered], completed, cutoff


def solve(native: Any, ctx: Context, budget: Budget, evaluate_many: Callable[[list], list[float]]) -> dict:
    """同じ方策の平均同士を比較。テープごとに都合の良い方策を後選択しない。"""
    policies = POLICIES if ctx.triggers[ctx.receiver] is None else POLICIES[:1]
    limit = (budget.milliseconds-EVALUATION_RESERVE_MS)/MS/LOW_SPEC_RATIO
    budget = replace(budget, deadline=min(budget.deadline, perf_counter()+limit))
    worlds, audits, completed, cutoff = balanced(native, ctx, budget, policies)
    if completed < MIN_TAPES:
        return dict(v_post=None, reason='time_budget_insufficient_tapes', branches=audits,
                    complete=False, completed_tapes=completed, budget_ms=budget.milliseconds,
                    attacker=ctx.attacker, receiver=ctx.receiver)
    invalid = [w.invalid for w in worlds if w.invalid]
    if invalid:
        return dict(v_post=None, reason=invalid[0], branches=audits, complete=False)
    values = evaluate_branches(worlds, evaluate_many)
    means = {p: float(np.mean(values[i*completed:(i+1)*completed])) for i, p in enumerate(policies)}
    chosen = (max if ctx.receiver == 0 else min)(means, key=means.get)
    for audit, probability in zip(audits, values, strict=True):
        audit['p1'] = probability
    return dict(v_post=means[chosen], reason=None, selected_policy=chosen, policy_means=means,
                branches=audits, complete=True, evidence_max=ctx.evidence_max,
                attacker=ctx.attacker, receiver=ctx.receiver,
                samples=completed, requested_samples=budget.samples, cutoff=cutoff,
                budget_ms=budget.milliseconds,
                unknown_distribution='iid_uniform_ordered_colors', common_index_known=False)
