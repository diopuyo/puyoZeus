"""時刻付きYを落とさずZまで評価。同じ手番方策をテープ平均してから選ぶ。"""
from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
from time import perf_counter
from typing import Any, Callable
import numpy as np
from src.phase_j._live_ex_frozen.src.indicators_v2 import SEC_PER_HAND
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import Context
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Budget, SearchBudgetExpired
from src.phase_j._live_ex_frozen.src.prefire_search_v411 import RecordedPlan, guard, BoardMismatch
from src.phase_j._live_ex_frozen.src.prefire_search_v412 import candidates, NO_FIRE
from src.phase_j._live_ex_frozen.src.prefire_second_v411 import validate, surplus
from src.phase_j._live_ex_frozen.src.prefire_second_v410 import evaluate_branches, EVALUATION_RESERVE_MS, MIN_TAPES, MS
from src.phase_j._live_ex_frozen.src.prefire_tsumo_v411 import sample_tape
from src.phase_j._live_ex_frozen.src.prefire_worker import LOW_SPEC_RATIO


def y_candidates(native: Any, ctx: Context, budget: Budget, tapes: list) -> tuple:
    if ctx.triggers[ctx.receiver] is not None:
        y = RecordedPlan(ctx.receiver, 'observed', ctx.input_sec, ctx.ends[ctx.attacker]+SEC_PER_HAND)
        y.fire, y.end, y.reason = ctx.triggers[ctx.receiver], ctx.ends[ctx.receiver], 'already_fired_observed'
        return {'observed': y}, True
    return candidates(native, ctx, ctx.world, ctx.receiver, ctx.input_sec,
                      ctx.ends[ctx.attacker]+SEC_PER_HAND, tapes[ctx.receiver], budget, 'Y')


def z_candidates(native: Any, ctx: Context, budget: Budget, tapes: list, y: RecordedPlan) -> tuple:
    world = ctx.world.copy(); world.add(y.events)
    start = max(ctx.input_sec, ctx.ends[ctx.attacker])
    deadline = y.end if y.end is not None else start
    return candidates(native, ctx, world, ctx.attacker, start, deadline, tapes[ctx.attacker], budget, 'Z')


def finish(ctx: Context, y: RecordedPlan, z: RecordedPlan) -> tuple:
    world = ctx.world.copy(); world.add(y.events); world.add(z.events)
    events = list(world.events)
    world.finish()
    for raw in world.boards:
        guard(raw, ctx.colors)
    record = dict(Y=y.audit(), Z=z.audit(), accounting=world.audit(), events=[asdict(e) for e in events],
        factual_reply_used=ctx.triggers[ctx.receiver] is not None,
        observed_reply_fire_sec=ctx.triggers[ctx.receiver], ability_reply_fire_sec=y.fire,
        y_surplus=surplus(world, events, ctx.receiver,
                          'Y' if ctx.triggers[ctx.receiver] is None else f'C{ctx.receiver}'))
    return world, record


def archive(pools: list, label: str, parent: str | None = None) -> list:
    return [dict(sample=i, label=label, parent=parent, policy=k, plan=p.audit())
            for i, pool in enumerate(pools) for k, p in pool.items()]


def align(pools: list) -> tuple[list, list]:
    """指定手番で発火できないテープは非発火を続ける。同じ方策の失敗も平均に入れる。"""
    aligned = [dict(p) for p in pools]
    union = set.union(*(set(p) for p in aligned))
    for policy in union:
        for pool in aligned:
            if policy not in pool and NO_FIRE in pool:
                fallback = deepcopy(pool[NO_FIRE])
                fallback.policy = policy
                fallback.reason = 'scheduled_fire_unavailable'
                pool[policy] = fallback
    common = sorted(set.intersection(*(set(p) for p in aligned)))
    return aligned, common


def collect(native: Any, ctx: Context, budget: Budget, knowledge: dict) -> tuple:
    tapes = [[sample_tape(knowledge['sides'][s], ctx.colors, i, s) for s in (0, 1)]
             for i in range(budget.samples)]
    ypools = [y_candidates(native, ctx, budget, tape) for tape in tapes]
    saved = archive([p for p, _ in ypools], 'Y')
    worlds, records, rejected = [], [], []
    complete = all(c for _, c in ypools)
    ys, shared_y = align([p for p, _ in ypools])
    try:
        for ym in shared_y:
            budget.check()
            zpools = [z_candidates(native, ctx, budget, tape, pool[ym]) for tape, pool in zip(tapes, ys)]
            saved.extend(archive([p for p, _ in zpools], 'Z', ym))
            complete &= all(c for _, c in zpools)
            zs, shared_z = align([p for p, _ in zpools])
            for zm in shared_z:
                budget.check()
                unit = [finish(ctx, yp[ym], zp[zm]) for yp, zp in zip(ys, zs)]
                if any(w.invalid for w, _ in unit):
                    rejected.append(dict(y=ym, z=zm, reasons=[w.invalid for w, _ in unit]))
                    continue
                for i, (world, record) in enumerate(unit):
                    worlds.append(world)
                    records.append(dict(record, sample=i, y_policy=ym, z_policy=zm, tapes=tapes[i]))
    except SearchBudgetExpired:
        complete = False
    return worlds, records, saved, complete, rejected


def aggregate(records: list, receiver: int) -> dict:
    modes = sorted({(r['y_policy'], r['z_policy']) for r in records})
    means = {m: float(np.mean([r['p1'] for r in records if (r['y_policy'], r['z_policy'])==m])) for m in modes}
    attacker_best, receiver_best = (max, min) if receiver==1 else (min, max)
    responses = {y: attacker_best([m for m in modes if m[0]==y], key=means.get) for y, _ in modes}
    chosen = receiver_best(responses.values(), key=means.get)
    # 実行予測は最初の合法発火方策。最善能力と別に保存する。
    first = min(modes)
    return dict(v_post=means[chosen], selected_policy=list(chosen),
        policy_means={f'{y}/{z}': p for (y, z), p in means.items()},
        predicted_execution=dict(policy=list(first), p1=means[first]),
        best_ability=dict(policy=list(chosen), p1=means[chosen]))


def solve(native: Any, ctx: Context, budget: Budget, evaluate: Callable, knowledge: dict) -> dict:
    if budget.milliseconds != 100:
        raise ValueError('v4.12の固定候補は100msだけ')
    if knowledge['evidence_max'] > ctx.input_sec:
        raise ValueError('未来のツモ観測')
    limit = (budget.milliseconds-EVALUATION_RESERVE_MS)/MS/LOW_SPEC_RATIO
    budget = replace(budget, deadline=min(budget.deadline, perf_counter()+limit))
    blank = dict(v_post=None, active=False, complete=False, knowledge=knowledge)
    try:
        validate(ctx)
        worlds, audits, saved, complete, rejected = collect(native, ctx, budget, knowledge)
    except BoardMismatch as exc:
        return dict(blank, reason=str(exc))
    common = dict(branches=audits, candidates=saved, cutoff=not complete, rejected=rejected,
                  completed_tapes=MIN_TAPES if audits else 0, attacker=ctx.attacker, receiver=ctx.receiver)
    if not audits:
        return dict(blank, **common, reason='time_budget_no_complete_policy')
    values = evaluate_branches(worlds, evaluate)
    for audit, world, probability in zip(audits, worlds, values, strict=True):
        audit.update(p1=probability, b3_input=dict(boards=world.audit()['boards'], pending=world.audit()['pending']),
                     b3_output=None if any(world.dead) else probability)
    return dict(aggregate(audits, ctx.receiver), **common, active=True, reason=None, complete=True,
        knowledge=knowledge, evidence_max=max(ctx.evidence_max, knowledge['evidence_max']),
        actual_reply=dict(fired=ctx.triggers[ctx.receiver] is not None, trigger_sec=ctx.triggers[ctx.receiver]))
