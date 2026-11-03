"""同じテープでY/Zの即発火と伸長を比較し、1往復の盤面と会計を保存する。"""
from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
from time import perf_counter
from typing import Any, Callable
import numpy as np
from src.phase_j._live_ex_frozen.src.indicators_v2 import SEC_PER_HAND
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import Context
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Budget, SearchBudgetExpired
from src.phase_j._live_ex_frozen.src.prefire_search_v411 import plan, RecordedPlan, MODES, guard, ColorMismatch, BoardMismatch
from src.phase_j._live_ex_frozen.src.prefire_second_v410 import evaluate_branches, EVALUATION_RESERVE_MS, MIN_TAPES, MS
from src.phase_j._live_ex_frozen.src.prefire_world_v410 import World, GENERATE
from src.phase_j._live_ex_frozen.src.prefire_tsumo_v411 import sample_tape
from src.phase_j._live_ex_frozen.src.prefire_worker import LOW_SPEC_RATIO


def validate(ctx: Context) -> None:
    for raw in ctx.world.boards:
        guard(raw, ctx.colors)
    for event in ctx.world.events:
        if event.board:
            guard(event.board, ctx.colors)


def branch(native: Any, ctx: Context, budget: Budget, tapes: list, ym: str, zm: str,
           cache: dict | None = None) -> tuple:
    y = RecordedPlan(ctx.receiver, ym, ctx.input_sec, ctx.ends[ctx.attacker]+SEC_PER_HAND)
    if cache is not None and ym in cache:
        y = cache[ym]
    elif ctx.triggers[ctx.receiver] is None:
        y = plan(native, ctx, ctx.world, ctx.receiver, y.start, y.deadline, tapes[ctx.receiver], ym, budget, 'Y')
    else:
        y.fire, y.end, y.reason = ctx.triggers[ctx.receiver], ctx.ends[ctx.receiver], 'already_fired_observed'
    if cache is not None:
        cache[ym] = y
    world = ctx.world.copy()
    world.add(y.events)
    start = max(ctx.input_sec, ctx.ends[ctx.attacker])
    deadline = y.end if y.end is not None else start
    z = RecordedPlan(ctx.attacker, zm, start, deadline, reason='no_second_window')
    if deadline > start:
        z = plan(native, ctx, world, ctx.attacker, start, deadline, tapes[ctx.attacker], zm, budget, 'Z')
        world.add(z.events)
    events = list(world.events)
    world.finish()
    for raw in world.boards:
        guard(raw, ctx.colors)
    record = dict(Y=y.audit(), Z=z.audit(), accounting=world.audit(), events=[asdict(e) for e in events],
        factual_reply_used=ctx.triggers[ctx.receiver] is not None, observed_reply_fire_sec=ctx.triggers[ctx.receiver],
        ability_reply_fire_sec=y.fire, y_surplus=surplus(world, events, ctx.receiver,
            'Y' if ctx.triggers[ctx.receiver] is None else f'C{ctx.receiver}'))
    return world, record


def surplus(world: World, events: list, side: int, source: str) -> list[dict]:
    stamps = {e.t for e in events if e.kind == GENERATE and e.side == side and e.source == source}
    return [dict(t=row['t'], generated=row['amounts'][side], sent_surplus=row['sent'][side])
            for row in world.ledger.trace if row['kind'] == 'generate' and row['t'] in stamps]


def balanced(native: Any, ctx: Context, budget: Budget, knowledge: dict) -> tuple:
    modes = list(product(MODES if ctx.triggers[ctx.receiver] is None else ('observed',), MODES))
    worlds, records, completed, cutoff = [], [], 0, False
    for sample in range(budget.samples):
        tapes = [sample_tape(knowledge['sides'][i], ctx.colors, sample, i) for i in (0, 1)]
        unit, cache = [], {}
        try:
            for ym, zm in modes:
                budget.check()
                world, record = branch(native, ctx, budget, tapes, ym, zm, cache)
                unit.append((world, dict(record, sample=sample, y_policy=ym, z_policy=zm, tapes=tapes)))
            budget.check()
        except SearchBudgetExpired:
            cutoff = True
            break
        worlds.extend(w for w, _ in unit)
        records.extend(r for _, r in unit)
        completed += 1
    return worlds, records, completed, cutoff, modes


def aggregate(records: list[dict], modes: list, receiver: int) -> dict:
    means = {(y, z): float(np.mean([r['p1'] for r in records if (r['y_policy'], r['z_policy']) == (y, z)]))
             for y, z in modes}
    attacker_best = max if receiver == 1 else min
    responses = {y: attacker_best((m for m in modes if m[0] == y), key=means.get) for y, _ in modes}
    receiver_best = max if receiver == 0 else min
    chosen = receiver_best(responses.values(), key=means.get)
    immediate = ('immediate' if ('immediate', 'immediate') in means else 'observed', 'immediate')
    return dict(v_post=means[chosen], selected_policy=list(chosen),
        policy_means={f'{y}/{z}': p for (y, z), p in means.items()},
        predicted_execution=dict(policy=list(immediate), p1=means[immediate]),
        best_ability=dict(policy=list(chosen), p1=means[chosen]))


def solve(native: Any, ctx: Context, budget: Budget, evaluate: Callable,
          knowledge: dict) -> dict:
    if budget.milliseconds not in (100, 250):
        raise ValueError('v4.11の固定候補は100/250ms')
    if knowledge['evidence_max'] > ctx.input_sec:
        raise ValueError('未来のツモ観測')
    limit = (budget.milliseconds-EVALUATION_RESERVE_MS)/MS/LOW_SPEC_RATIO
    budget = replace(budget, deadline=min(budget.deadline, perf_counter()+limit))
    blank = dict(v_post=None, active=False, complete=False, knowledge=knowledge)
    try:
        validate(ctx)
        worlds, audits, n, cutoff, modes = balanced(native, ctx, budget, knowledge)
    except BoardMismatch as exc:
        return dict(blank, reason=str(exc))
    common = dict(branches=audits, completed_tapes=n, cutoff=cutoff, attacker=ctx.attacker, receiver=ctx.receiver)
    if n < MIN_TAPES:
        return dict(blank, **common, reason='time_budget_insufficient_tapes')
    invalid = next((w.invalid for w in worlds if w.invalid), None)
    if invalid:
        return dict(blank, **common, reason=invalid)
    values = evaluate_branches(worlds, evaluate)
    for audit, world, probability in zip(audits, worlds, values, strict=True):
        audit.update(p1=probability, b3_input=dict(boards=world.audit()['boards'],
            pending=world.audit()['pending']), b3_output=None if any(world.dead) else probability)
    return dict(aggregate(audits, modes, ctx.receiver), **common, active=True, reason=None, complete=True,
                knowledge=knowledge, evidence_max=max(ctx.evidence_max, knowledge['evidence_max']),
                actual_reply=dict(fired=ctx.triggers[ctx.receiver] is not None, trigger_sec=ctx.triggers[ctx.receiver]))
