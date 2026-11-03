"""B3 確率を呼ばない理論 pending 差 minimax (dl1k 研究専用・既定 OFF)。

原器は v4.15 `prefire_second_v415` の Y 列挙・2 テープ完成単位と、v4.12 の Z 候補・会計・finish。
B3 推論・v_post・p_new は呼ばない (evaluate_pair / aggregate を使わない)。
枝の取り出し: 各枝終了時の「攻め側宛 pending − 受け側宛 pending」= 受け側視点の量 d を
2 テープ平均し、各 Y に対し d 最小の Z、そこから d 最大の Y。同値は Y/Z 方策文字列の辞書順。
未完走テープは 0 で補わない (2 テープとも完成した組だけを比べる)。
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any
from src.phase_j._live_ex_frozen.src.dl1k_work_budget import CountedNative, WorkBudget, configured_limit

MS = 1000.
TAPES = 2
BUDGET_MS = 100
# v4.15 は B3 評価の 20ms を予算から取り置いた。本抽出器は B3 を呼ばないので取り置かない。
B3_RESERVE_SEC = 0.


def pair_difference(rows: list[dict], receiver: int) -> float:
    """完成 2 テープの d (攻め側宛 pending − 受け側宛 pending) の平均。"""
    attacker = 1-receiver
    return sum(r['accounting']['pending'][attacker]-r['accounting']['pending'][receiver] for r in rows)/TAPES


def complete_pairs(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """同じ Y/Z 方策で 2 テープとも完成した組だけ。"""
    pairs: dict[tuple[str, str], list[dict]] = {}
    for row in records:
        pairs.setdefault((row['y_policy'], row['z_policy']), []).append(row)
    return {k: sorted(v, key=lambda r: r['sample']) for k, v in pairs.items()
            if len(v) == TAPES and {r['sample'] for r in v} == set(range(TAPES))}


def choose(records: list[dict], receiver: int) -> dict | None:
    """B3 から独立した理論量の minimax。勝率・仮想死亡の 0/1 代入は使わない。"""
    pairs = complete_pairs(records)
    if not pairs:
        return None
    means = {pair: pair_difference(rows, receiver) for pair, rows in pairs.items()}
    responses = {y: min((p for p in means if p[0] == y), key=lambda p: (means[p], p[1])) for y, _ in means}
    selected = min(responses.values(), key=lambda p: (-means[p], p[0], p[1]))
    return dict(selected_policy=list(selected), pending_difference=means[selected],
                branches=pairs[selected], pair_means={f'{y}/{z}': d for (y, z), d in sorted(means.items())})


def completed_tapes(records: list[dict]) -> int:
    """どの組でも最大で何本のテープが完成したか (0〜2)。"""
    counts: dict[tuple[str, str], set] = {}
    for row in records:
        counts.setdefault((row['y_policy'], row['z_policy']), set()).add(row['sample'])
    return max((len(v) for v in counts.values()), default=0)


def complete_pair(ctx: Any, ys: list, zs: list, ym: str, zm: str, budget: Any) -> list[dict]:
    """同じ Y/Z 方策の 2 テープを完走させる。片方でも不成立なら組を捨てる。"""
    from src.phase_j._live_ex_frozen.src.prefire_second_v412 import finish
    from src.phase_j._live_ex_frozen.src.prefire_second_v411 import surplus
    from src.phase_j._live_ex_frozen.src.prefire_world_v410 import Event
    budget.check()
    unit = [finish(ctx, y[ym], z[zm]) for y, z in zip(ys, zs, strict=True)]
    budget.check()
    if any(w.invalid for w, _ in unit):
        return []
    rows = []
    for sample, (world, row) in enumerate(unit):
        events = [Event(**e) for e in row['events']]
        row.update(sample=sample, y_policy=ym, z_policy=zm,
                   z_surplus=surplus(world, events, ctx.attacker, 'Z'))
        rows.append(row)
    return rows


def z_pools(native: Any, ctx: Any, tapes: list, budget: Any, pools: list, ym: str) -> list[tuple]:
    """時間版は残り時間を等分し、仕事量版は全テープ共通の上限でZを探す。"""
    from src.phase_j._live_ex_frozen.src.prefire_second_v412 import z_candidates
    result = []
    for index, (tape, pool) in enumerate(zip(tapes, pools, strict=True)):
        if isinstance(budget, WorkBudget):
            result.append(z_candidates(native, ctx, budget, tape, pool[ym]))
            continue
        now = perf_counter()
        remaining = budget.deadline-B3_RESERVE_SEC-now
        limited = replace(budget, deadline=now+max(0., remaining)/(len(tapes)-index))
        result.append(z_candidates(native, ctx, limited, tape, pool[ym]))
    return result


def process_y(native: Any, ctx: Any, tapes: list, budget: Any, ys: list, ym: str, out: dict) -> None:
    """1 つの Y 方策について Z を列挙し、完成した組を記録する。"""
    from src.phase_j._live_ex_frozen.src.prefire_second_v412 import align, archive
    found = z_pools(native, ctx, tapes, budget, ys, ym)
    out['cutoff'] |= not all(done for _, done in found)
    out['candidates'].extend(archive([p for p, _ in found], 'Z', ym))
    zs, modes = align([p for p, _ in found])
    for zm in modes:
        out['records'].extend(complete_pair(ctx, ys, zs, ym, zm, budget))


def future_evidence(ctx: Any, knowledge: dict, evidence: dict) -> float:
    """要求に使う証跡の最大時刻。入力時刻を超えたら因果違反。"""
    return max(ctx.evidence_max, knowledge['evidence_max'], evidence.get('evidence_sec', ctx.input_sec))


def search(native: Any, ctx: Any, tapes: list, budget: Any, evidence: dict, out: dict) -> list:
    """v4.15 と同じ順 (早く発火する Y から) で Y を列挙し、各 Y の Z を完成させる。"""
    from src.phase_j._live_ex_frozen.src.prefire_search_v410 import SearchBudgetExpired
    from src.phase_j._live_ex_frozen.src.prefire_search_v411 import BoardMismatch
    from src.phase_j._live_ex_frozen.src.prefire_second_v411 import validate
    from src.phase_j._live_ex_frozen.src.prefire_second_v412 import align
    from src.phase_j._live_ex_frozen.src.prefire_second_v415 import units
    completed: set[str] = set()
    pools: list = [{}, {}]
    try:
        validate(ctx)
        for pools in units(native, ctx, tapes, budget, evidence):
            ys, modes = align(pools)
            order = sorted((m for m in modes if m not in completed), key=lambda m: (
                min((p[m].fire for p in ys if p[m].fire is not None), default=float('inf')), m))
            for ym in order:
                budget.check()
                process_y(native, ctx, tapes, budget, ys, ym, out)
                completed.add(ym)
    except SearchBudgetExpired:
        out['cutoff'] = True
    except BoardMismatch as exc:
        out['reason'] = str(exc)
    return pools


def solve(native: Any, ctx: Any, knowledge: dict, evidence: dict, budget_ms: int = BUDGET_MS,
          work_budget: bool = False, work_limit: int | None = None) -> dict:
    """2テープの理論抽出。既定100ms、明示ON時は仕事量だけで打ち切る。"""
    from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Budget
    from src.phase_j._live_ex_frozen.src.prefire_second_v412 import archive
    from src.phase_j._live_ex_frozen.src.prefire_tsumo_v411 import sample_tape
    if budget_ms != BUDGET_MS:
        raise ValueError('固定仕様は探索予算 100ms だけ')
    start = perf_counter()
    work = WorkBudget(configured_limit(work_budget, work_limit))
    budget = work if work.limit is not None else Budget(budget_ms, deadline=start+budget_ms/MS)
    if future_evidence(ctx, knowledge, evidence) > ctx.input_sec:
        raise ValueError('未来の証跡')
    tapes = [[sample_tape(knowledge['sides'][s], ctx.colors, i, s) for s in (0, 1)] for i in range(TAPES)]
    out = dict(records=[], candidates=[], cutoff=False)
    pools = search(CountedNative(native, work), ctx, tapes, budget, evidence, out)
    out['candidates'] = archive(pools, 'Y')+out['candidates']
    selection = choose(out['records'], ctx.receiver)
    out.update(selection=selection, complete=not out['cutoff'] and 'reason' not in out,
               completed_tapes=completed_tapes(out['records']), elapsed_ms=(perf_counter()-start)*MS,
               receiver=ctx.receiver, attacker=ctx.attacker,
               budget_ms=budget_ms if work.limit is None else None,
               budget_kind='work' if work.limit is not None else 'time',
               work_limit=work.limit, work_units=work.used, work_counts=work.counts,
               evidence_max=future_evidence(ctx, knowledge, evidence),
               actual_reply=dict(fired=ctx.triggers[ctx.receiver] is not None,
                                 trigger_sec=ctx.triggers[ctx.receiver]))
    return out
