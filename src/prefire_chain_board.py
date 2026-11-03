"""予測層だけの終了盤。v3.4候補生成とv3.10近傍修復を通知得点に適合する。"""
from __future__ import annotations
from src import prefire_work_budget as work_budget
from time import perf_counter
from typing import Any
from src.exchange_count_v34 import candidates, probabilities
from src.exchange_count_v310 import neighbours, MAX_EDITS
from src.exchange_count_prediction import simulate

REPAIR_SEC = .10
REPAIR_LIMIT = 800
BEAM = 8


def error(value: dict, observations: list[dict]) -> float:
    return sum(abs(sum(value['scores'][:o['count']])-o['score'])+
               max(0, o['count']-len(value['scores'])) for o in observations)


def repair(anchors: list[dict], observed: list[dict], colors: list[int],
           probs: list[dict], policy: dict, native: Any) -> list[dict]:
    """式の左右を持たない通知では累積段数・得点を制約にする。未来の最終段は使わない。"""
    began, tested, seen = perf_counter(), 0, set()
    frontier = sorted(anchors, key=lambda v: (error(v, observed), -v['log_weight']))[:BEAM]
    best = list(frontier)
    for _ in range(MAX_EDITS):
        values = []
        streams = [iter(neighbours(v, colors, probs, policy['prob_floor'])) for v in frontier]
        while streams and (work_budget.enabled() or perf_counter()-began < REPAIR_SEC) and tested < REPAIR_LIMIT:
            live = []
            for stream in streams:
                if work_budget.enabled() and tested >= work_budget.REPAIR_SIMULATIONS:
                    break
                item = next(stream, None)
                if item is None:
                    continue
                live.append(stream)
                board, weight, changes = item
                if tuple(board) not in seen:
                    seen.add(tuple(board))
                    tested += 1
                    values.append(dict(simulate(board, native), log_weight=weight, changes=changes))
            streams = live
        best = sorted(best+values, key=lambda v: (error(v, observed), -v['log_weight']))[:BEAM]
        if any(error(v, observed) == 0 for v in best):
            break
        frontier = best
    return best


def initial(board: list[int], history: list[dict], known: tuple, colors: list[int],
            stamp: float, policy: dict, native: Any) -> dict:
    sources = [dict(kind='cnn', t=stamp, board=board)]
    context = dict(history=history)
    pair = dict(identified=bool(known), pair=list(known[0]) if known else [], observed_sec=stamp)
    out = candidates(sources, [], colors, policy, stamp, native, context, pair)
    values = out['candidates']
    if not values:
        # 起点候補が発火を再現できなくても古い盤へ戻らず、物理終了盤と不一致を明示する。
        clean = [c if c in (0, 9, *colors) else 0 for c in board]
        values = [dict(simulate(clean, native), log_weight=0.)]
    return dict(anchors=values, pool=values, observed=[],
        probs=probabilities(sources, colors, context, policy), evidence=out['evidence_times'])


def advance(state: dict, count: int, score: int, stamp: float, colors: list[int],
            policy: dict, native: Any) -> dict:
    if any(t > stamp for t in state['evidence']):
        raise ValueError('終了盤に未来参照')
    state['observed'].append(dict(t=stamp, count=count, score=score))
    accepted = [v for v in state['pool'] if error(v, state['observed']) == 0]
    state['pool'] = accepted or repair(state['anchors'], state['observed'], colors, state['probs'], policy, native)
    best = min(state['pool'], key=lambda v: (error(v, state['observed']), -v['log_weight']))
    return dict(board=best['final'], remaining_stages=max(0, len(best['scores'])-count),
        conflict=error(best, state['observed']) != 0,
        evidence_times=state['evidence']+[o['t'] for o in state['observed']],
        origin=best['origin'], erased=sum(best['erased']))
