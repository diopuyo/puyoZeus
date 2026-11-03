"""同じ2秒探索の完了時刻付きノードから1秒時点の暫定木を再現する。"""
from __future__ import annotations

import time
from typing import Any
from scripts.exchange_review_io import grid
from scripts.exchange_review_engine import (WIDTH, DEPTH, LEVELS, HAND_MS, CHAIN_MS,
                                          TARGETS, DEADLINES, THREADS, choose)
from scripts.exchange_review_v2_budget import BACK_RATIO, RETURN_RESERVE_MS
from scripts.exchange_review_v2 import recommendation

BUDGET_MS, CHECKPOINT_MS = 2000, 1000
CLOCK_SCALE = BACK_RATIO * BUDGET_MS / (BUDGET_MS - RETURN_RESERVE_MS)


def prefix(node: dict, cutoff: float) -> dict | None:
    """未来の子と、未来の子から親へ伝播した打切り印の両方を除く。"""
    if node['completed_ms'] > cutoff:
        return None
    children = [p for child in node['children'] if (p := prefix(child, cutoff)) is not None]
    return node | dict(children=children, truncated=node['progress_truncated'])


def checkpoint(tree: dict) -> dict:
    # 最終計算時間から境界を逆算せず、開始前に決めた50msの返却余裕を使う。
    cutoff = (CHECKPOINT_MS - RETURN_RESERVE_MS) / BACK_RATIO
    roots = [p for node in tree['own'] if (p := prefix(node, cutoff)) is not None]
    return dict(own=roots, opponent=[], budget_exhausted=True, cutoff_pc1_ms=cutoff)


def compute(event: dict, native: Any) -> dict:
    seed, fg, data = event['seed'], event['foreground'], event['exchange_input']
    assert seed.get('indices'), '種/手番未特定'
    attacker, receiver = data['attacker'], 1 - data['attacker']
    began = time.perf_counter()
    tree = native.ama_exchange_py(grid(data['receiver_board']), fg['opponent_board'],
        [tuple(pair) for pair in seed['tape']], (seed['indices'][receiver], seed['indices'][attacker]),
        fg['k'], WIDTH, DEPTH, data['colors'], LEVELS, HAND_MS, CHAIN_MS, TARGETS, DEADLINES,
        incoming=fg['incoming'], both=False, budget_ms=BUDGET_MS, parallel=True,
        num_threads=THREADS, advance_opponent=True,
        opponent_busy_ms=max(0, round(data.get('recursive_opponent_busy_sec', data['remaining_sec']) * 1000)),
        budget_clock_scale=CLOCK_SCALE, record_progress=True)
    elapsed = (time.perf_counter() - began) * 1000
    low_spec = elapsed * BACK_RATIO
    assert low_spec <= BUDGET_MS, f'2秒の公開期限超過: {low_spec}'
    # マーシャリングと全木返却の費用も各完成時刻に足す、保守的な上界。
    overhead = max(0., elapsed - tree['progress_elapsed_ms'])
    assert overhead * BACK_RATIO <= RETURN_RESERVE_MS, '返却余裕を超過'
    partial = checkpoint(tree) if low_spec > CHECKPOINT_MS else None
    final_choice = recommendation(event | dict(background=dict(tree=tree)), True)
    partial_choice = recommendation(event | dict(background=dict(tree=partial)), True) if partial else None
    if partial_choice:
        partial_choice['truncated'] = True
    return dict(tree=tree, partial_tree=partial, compute_ms=elapsed, low_spec_ms=low_spec,
        trace_overhead_ms=overhead, budget_ms=BUDGET_MS, budget_clock_scale=CLOCK_SCALE,
        final_choice=final_choice, partial_choice=partial_choice, **choose(tree))


def check_prefix(partial: dict, overhead: float) -> int:
    """暫定値が参照した全ノードが1秒以前に完成していたことを検査する。"""
    count, pending = 0, list(partial['own'])
    while pending:
        node = pending.pop()
        assert (node['completed_ms'] + overhead) * BACK_RATIO <= CHECKPOINT_MS
        assert node['truncated'] == node['progress_truncated']
        pending.extend(node['children'])
        count += 1
    return count
