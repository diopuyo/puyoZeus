"""低スペックの仮想時計で1秒以内の再帰を読み直す、試作専用入口。"""
from __future__ import annotations

import time
from typing import Any
from scripts.exchange_review_io import grid
from scripts.exchange_review_engine import (WIDTH, DEPTH, LEVELS, HAND_MS, CHAIN_MS,
                                          TARGETS, DEADLINES, THREADS, choose)

BUDGET_MS, RETURN_RESERVE_MS = 1000, 50
BACK_RATIO = max(176.5 / 108.3, 3263.8 / 1981.2)
CLOCK_SCALE = BACK_RATIO * BUDGET_MS / (BUDGET_MS - RETURN_RESERVE_MS)


def compute(event: dict, native: Any) -> dict:
    seed, fg, data = event['seed'], event['foreground'], event['exchange_input']
    if not seed.get('indices'):
        return dict(status='種/手番未特定', compute_ms=0, candidate=None, truncated=True)
    attacker, receiver = data['attacker'], 1 - data['attacker']
    began = time.perf_counter()
    tree = native.ama_exchange_py(grid(data['receiver_board']), fg['opponent_board'],
        [tuple(pair) for pair in seed['tape']], (seed['indices'][receiver], seed['indices'][attacker]),
        fg['k'], WIDTH, DEPTH, data['colors'], LEVELS, HAND_MS, CHAIN_MS, TARGETS, DEADLINES,
        incoming=fg['incoming'], both=False, budget_ms=BUDGET_MS, parallel=True,
        num_threads=THREADS, advance_opponent=True,
        opponent_busy_ms=max(0, round(data.get('recursive_opponent_busy_sec', data['remaining_sec']) * 1000)),
        budget_clock_scale=CLOCK_SCALE)
    elapsed = (time.perf_counter() - began) * 1000
    # 完了後の結果を時刻だけ縮めて公開しない。超過した場合は描画を止める。
    if elapsed * BACK_RATIO > BUDGET_MS:
        raise RuntimeError(f'再帰の公開期限超過: {elapsed * BACK_RATIO:.3f}ms')
    return dict(status='計算済み', compute_ms=elapsed, **choose(tree), tree=tree,
                low_spec_ms=elapsed * BACK_RATIO, budget_ms=BUDGET_MS,
                budget_clock_scale=CLOCK_SCALE, return_reserve_ms=RETURN_RESERVE_MS)
