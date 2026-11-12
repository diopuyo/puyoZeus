"""死亡証明前の写しでretR・xcを計算し、元の状態で証明・探索する。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import copy, deepcopy
from typing import Any

import numpy as np

SNAPSHOT_FIELDS = ('state', 'chains', 'knowledge', 'retr_fire', 'retr_pre', 'retr_seed')
RETURN_FIELDS = ('retr_ama', 'retr_pre', 'retr_seed', 'retr_diag', 'xc_diag')
WORKERS = 1


def snapshot(engine: Any, value: dict, extra_memo: dict | None = None) -> tuple[Any, dict]:
    """可変の状態を分離する。種の表と検索索引だけは読み取り専用で共有する。"""
    memo = {id(engine.native): engine.native}
    if extra_memo is not None:
        memo.update(extra_memo)
    lookup = engine.knowledge.lookup
    if lookup is not None:
        for name in ('table', 'keys', 'order'):
            item = getattr(lookup, name)
            memo[id(item)] = item
    clone = copy(engine)
    for name in SNAPSHOT_FIELDS:
        if hasattr(engine, name):
            setattr(clone, name, deepcopy(getattr(engine, name), memo))
    # Amaはこの作業スレッドだけが触る。元の探索は別のnative APIを使う。
    return clone, deepcopy(value, memo)


def receiver(clone: Any, value: dict) -> tuple[dict, dict]:
    from . import live_ex_compute as calc
    result, times = {}, {}
    clone.retr_diag = [None, None]
    calc.timed(times, 'retr', calc.add_retr, clone, value, result)
    calc.retr_details(clone, value, result)
    if calc.xc_enabled():
        calc.timed(times, 'xc', calc.add_xc, clone, result)
    return result, times


def parallel(engine: Any, value: dict, result: dict, times: dict) -> bool:
    """写せない場合は元の直列処理へ戻す。例外時も受け側の完了を待つ。"""
    from . import live_ex_compute as calc
    try:
        clone, frozen = calc.timed(times, 'snapshot', snapshot, engine, value)
    except Exception as exc:
        result['ex_speed'] = dict(parallel=False, fallback=repr(exc))
        return False
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future = pool.submit(copy_context().run, receiver, clone, frozen)
        try:
            result['proof'] = calc.timed(times, 'proof', calc.proof_of, engine, value)
            result['output'], result['reason'], result['observed_scores'] = calc.timed(
                times, 'search', calc.search_of, engine, value)
        finally:
            received, receiver_times = future.result()
            result.update(received)
            times.update(receiver_times)
            for name in RETURN_FIELDS:
                if hasattr(clone, name):
                    setattr(engine, name, getattr(clone, name))
    result['ex_speed'] = dict(parallel=True)
    return True


def matching_positions(tape: np.ndarray, codes: np.ndarray) -> list[int]:
    """元の全位置比較と同じ候補を昇順で返す。探索条件は変更しない。"""
    size = len(codes)
    count = len(tape) - size + 1
    if count <= 0:
        return []
    matches = np.ones(count, dtype=bool)
    for offset, code in enumerate(codes):
        matches &= tape[offset:offset + count] == code
    return np.flatnonzero(matches).tolist()
