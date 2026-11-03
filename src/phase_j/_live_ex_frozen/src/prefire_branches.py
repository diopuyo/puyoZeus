"""v4.4専用の分岐曲線。既定OFF、物理計算はRust/bitboardのみ。"""
from __future__ import annotations
from src.phase_j._live_ex_frozen.src import prefire_work_budget as work_budget
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from time import perf_counter
from typing import Any
import numpy as np
from src.phase_j._live_ex_frozen.src.prefire_growth import SideInput, validate_side, CURVE_HORIZON, ROLLOUT_COUNT, VISIBLE_PAIRS
from src.phase_j._live_ex_frozen.src.prefire_worker import LOW_SPEC_RATIO, MAX_LOW_SPEC_SEC
from src.phase_j._live_ex_frozen.src.production_config import PREFIRE_V4_ENABLED_DEFAULT

BUDGET_SEC = .025
MS = 1000.


def rollout(task: tuple, native: Any) -> list:
    request, pairs, deadline = task
    return native.ama_branches_py(list(request.board), pairs, work_budget.CURVE_HANDS if work_budget.enabled() else CURVE_HORIZON,
        list(request.colors), budget_ms=None if work_budget.enabled() else max(0., deadline-perf_counter())*MS,
        visible_pairs=len(request.known))


def calculate(request: SideInput, native: Any, pool: ThreadPoolExecutor,
              enabled: bool = PREFIRE_V4_ENABLED_DEFAULT) -> dict | None:
    if not enabled:
        return None
    validate_side(request)
    began = perf_counter()
    rng = np.random.default_rng(request.seed)
    tasks = []
    for _ in range(ROLLOUT_COUNT):
        unknown = rng.choice(request.colors, size=(CURVE_HORIZON+VISIBLE_PAIRS-len(request.known), 2))
        tasks.append((request, [tuple(p) for p in request.known]+[tuple(map(int, p)) for p in unknown], began+BUDGET_SEC))
    samples = list(pool.map(lambda task: rollout(task, native), tasks))
    length = min(map(len, samples))
    values = np.asarray([s[:length] for s in samples])
    elapsed = (perf_counter()-began)*MS*LOW_SPEC_RATIO
    branches = {name: dict(scores=values[:, :, i].mean(axis=0).tolist() if length else [],
        chains=values[:, :, i+1].mean(axis=0).tolist() if length else [])
        for name, i in (('provocation', 1), ('main', 3))}
    scores = np.maximum(branches['provocation']['scores'], branches['main']['scores']).tolist()
    chains = np.maximum(branches['provocation']['chains'], branches['main']['chains']).tolist()
    return dict(request=asdict(request), game=request.game, side=request.side,
        input_sec=request.input_sec, ready_sec=request.input_sec+elapsed/MS,
        low_spec_ms=elapsed, over_budget=False if work_budget.enabled() else elapsed > MAX_LOW_SPEC_SEC*MS,
        scores=scores, chains=chains, branches=branches, computed_k=length-1)
