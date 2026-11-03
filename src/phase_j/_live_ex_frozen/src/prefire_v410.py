"""既定OFFのv4.10。採用済みv4.9gの死亡証明と本番設定は変更しない。"""
from __future__ import annotations

from typing import Any
from time import perf_counter
import numpy as np
from src.phase_j._live_ex_frozen.src.prefire_v49g import PostV49G, LandingV49G
from src.phase_j._live_ex_frozen.src.prefire_v49d import display as baseline_display
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import context
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Budget
from src.phase_j._live_ex_frozen.src.prefire_second_v410 import solve, CONTINUOUS_EPS

PREFIRE_V410_ENABLED_DEFAULT = False
DEFAULT_BUDGET_MS = 100


class LandingV410(LandingV49G):
    """証明applyは継承し、連続値だけ別の時計と会計で求める。"""

    def __init__(self, budget_ms: int = DEFAULT_BUDGET_MS) -> None:
        super().__init__(diagnose_guarded=False)
        self.budget = Budget(budget_ms)

    def continuous(self, engine: Any, value: dict, evaluate: Any, clock: Any,
                   variants: tuple = ('rep',)) -> dict:
        ctx, reason = context(engine, value)
        if ctx is None:
            value['v410'] = dict(active=False, reason=reason, budget_ms=self.budget.milliseconds)
            return super().continuous(engine, value, evaluate, clock, variants)
        tick, b3_before = perf_counter(), clock.seconds.get('b3', 0.)
        detail = solve(engine.native, ctx, self.budget, evaluate.many)
        elapsed = perf_counter()-tick-(clock.seconds.get('b3', 0.)-b3_before)
        clock.seconds['enumerate'] = clock.seconds.get('enumerate', 0.)+max(0., elapsed)
        clock.calls['enumerate'] = clock.calls.get('enumerate', 0)+1
        value['v410'] = dict(detail, active=detail['v_post'] is not None,
                             actual_reply=dict(fired=ctx.triggers[ctx.receiver] is not None,
                                               trigger_sec=ctx.triggers[ctx.receiver]))
        if detail['v_post'] is None:
            return super().continuous(engine, value, evaluate, clock, variants)
        branches = detail['branches']
        self.last_enum = dict(sims=None, steps=None, work=None, truncated=False, b3_evals=len(branches))
        return dict(values={k: detail['v_post'] for k in variants}, receiver=ctx.receiver,
                    hands=max(b['Y']['moves'] for b in branches), immediate=False,
                    responses=len(branches), strata=[], complete=True, nodes=0)


class PostV410(PostV49G):
    """明示ON時だけ接続する。OFFのインスタンスはcalculate_recordで値を作らない。"""

    def __init__(self, predictor: Any, table: Any, policy: dict, native: Any,
                 enabled: bool = PREFIRE_V410_ENABLED_DEFAULT, pre_model: dict | None = None,
                 budget_ms: int = DEFAULT_BUDGET_MS) -> None:
        super().__init__(predictor, table, policy, native, enabled, pre_model,
                         receiver_chain_open=True, diagnose_guarded=False)
        self.landing = LandingV410(budget_ms)


def display(b3: float, baseline: float, projection: dict | None, c: float,
            pre_delta: float, terminal: bool = False, menu: bool = False,
            enabled: bool = PREFIRE_V410_ENABLED_DEFAULT) -> float:
    """試作の連続値だけで新たな0/100%を作らない。死亡証明・実終端は従来と同一。"""
    result = baseline_display(b3, baseline, projection, c, pre_delta, terminal, menu, enabled)
    if not enabled or terminal or menu or projection is None:
        return result
    if len(projection.get('death_sides', [])) == 1 or not projection.get('v410', {}).get('active'):
        return result
    return float(np.clip(result, CONTINUOUS_EPS, 1-CONTINUOUS_EPS))
