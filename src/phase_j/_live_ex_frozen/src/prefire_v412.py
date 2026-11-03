"""既定OFF。採用済みGを残し、新しい1往復予測を弱い補正として渡す。"""
from __future__ import annotations

from time import perf_counter
from typing import Any
from src.phase_j._live_ex_frozen.src.prefire_v49g import PostV49G, LandingV49G
from src.phase_j._live_ex_frozen.src.prefire_v49d import display as display_g
from src.phase_j._live_ex_frozen.src.prefire_v410 import display as display_new
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import context
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Budget
from src.phase_j._live_ex_frozen.src.prefire_tsumo_v411 import Knowledge, SeedLookup
from src.phase_j._live_ex_frozen.src.prefire_second_v412 import solve

PREFIRE_V412_ENABLED_DEFAULT = False
BETA_CANDIDATES = (0., .1, .2, .3, .4, .5)
BUDGETS = (100,)
MS = 1000.


class LandingV412(LandingV49G):
    """Gの証明・連続値を維持したまま新しい予測を別フィールドへ置く。"""

    def __init__(self, budget_ms: int = 100) -> None:
        super().__init__(diagnose_guarded=False)
        if budget_ms not in BUDGETS:
            raise ValueError('v4.12の候補外予算')
        self.budget = Budget(budget_ms)

    def continuous(self, engine: Any, value: dict, evaluate: Any, clock: Any,
                   variants: tuple = ('rep',)) -> dict:
        baseline = super().continuous(engine, value, evaluate, clock, variants)
        tick, b3_before = perf_counter(), clock.seconds.get('b3', 0.)
        ctx, reason = context(engine, value)
        if len(value.get('death_sides', [])) == 1:
            ctx, reason = None, 'certified_terminal'
        value['v412'] = (solve(engine.native, ctx, self.budget, evaluate.many, engine.knowledge.snapshot(ctx))
                         if ctx else dict(active=False, v_post=None, reason=reason))
        elapsed = perf_counter()-tick-(clock.seconds.get('b3', 0.)-b3_before)
        clock.seconds['enumerate'] = clock.seconds.get('enumerate', 0.)+max(0., elapsed)
        value['v412']['budget_ms'] = self.budget.milliseconds
        return baseline


class PostV412(PostV49G):
    def __init__(self, predictor: Any, table: Any, policy: dict, native: Any,
                 enabled: bool = PREFIRE_V412_ENABLED_DEFAULT, pre_model: dict | None = None,
                 budget_ms: int = 100, seed_lookup: SeedLookup | None = None) -> None:
        self.knowledge = Knowledge(seed_lookup)
        super().__init__(predictor, table, policy, native, enabled, pre_model,
                         receiver_chain_open=True, diagnose_guarded=False)
        self.landing = LandingV412(budget_ms)

    def _current(self, stamp: float, game: int, raw: list[dict]) -> None:
        super()._current(stamp, game, raw)
        if self.enabled:
            self.knowledge.observe(stamp, game, raw)


def display(b3: float, baseline: float, projection: dict | None, c: float,
            pre_delta: float, terminal: bool = False, menu: bool = False,
            enabled: bool = PREFIRE_V412_ENABLED_DEFAULT, beta: float = 0.) -> float:
    if not enabled:
        return baseline
    if not 0. <= beta <= .5:
        raise ValueError('弱い補正βは0〜0.5')
    g = display_g(b3, baseline, projection, c, pre_delta, terminal, menu, True)
    detail = projection.get('v412', {}) if projection else {}
    if g in (0., 1.) or not detail.get('active') or terminal or menu:
        return g
    trial = dict(projection, v_post=detail['v_post'], v410=dict(active=True))
    new = display_new(b3, baseline, trial, c, pre_delta, terminal, menu, True)
    return float(g+beta*(new-g))
