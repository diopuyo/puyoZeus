"""v4.12を基点にした既定OFFの独立試作。本番の採用値を参照変更しない。"""
from __future__ import annotations

from time import perf_counter
from typing import Any
from src.phase_j._live_ex_frozen.src.prefire_v412 import PostV412,LandingV49G,display as previous_display,BETA_CANDIDATES
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import context
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Budget
from src.phase_j._live_ex_frozen.src.prefire_second_v415 import solve
from src.phase_j._live_ex_frozen.src.prefire_zero_v415 import PlacementEvidence

PREFIRE_V415_ENABLED_DEFAULT=False
BUDGETS=(100,)
MS=1000.


class LandingV415(LandingV49G):
    def __init__(self, budget_ms: int = 100) -> None:
        super().__init__(diagnose_guarded=False)
        if budget_ms not in BUDGETS:
            raise ValueError('v4.15の候補外予算')
        self.budget=Budget(budget_ms)

    def continuous(self, engine: Any, value: dict, evaluate: Any, clock: Any,
                   variants: tuple = ('rep',)) -> dict:
        baseline=super().continuous(engine,value,evaluate,clock,variants)
        start,b3_before=perf_counter(),clock.seconds.get('b3',0.)
        ctx,reason=context(engine,value)
        if len(value.get('death_sides',[]))==1:
            ctx,reason=None,'certified_terminal'
        offset=(start-engine.request_start)*MS
        detail=(solve(engine.native,ctx,self.budget,evaluate.many,engine.knowledge.snapshot(ctx),
                      engine.placements.snapshot(ctx,engine.state)) if ctx else
                dict(active=False,v_post=None,reason=reason,checkpoints=[]))
        for checkpoint in detail['checkpoints']:
            checkpoint['request_ms']=offset+checkpoint['elapsed_ms']
        detail.update(budget_ms=self.budget.milliseconds,trial_ms=(perf_counter()-start)*MS)
        value['v415']=detail
        elapsed=perf_counter()-start-(clock.seconds.get('b3',0.)-b3_before)
        clock.seconds['enumerate']=clock.seconds.get('enumerate',0.)+max(0.,elapsed)
        return baseline


class PostV415(PostV412):
    def __init__(self,predictor: Any,table: Any,policy: dict,native: Any,
                 enabled: bool = PREFIRE_V415_ENABLED_DEFAULT,pre_model: dict | None = None,
                 budget_ms: int = 100,seed_lookup: Any = None) -> None:
        super().__init__(predictor,table,policy,native,enabled,pre_model,budget_ms,seed_lookup)
        self.landing=LandingV415(budget_ms)
        self.placements=PlacementEvidence()
        self.request_start=0.

    def _current(self,stamp: float,game: int,raw: list[dict]) -> None:
        super()._current(stamp,game,raw)
        if self.enabled:
            self.placements.observe(self.state,raw,stamp)

    def calculate_record(self,record: dict,raw: list[dict]) -> dict | None:
        self.request_start=perf_counter()
        value=super().calculate_record(record,raw)
        if value is not None:
            value['pc_ms']=(perf_counter()-self.request_start)*MS
        return value

    def synchronize(self,record: dict,raw: list[dict]) -> None:
        super().synchronize(record,raw)
        self.last_record,self.last_raw=record,raw


def display(b3: float,baseline: float,projection: dict | None,c: float,pre_delta: float,
            terminal: bool = False,menu: bool = False,enabled: bool = PREFIRE_V415_ENABLED_DEFAULT,
            beta: float = 0.) -> float:
    mapped=dict(projection,v412=projection.get('v415',{})) if projection else projection
    return previous_display(b3,baseline,mapped,c,pre_delta,terminal,menu,enabled,beta)
