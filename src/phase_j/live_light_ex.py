"""凍結EXを変えず、隠し段の候補ごとに独立した状態で計算する。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np

from src.light_consumption import enabled
from src.light_hidden_posterior import MAX_CANDIDATES, scenarios
from ._live_ex_frozen.src.dl1k_extractor import extra_from
from .live_ex_compute import compute


def forecast_memo(engine: Any) -> dict:
    """観測を進めない計算用に、現在の予測だけを独立した写しにする。"""
    memo = {}
    # 計算中は観測を進めない。数千の物理候補ではなく、この時点の公開予測を固定する。
    tracker = getattr(engine, 'retr_fire', None)
    if tracker is not None:
        for side, owner in enumerate(tracker.sides):
            memo[id(owner)] = FixedForecast(deepcopy(tracker.current(side, engine.state.t)))
    return memo


def clone_engine(engine: Any) -> Any:
    """native関数と読取専用種表の配列以外は共有しない。"""
    pool = ThreadPoolExecutor(max_workers=1)
    memo = {id(engine.native): engine.native, id(engine.pool): pool, **forecast_memo(engine)}
    lookup = engine.knowledge.lookup
    if lookup is not None:
        for name in ('table', 'keys', 'order'):
            item = getattr(lookup, name)
            memo[id(item)] = item
    ama = getattr(engine, 'retr_ama', None)
    if ama is not None and ama.proc is not None:
        pool.shutdown()
        raise RuntimeError('候補分離には配布構成のnative AMAが必要です')
    try:
        return deepcopy(engine, memo)
    except Exception:
        pool.shutdown()
        raise


class FixedForecast:
    def __init__(self, forecast: Any) -> None:
        self.forecast = forecast

    def current(self, now: float | None = None) -> Any:
        return self.forecast


class ScenarioForecast:
    """複製した追跡器にだけ候補の段列と終端を与える。"""

    def __init__(self, tracker: Any, sides: tuple, now: float) -> None:
        self.tracker, self.sides = tracker, tracker.sides
        self.selected = {}
        for i, row in enumerate(sides):
            original = tracker.current(i, now)
            if row is not None and original is not None:
                self.selected[i] = replace(original, steps=tuple(row['steps']),
                    final_board=np.asarray(row['final'], dtype=np.int8), lower_bound=False)

    def current(self, side: int, now: float) -> Any:
        return self.selected.get(side) or self.tracker.current(side, now)


def install(engine: Any, scenario: dict) -> None:
    engine.retr_fire = ScenarioForecast(engine.retr_fire, scenario['sides'], engine.state.t)
    for i, row in enumerate(scenario['sides']):
        if row is None:
            continue
        chain = engine.chains[i]
        if chain is None:
            engine._accept_forecasts([dict(side=i, trigger=row['chain_id'],
                input_sec=engine.state.t, evidence_times=[engine.state.t],
                game=engine.state.game_idx, board=np.asarray(row['final']).ravel().tolist(),
                scores=list(row['steps']), sent=0)], engine.state.t, engine.state.game_idx)
            chain = engine.chains[i]
        chain['best']['scores'] = list(row['steps'])
        chain['result']['board'] = np.asarray(row['final']).ravel().tolist()
        from ._live_ex_frozen.src.prefire_physics_v49c import reconciled_send
        if chain.get('external') and engine.state.sl.core.origin_sec is not None:
            chain['external_sent'] = reconciled_send(chain, engine.state.sl.core, i)[0]


def calculate_candidates(engine: Any, limit: int = MAX_CANDIDATES) -> list[dict]:
    if not enabled() or not hasattr(engine, 'retr_fire'):
        return []
    selected = scenarios(engine.retr_fire.sides, limit)
    results = []
    ama = None
    for scenario in selected:
        clone = clone_engine(engine)
        try:
            if ama is not None:
                # 同じ盤・同じツモのnative結果だけを、この更新内で再利用する。
                # 本線のキャッシュ、候補ごとの保持値には戻さない。
                clone.retr_ama = ama
                clone.retr_pre = getattr(clone, 'retr_pre', {})
                clone.retr_seed = getattr(clone, 'retr_seed', {})
            install(clone, scenario)
            result = compute(clone)
            ama = getattr(clone, 'retr_ama', None)
            extra = extra_from(result)
            extra.update(result.get('xc', {}))
            results.append(dict(scenario, extra=extra, result=result))
        finally:
            clone.close()
    return results


def calculate_xc(engine: Any, limit: int = MAX_CANDIDATES) -> list[dict]:
    if not enabled() or not hasattr(engine, 'retr_fire'):
        return []
    from .live_ex_compute import add_xc
    results = []
    for scenario in scenarios(engine.retr_fire.sides, limit):
        clone = clone_engine(engine)
        try:
            install(clone, scenario)
            clone.projection()
            result: dict = {}
            add_xc(clone, result)
            results.append(dict(scenario, xc=result.get('xc', {})))
        finally:
            clone.close()
    return results
