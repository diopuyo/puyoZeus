"""共通連鎖IDの最初の台帳基点を保持する既定OFFの観測状態。"""
from __future__ import annotations

import os
from typing import Any

ENV = 'PUYO_RETR_FIRE_ORIGIN_HOLD'
SIDES = (0, 1)


def first_origin(engine: Any, side: int, chain_id: float) -> dict | None:
    """式の初段より前の台帳だけを使う。未来の得点や後段の端数は使わない。"""
    rows = getattr(engine, 'ledger_history', ())
    baseline = next((row for row in reversed(rows) if row[0] < chain_id), None)
    if baseline is None:
        if engine.state.game_start_t > chain_id:
            return None
        baseline = (engine.state.game_start_t, 0, 0, 0, 0)
    return dict(chain_id=chain_id, started=chain_id, generated_at_fire=baseline[1+side],
                carry_at_fire=baseline[3+side], evidence_t=baseline[0])


def observe(engine: Any) -> None:
    """実台帳の加算・相殺は維持し、再通知時の射影の控除基点だけを揃える。"""
    if os.environ.get(ENV, '0') != '1':
        return
    game = engine.state.game_idx
    if getattr(engine, 'retr_origin_game', None) != game:
        engine.retr_origin_game, engine.retr_origins = game, [{}, {}]
    tracker = getattr(engine, 'retr_fire', None)
    if not hasattr(tracker, 'current'):
        return
    for side in SIDES:
        forecast = tracker.current(side, engine.state.t)
        if forecast is None or forecast.chain_id > engine.state.t:
            continue
        cache, identity = engine.retr_origins[side], float(forecast.chain_id)
        if identity not in cache:
            origin = first_origin(engine, side, identity)
            if origin is not None:
                cache[identity] = origin
        chain, origin = engine.chains[side], cache.get(identity)
        if chain is None or origin is None or chain.get('external'):
            continue
        prior = dict(started=chain['started'], generated_at_fire=chain['generated_at_fire'],
                     carry_at_fire=chain['carry_at_fire'])
        if chain.get('retr_origin', {}).get('chain_id') == identity:
            prior = chain['retr_origin']['before']
        chain.update({name: origin[name] for name in ('started', 'generated_at_fire', 'carry_at_fire')})
        chain['retr_origin'] = dict(origin, before=prior)
