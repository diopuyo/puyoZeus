"""leanの履歴と生NEXTを分け、一意な組位置だけを同期する既定OFFの経路。"""
from __future__ import annotations

import os
from typing import Any

ENV = 'PUYO_RETR_TSUMO_SYNC'
FIELDS = ('retr_next_pair', 'retr_dnext_pair')
FOLLOWUP_FLAGS = ('PUYO_RETR_SMALL_CHAIN_CONTINUE', 'PUYO_RETR_FIRE_ORIGIN_HOLD', ENV)
RELATED_FLAGS = FOLLOWUP_FLAGS + ('PUYO_RETR_RECEIVER_FORECAST', 'PUYO_RETR_OWN_CHAIN', 'PUYO_RETR_LONG_ATTACK_H')


def enabled() -> bool:
    from .dl1k_retr_history import enabled as history_enabled
    return os.environ.get(ENV, '0') == '1' or history_enabled()


def pair(value: Any) -> tuple | None:
    raw = value.get('tuple') if isinstance(value, dict) else value
    if raw is None or len(raw) != 2 or any(c not in range(1, 6) for c in raw):
        return None
    return tuple(sorted(map(int, raw)))


def observe(engine: Any, raw: list[dict]) -> None:
    """今の観測が欠測ならNoneを保存し、古い生NEXTを流用しない。"""
    from .dl1k_retr_fall import observe as observe_fall
    observe_fall(engine)
    from .dl1k_retr_history import observe as observe_history
    observe_history(engine, raw)
    if enabled():
        engine.retr_observed_next = [dict(t=engine.state.t, game=engine.state.game_idx,
            pairs=[pair(row.get(key)) for key in FIELDS]) for row in raw]


def synchronize(engine: Any, side: int, busy: bool, seed: int, first: int, depth: int) -> int:
    """現在のNEXT2組が種表の絶対位置全体で一意に一致した時だけ位置を替える。"""
    if not enabled():
        return first
    if not hasattr(engine, 'retr_sync_diag'):
        engine.retr_sync_diag = {}
    detail = dict(seed=seed, start_before=first, start_after=first, changed=False, reason='observed_missing')
    engine.retr_sync_diag[side] = detail
    observed = getattr(engine, 'retr_observed_next', ())
    current = observed[side] if side < len(observed) else None
    if current is None or current['t'] != engine.state.t or current['game'] != engine.state.game_idx:
        return first
    if any(p is None for p in current['pairs']):
        return first
    colors = engine.knowledge.lookup.table.abs_colors[seed]
    candidates = [i for i in range(len(colors)-1) if
                  (pair(colors[i]), pair(colors[i+1])) == tuple(current['pairs'])]
    detail.update(candidates=candidates, observed=current['pairs'])
    from .dl1k_retr_history import enabled as history_enabled, candidates as from_history
    if history_enabled():
        candidates, size = from_history(engine, side, seed, current['pairs'], candidates)
        detail.update(history_candidates=candidates, history_size=size)
    if len(candidates) != 1:
        detail['reason'] = 'ambiguous' if candidates else 'no_match'
        return first
    start = candidates[0] if busy else candidates[0]-1
    if start < 0 or start+depth > len(colors):
        detail['reason'] = 'short_tape'
        return first
    detail.update(start_after=start, changed=start != first, reason='unique')
    return start


def add_input_key(engine: Any, state: dict) -> None:
    """新旗OFFでは旧キーを保ち、ON時は異なる方針・生NEXTの結果を混ぜない。"""
    if any(os.environ.get(flag, '0') == '1' for flag in FOLLOWUP_FLAGS):
        state['retr_followup_flags'] = {flag: os.environ.get(flag, '0') == '1' for flag in RELATED_FLAGS}
        if enabled():
            state['retr_observed_next'] = getattr(engine, 'retr_observed_next', None)
    from .dl1k_retr_fall import enabled as fall_enabled
    if fall_enabled():
        state['retr_fall_board'] = getattr(engine, 'retr_fall_rows', None)
        state['retr_fall_flags'] = {flag: os.environ.get(flag, '0') == '1' for flag in RELATED_FLAGS}
    from .dl1k_retr_history import enabled as history_enabled, input_key
    if history_enabled():
        state['retr_sync_history'] = input_key(engine)
        state['retr_observed_next'] = getattr(engine, 'retr_observed_next', None)
        state['retr_history_flags'] = {flag: os.environ.get(flag, '0') == '1' for flag in RELATED_FLAGS}
