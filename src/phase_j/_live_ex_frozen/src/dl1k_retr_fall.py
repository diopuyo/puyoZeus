"""落下開始時に確定予告から減った量だけを、古い確定盤へ足し戻す。"""
from __future__ import annotations

import math
import os
from typing import Any

ENV = 'PUYO_RETR_FALL_BOARD'
SIDES = ('p1', 'p2')


def enabled() -> bool:
    return os.environ.get(ENV, '0') == '1'


def observe(engine: Any) -> None:
    """相殺による減少を除外し、後の確定盤が届いたら控えを捨てる。"""
    if not enabled():
        return
    game, now = engine.state.game_idx, engine.state.t
    if getattr(engine, 'retr_fall_game', None) != game:
        engine.retr_fall_game, engine.retr_fall_rows = game, [None, None]
        engine.retr_fall_previous = [None, None]
    snap = engine.state.sl_snap or {}
    for side, name in enumerate(SIDES):
        state = engine.state.sides[side]
        current = dict(t=now, pending=snap.get(f'pending_confirmed_{name}'),
                       offset=snap.get(f'offset_{name}'), state=state.state)
        previous, row = engine.retr_fall_previous[side], engine.retr_fall_rows[side]
        if row is not None and state.board_t > row['board_t']:
            row = None
        if state.state == 'OJAMA_FALL':
            if previous is None or previous['state'] != 'OJAMA_FALL':
                row = dict(started=now, board_t=state.board_t, amount=0, evidence=[])
            if row is not None and previous is not None:
                values = (previous['pending'], current['pending'], previous['offset'], current['offset'])
                if all(v is not None for v in values):
                    decrease = previous['pending']-current['pending']
                    if decrease > 0 and previous['offset'] == current['offset']:
                        row['amount'] += decrease
                        row['evidence'].append(dict(t=now, decrease=decrease))
        engine.retr_fall_rows[side] = row
        engine.retr_fall_previous[side] = current


def amount(engine: Any, side: int) -> int | None:
    """同じ試合・古い盤・落下中という条件を全部満たす時だけ返す。"""
    if not enabled() or getattr(engine, 'retr_fall_game', None) != engine.state.game_idx:
        return None
    row = engine.retr_fall_rows[side]
    state = engine.state.sides[side]
    if row is None or state.state != 'OJAMA_FALL' or state.board_t != row['board_t'] or state.board_t >= row['started']:
        return None
    return int(row['amount']) if row['amount'] > 0 else None


def apply(run: Any, side: int, board: Any, incoming: int, hands: int) -> bool:
    """小連鎖のない落下盤も、復元できなければ下限と欠測を返す。"""
    from . import dl1k_retr as retr
    from .dl1k_retr_continue import apply as small_chain
    if not enabled() or run.resolving[side] or run.engine.state.sides[side].state != 'OJAMA_FALL':
        return False
    restored = amount(run.engine, side)
    run.diag[side]['fall_board'] = dict(restored=restored, reason='pending_decrease' if restored else 'no_decrease')
    if small_chain(run, side, board, incoming, hands):
        return True
    if restored is None:
        run.why[side] = 'landing_board_unresolved'
        run.store(side, board, incoming, 0, 0, dict(kind='none'), None)
        for stem in ('retR_lethal', 'retR_r_room', 'retR_rvis_room'):
            key = f'{stem}_{retr.NAMES[side]}'
            if key in run.extra:
                run.extra[key] = math.nan
        run.engine.retr_pre[side] = (run.game, tuple(run.extra[f'{key}_{retr.NAMES[side]}'] for key in retr.STEMS))
        if run.vis_on:
            run.engine.retr_visible_pre[side] = tuple(run.extra[f'{key}_{retr.NAMES[side]}'] for key in retr.VISIBLE_STEMS)
        return True
    pending = retr.confirmed_pending(run.engine, side)+restored
    capacity, ret, profile = run.capacity(side, board, hands, pending)
    if capacity is not None:
        run.store(side, board, incoming, hands, capacity, ret, profile)
    return True
