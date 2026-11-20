"""小連鎖の消去後から追加の返しを探索する既定OFFの経路。"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

ENV = 'PUYO_RETR_SMALL_CHAIN_CONTINUE'
EPSILON = 1e-9


def enabled() -> bool:
    return os.environ.get(ENV, '0') == '1'


def search(run: Any, side: int, board: np.ndarray, hands: int, pending: int) -> tuple:
    """発火に使った操作中の組を再使用せず、NEXTから探索する。"""
    from . import dl1k_retr as retr
    if hands <= 0:
        return 0, 0, None, 'no_time'
    pairs, why = retr.seed_pairs(run.engine, side, True, getattr(run.engine, 'retr_seed_matcher', None))
    if pairs is None or len(pairs) < retr.DEPTH:
        return 0, 0, None, why if pairs is None else 'short_seed_pairs'
    encoded = retr.encode(board, pairs)
    if encoded is None:
        return 0, 0, None, 'five_colors_or_unknown'
    profile = run.engine.retr_ama.profile(*encoded, pending=pending, rate=run.rate)
    return (int(profile[min(hands, retr.DEPTH)-1])//run.rate,
            int(profile[min(hands, retr.BUSY_VISIBLE_HANDS)-1])//run.rate, profile, 'ama')


def timing(run: Any, side: int, count: int, score: int, hands: int) -> tuple[float, int]:
    """既観測の同じ小連鎖なら経過時間を使い、次の設置1手を確保する。"""
    from . import dl1k_retr as retr
    from .prefire_context_v410 import duration
    end = run.now + duration(count)
    forecast = run.fix.current_forecast(run.engine, side)
    if forecast is not None and forecast.has_prediction and len(forecast.steps) == count and sum(forecast.steps) == score:
        end = min(end, retr.forecast_end(forecast, run.now))
    deadline = run.ends[1-side]
    deadline = run.now + hands*retr.SEC_PER_HAND if deadline is None else deadline + retr.SEC_PER_HAND
    budget = deadline-end-retr.SEC_PER_HAND
    remaining = min(retr.MAX_H, max(0, math.floor((budget+EPSILON)/retr.SEC_PER_HAND)))
    from . import dl1k_retr_deadline as refire
    if refire.enabled() and run.ends[1-side] is not None:
        remaining = refire.hands(end, run.ends[1-side], retr.SEC_PER_HAND, retr.MAX_H)
    return end, remaining


def store(run: Any, side: int, incoming: int, small: int, final: np.ndarray,
          hands: int, future: int, visible: int, profile: Any, detail: dict) -> None:
    """CとCvisに小連鎖を一度だけ加え、致死と空きは消去後の盤で測る。"""
    from . import dl1k_retr as retr
    remainder = run.fix.write_own_values(run.extra, side, incoming, small, future, visible, hands, final)
    run.diag[side].update(h=hands, small_chain=detail)
    run.why[side] = 'small_chain_continue'
    values = tuple(run.extra[f'{key}_{retr.NAMES[side]}'] for key in retr.STEMS)
    run.engine.retr_pre[side] = (run.game, values)
    if run.vis_on:
        run.engine.retr_visible_pre[side] = tuple(run.extra[f'{key}_{retr.NAMES[side]}'] for key in retr.VISIBLE_STEMS)
    if run.sec_on:
        after = dict(kind='ama', cummax=profile, h=hands) if future and profile else dict(kind='none')
        run.second(side, dict(kind='after_small', start=detail['end']+retr.SEC_PER_HAND,
                              end=detail['end'], after=after), remainder)


def apply(run: Any, side: int, board: np.ndarray, incoming: int, hands: int) -> bool:
    """未復元の落下中盤は追加探索せず、既存LANDINGは確定待ち予告にだけ使う。"""
    from . import dl1k_retr as retr
    from . import dl1k_retr_fall as fall
    fall_on = fall.enabled() and run.engine.state.sides[side].state == 'OJAMA_FALL'
    if not (enabled() or fall_on) or run.resolving[side] or hands <= 0:
        return False
    cells = [c if 0 <= c <= 5 or c == 9 else 9 for c in map(int, board.ravel())]
    trace = run.engine.native.simulate_chain_trace_py(cells, True)
    if not trace:
        return False
    score = sum(int(step[4]) for step in trace)
    small = score//run.rate
    final = np.asarray(trace[-1][1], dtype=np.int64).reshape(retr.ROWS, retr.COLS)
    end, remaining = timing(run, side, len(trace), score, hands)
    pending = max(0, retr.confirmed_pending(run.engine, side)-small)
    state = run.engine.state.sides[side]
    falling = state.state == 'OJAMA_FALL'
    restored = fall.amount(run.engine, side) if falling else None
    unresolved = falling and restored is None
    if restored is not None:
        pending = max(0, retr.confirmed_pending(run.engine, side)+restored-small)
    if unresolved:
        future, visible, profile, why = 0, 0, None, 'landing_board_unresolved'
        remaining = 0
    else:
        future, visible, profile, why = search(run, side, final, remaining, pending)
    detail = dict(steps=len(trace), score=score, small=small, h_before=hands, h_after=remaining,
                  end=end, wait_sec=end-run.now+retr.SEC_PER_HAND, pending=pending,
                  board_t=state.board_t, after_why=why, lower_bound=why != 'ama')
    store(run, side, incoming, small, final, remaining, future, visible, profile, detail)
    if unresolved:
        # 落下位置が不明なので、消去後盤だけで死亡・roomを断定しない。
        for stem in ('retR_lethal', 'retR_r_room', 'retR_rvis_room'):
            if f'{stem}_{retr.NAMES[side]}' in run.extra:
                run.extra[f'{stem}_{retr.NAMES[side]}'] = math.nan
        run.engine.retr_pre[side] = (run.game, tuple(run.extra[f'{key}_{retr.NAMES[side]}'] for key in retr.STEMS))
        if run.vis_on:
            run.engine.retr_visible_pre[side] = tuple(run.extra[f'{key}_{retr.NAMES[side]}'] for key in retr.VISIBLE_STEMS)
    return True
