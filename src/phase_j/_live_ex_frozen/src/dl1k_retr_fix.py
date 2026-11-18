"""既定OFFの返し計算。既発火分と連鎖後の探索分を区別する。"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from src.phase_j._live_ex_frozen.src.exchange_count_ledger import STEP_SEC
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import duration
from src.phase_j._live_ex_frozen.src.prefire_pending_v47 import attack, fork_ledger, project
from src.phase_j._live_ex_frozen.src.prefire_post_v47 import remaining_send
from src.phase_j._live_ex_frozen.src.scoring import chain_power

OWN_CHAIN_ENV = 'PUYO_RETR_OWN_CHAIN'
LONG_ATTACK_ENV = 'PUYO_RETR_LONG_ATTACK_H'
RECEIVER_FORECAST_ENV = 'PUYO_RETR_RECEIVER_FORECAST'
MIN_STEP_BASE_SCORE = 40
HAND_EPSILON = 1e-9
OWN_DIAG_KEYS = ('I_before_own', 'own_chain', 'own_source', 'own_after', 'own_after_why',
                 'own_lower_bound', 'end_s')
LONG_DIAG_KEYS = ('end_before_long', 'end_s', 'long_attack_h')


def enabled(name: str) -> bool:
    """既定値も不正な値もOFF。"""
    return os.environ.get(name, '0') == '1'


def diagnostic_keys() -> tuple[str, ...]:
    """旗OFFでは診断の保存形式も変えない。"""
    keys = OWN_DIAG_KEYS if enabled(OWN_CHAIN_ENV) else ()
    keys += LONG_DIAG_KEYS if enabled(LONG_ATTACK_ENV) else ()
    keys += ('receiver_forecast_steps',) if enabled(RECEIVER_FORECAST_ENV) else ()
    if enabled('PUYO_RETR_SMALL_CHAIN_CONTINUE'):
        keys += ('small_chain',)
    if enabled('PUYO_RETR_FIRE_ORIGIN_HOLD'):
        keys += ('origin_s', 'origin_o')
    if enabled('PUYO_RETR_TSUMO_SYNC'):
        keys += ('tsumo_sync',)
    return tuple(dict.fromkeys(keys))


def current_forecast(engine: Any, side: int) -> Any:
    """共通予測はその時点で有効な連鎖だけを取得する。"""
    tracker = getattr(engine, 'retr_fire', None)
    return tracker.current(side, engine.state.t) if hasattr(tracker, 'current') else None


def receiver_forecast(engine: Any, side: int) -> Any:
    """既定OFF。短いEX予測に隠れた現在の共通予測を受け側にも使う。"""
    if not enabled(RECEIVER_FORECAST_ENV) or not engine._resolving(side):
        return None
    chain, fc = engine.chains[side], current_forecast(engine, side)
    if chain is None or fc is None or not fc.has_prediction:
        return None
    return fc if len(fc.steps) > len(chain['best']['scores']) else None


def updated_end(engine: Any, value: dict, side: int, end: float | None) -> float | None:
    """現在までの予測の伸びと段の遅れを毎回反映する。未来の実段数は使わない。"""
    if not enabled(LONG_ATTACK_ENV) or not engine._resolving(side):
        return end
    from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
    chain, now = engine.chains[side], engine.state.t
    candidates = [] if end is None else [end]
    if chain is not None:
        obs = chain.get('observed', [])
        n_obs = max((int(ob['count']) for ob in obs), default=0)
        n_pred = len(chain['best']['scores']) if value['board_prediction_active'][side] else n_obs
        if n_pred:
            candidates.append(float(chain['started']) + duration(n_pred))
        stamps = [float(ob['t']) for ob in obs if ob.get('t') is not None]
        if stamps:
            candidates.append(max(stamps) + max(1, n_pred - n_obs) * STEP_SEC)
    fc = current_forecast(engine, side)
    if fc is not None and fc.has_prediction:
        candidates.append(retr.forecast_end(fc, now))
    if fc is not None and fc.observed:
        candidates.append(float(fc.last_valid_t) + max(1, len(fc.steps)-len(fc.observed))*STEP_SEC)
    return max(now, *candidates) if candidates else end


def observed_scores(chain: dict) -> list[int]:
    """段の得点が無い箇所だけ4個1色消しの下限で埋める。未観測の未来は足さない。"""
    observed = chain.get('observed', [])
    count = max((int(ob['count']) for ob in observed), default=0)
    scores = [MIN_STEP_BASE_SCORE * max(1, chain_power(i+1)) for i in range(count)]
    for ob in observed:
        index = int(ob['count']) - 1
        if index >= 0 and ob.get('score') is not None:
            scores[index] = int(ob['score'])
    return scores


def unreflected(engine: Any, side: int, scores: list[int]) -> int:
    """段ごとのレート・端数を維持し、既に台帳へ反映した送りを除く。"""
    chain = dict(engine.chains[side])
    chain['best'] = dict(chain['best'], scores=scores)
    chain.pop('external_sent', None)
    snap = engine.state.sl_snap
    first = snap['margin_origin_sec']
    first = engine.state.first_move_t() if first is None else first
    first = engine.state.game_start_t if first is None else first
    return remaining_send(chain, snap[f'generated_p{side+1}'], first)


def own_material(engine: Any, value: dict, side: int) -> tuple[int, Any, str]:
    """既定はEX優先。新旗では長い共通予測を採用し、欠測時は観測下限を使う。"""
    chain, fc = engine.chains[side], current_forecast(engine, side)
    newer = receiver_forecast(engine, side)
    if newer is not None:
        return unreflected(engine, side, list(newer.steps)), newer.final_board, 'receiver_forecast'
    if value['board_prediction_active'][side] and chain is not None:
        amount = (int(value['remaining'][side]) if 'remaining' in value else
                  unreflected(engine, side, list(chain['best']['scores'])))
        return amount, value['boards'][side], 'ex'
    if chain is None:
        return 0, None, 'no_chain'
    if fc is not None and fc.has_prediction:
        return unreflected(engine, side, list(fc.steps)), fc.final_board, 'forecast'
    scores = list(fc.observed) if fc is not None else observed_scores(chain)
    return unreflected(engine, side, scores), None, 'observed_lower_bound'


def incoming_before_own(engine: Any, value: dict, side: int, diag: dict) -> int:
    """Iに受け側の未反映送りを含めず、Cとの二重相殺を防ぐ。攻め側自身の相殺は維持する。"""
    from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
    attacker, tracker = 1-side, getattr(engine, 'retr_fire', None)
    if diag['fire_back'] is not None:
        if hasattr(tracker, 'current'):
            fc = tracker.current(attacker, engine.state.t)
        else:
            fc = tracker.for_chain(attacker, engine.chains[attacker]['started'], engine.state.t)
        return retr.pending_with_fire(engine, attacker, fc.steps, exclude_side=side)[side]
    amounts = list(value['remaining'])
    amounts[side] = 0
    return int(project(engine.state.sl.core, amounts, engine.state.t, [None, None], [None, None])['pending'][side])


def confirmed_after_own(engine: Any, value: dict, side: int, diag: dict, own: int) -> int:
    """最新予告から相殺する台帳の順序を守り、古い確定予告を一律に減らさない。"""
    amounts = list(value['remaining'])
    attacker, tracker = 1-side, getattr(engine, 'retr_fire', None)
    if diag['fire_back'] is not None:
        fc = (tracker.current(attacker, engine.state.t) if hasattr(tracker, 'current') else
              tracker.for_chain(attacker, engine.chains[attacker]['started'], engine.state.t))
        chain = dict(engine.chains[attacker], best=dict(engine.chains[attacker]['best'], scores=list(fc.steps)))
        snap = engine.state.sl_snap
        first = snap['margin_origin_sec']
        first = engine.state.first_move_t() if first is None else first
        first = engine.state.game_start_t if first is None else first
        amounts[attacker] = remaining_send(chain, snap[f'generated_p{attacker+1}'], first)
    amounts[side] = own
    ledger = fork_ledger(engine.state.sl.core)
    for i, amount in enumerate(amounts):
        attack(ledger, i, int(amount), engine.state.t)
    return int(ledger._pending(side, True))


def after_chain(engine: Any, side: int, board: Any, h: int, rate: int,
                pending: int) -> tuple[int, int, str]:
    """予測最終盤と残り手数で組める分。不明な材料は推定の下限0と理由を返す。"""
    from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
    if h == 0:
        return 0, 0, 'h0'
    if board is None:
        return 0, 0, 'no_final_lower_bound'
    pairs, why = retr.seed_pairs(engine, side, True, getattr(engine, 'retr_seed_matcher', None))
    if pairs is None or len(pairs) < retr.DEPTH:
        return 0, 0, why if pairs is None else 'short_seed_pairs'
    enc = retr.encode(board, pairs)
    if enc is None:
        return 0, 0, 'five_colors_or_unknown'
    profile = (engine.retr_ama.profile(*enc, pending=pending, rate=rate) if retr.landing_enabled()
               else engine.retr_ama.profile(*enc))
    return (int(profile[min(h, retr.DEPTH)-1])//rate,
            int(profile[min(h, retr.BUSY_VISIBLE_HANDS)-1])//rate, 'ama')


def write_own_values(extra: dict, side: int, incoming: int, own: int, future: int,
                     visible: int, h: int, board: Any) -> float:
    """列の順序・単位を維持し、既発火分をCvisにも含める。盤不明の致死・空き比はNaN。"""
    from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
    capacity, remainder, lethal = own+future, incoming-own-future, math.nan
    if board is not None:
        lethal = float(remainder > 0 and retr.heights(board)[retr.DEATH_COL] +
                       min(max(remainder, 0), retr.DROP_CAP)//retr.COLS >= retr.DEATH_H)
    for stem, val in zip(retr.STEMS, (capacity, remainder, h, lethal)):
        extra[f'{stem}_{retr.NAMES[side]}'] = float(val)
    if retr.visible_enabled():
        cvis = own+visible
        ratios = (math.nan, math.nan)
        if board is not None:
            # 既発火分も見えるので、h=0による旧関数のゼロ化をここでは適用しない。
            cv = retr.visible_values(cvis, incoming, 1, True, board, None, 1)
            cc = retr.visible_values(capacity, incoming, 1, True, board, None, 1)
            ratios = (cv[-2], cc[-1])
        vals = (cvis, incoming-cvis, capacity-cvis, cvis/capacity if capacity else 1., *ratios)
        for stem, val in zip(retr.VISIBLE_STEMS, vals):
            extra[f'{stem}_{retr.NAMES[side]}'] = float(val)
    return float(remainder)


def apply_own(engine: Any, value: dict, side: int, ends: list,
              extra: dict, diag: dict, rate: int) -> tuple[str, float]:
    """連鎖中専用。旧値の持ち越しはせず、現在の未反映送りと連鎖後の探索を足す。"""
    from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
    own, final, source = own_material(engine, value, side)
    ends = list(ends)
    if source in ('forecast', 'receiver_forecast') and ends[side] is None:
        ends[side] = retr.forecast_end(current_forecast(engine, side), engine.state.t)
    board = None if final is None else np.asarray(final, dtype=np.int64).reshape(retr.ROWS, retr.COLS)
    incoming = incoming_before_own(engine, value, side, diag)
    attacker, now = 1-side, engine.state.t
    if not engine._resolving(attacker) and incoming == 0:
        return 'no_attack', math.nan
    h = 0
    if ends[side] is not None and ends[attacker] is not None and engine._resolving(attacker):
        h = min(retr.MAX_H, max(0, int((ends[attacker]+retr.SEC_PER_HAND-ends[side]+HAND_EPSILON)/retr.SEC_PER_HAND)))
    elif not engine._resolving(attacker) and incoming > own:
        h = 1
    pending = confirmed_after_own(engine, value, side, diag, own) if retr.landing_enabled() else 0
    future, visible, why = after_chain(engine, side, board, h, rate, pending)
    remainder = write_own_values(extra, side, incoming, own, future, visible, h, board)
    diag.update(I_before_own=diag['I'], I=incoming, h=h, own_chain=own, own_source=source,
                own_after=future, own_after_why=why, own_lower_bound=source in ('observed_lower_bound', 'no_chain')
                or (h > 0 and why != 'ama') or ends[side] is None, end_s=ends[side], t=now)
    return 'own_chain', remainder
