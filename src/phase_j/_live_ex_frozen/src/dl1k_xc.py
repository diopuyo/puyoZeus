"""依頼 B: 撃ち合いの決着の列 xc_* (dl1k 研究専用・既定 OFF、PUYO_EX_XC=1 で計算)。PC1 2026-10-29 の定め。

retR と同じ要求点 (EX の撃ち合いの行)・同じ ExEngine の状態 (射影の直後) で、両側の連鎖の予測を A の ChainForecast
(engine.retr_fire、B3 と同じ予測) から取る。列は p1/p2 の対 (符号つきの差の列は作らない。向き 1 は p1/p2 を入れ替えるだけ)。
- 予測を使う側: EX の連鎖が進行中 (chain が有り・終わっていない) で、ChainForecast がその側に予測 (has_prediction) と
  連鎖の後の盤を持つ時だけ。それ以外の側は「予測なし」(mk_xc = 0)。
- xc_net_recv_s (個): 撃ち合いが全部終わった後に s に残る正味のおじゃま。EX の射影 (PostV47.projection と同じ project) で、
  予測を使う側の連鎖の段の得点を予測の段 (観測済み + 残り) に替えて、両側の残りの送りを台帳に足して相殺した後の s の予告。
  残りの送りは remaining_send (段ごとの時刻のおじゃまレート (マージンタイム込み) と、発火時の端数の持ち越し、台帳に反映済みを引く)
  なので、定義の max(0, (pend_s − pend_o) + (rem_gen_o − rem_gen_s)) を台帳のレート・端数のとおりに計算したものになる。
  どちらも連鎖していなければ台帳の予告と同じ。
- xc_net_recv_nf_s: nonsat (b3_column_spec.nonsat、log1p(x)/(3+log1p(x)))。
- 撃ち合いの後の盤: 予測を使う側は予測の連鎖の後の盤 (Forecast.final_board)、それ以外は今の確定盤 (state.sides[s].board)。
  s が連鎖中で予測も救済も無い時も今の盤 (mk_xc = 0)。おじゃまはまだ降らせない。
  - xc_post_color_s: 色ぷよ (1..5) の数。
  - xc_post_fire_s・xc_post_maxchain_s: 同じ盤の ind_immediate_fire_power・ind_current_max_chain (b3_features.static_indicators、
    B3 の ind_* と同じ計算・正規化した score)。色の衝突 (5 色) の盤は NaN (B3 と同じ扱い)。
- xc_lethal_s (0/1)・xc_death_room_s (個): 撃ち合いの後の盤に xc_net_recv_s が全部降った時に、3 列目 (列番号 2) が
  可視最上段 (行 1、高さ 12) まで埋まるか。降らせ方は最悪側の決め打ち: 6 列に均等 (n//6 段ずつ) で、余りは 3 列目に入れる
  → 3 列目に入る段 = ceil(n/6)。致死 = n > 0 かつ 高さ_3 + ceil(n/6) ≥ 12。
  death_room = 致死になる最小の n − 今の n (0 以上): 空き段 room = 12 − 高さ_3 として 6·(room−1)+1 − n (room ≤ 0 なら 0)。
  retR の致死 (retR_lethal: 返した後の残り R の 1 回の着地 min(R,30)//6 で次の着地で死ぬか) とは意味が違う:
  こちらは「撃ち合いの正味が全部降ったら死ぬか」(1 回 30 個の上限を入れない)。境目は death_room の連続値で学習に任せる。
- mk_xc_s: 側 s の連鎖の予測を使ったか (1/0)。
撃ち合いの外の行は結合 (scripts/dl1k_retr_join.py) で NaN。"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from src.phase_j._live_ex_frozen.src.b3_column_spec import nonsat
from src.phase_j._live_ex_frozen.src.exchange_count_ledger import FORMULA_SCORE_DELAY_SEC, STEP_SEC

SIDES = (0, 1)
NAMES = ('p1', 'p2')
STEMS = ('xc_net_recv', 'xc_net_recv_nf', 'xc_post_color', 'xc_post_fire', 'xc_post_maxchain', 'xc_lethal', 'xc_death_room',
         'mk_xc')
COLUMNS = tuple(f'{s}_{p}' for s in STEMS for p in NAMES)
DEATH_COL, DEATH_H, ROWS, COLS = 2, 12, 13, 6
XC_ENV = 'PUYO_EX_XC'


def xc_enabled() -> bool:
    return os.environ.get(XC_ENV) == '1'


def blank() -> dict:
    return {c: math.nan for c in COLUMNS}


def col_height(board: np.ndarray, col: int = DEATH_COL) -> int:
    """dl1k_retr.heights と同じ (行 0 = 隠し段が上、高さは隠し段を含む)。"""
    occ = np.asarray(board)[:, col] != 0
    return int(ROWS-occ.argmax()) if occ.any() else 0


def lethal_room(height3: int, n: int) -> tuple[float, float]:
    """3 列目の高さと降る数 n から (致死, あと何個で致死)。最悪側の降らせ方 (3 列目に ceil(n/6) 段)。"""
    n = max(int(n), 0)
    room = DEATH_H-height3
    lethal = float(n > 0 and height3+(n+5)//6 >= DEATH_H)
    death_room = float(max(0, 6*(room-1)+1-n)) if room > 0 else 0.0
    return lethal, death_room


def post_indicators(board: np.ndarray) -> tuple[float, float]:
    """盤の (ind_immediate_fire_power, ind_current_max_chain)。B3 の ind_* と同じ計算 (色の衝突は NaN)。"""
    from src.phase_j._live_ex_frozen.src.b3_column_spec import IND_NAMES
    from src.phase_j._live_ex_frozen.src.b3_features import static_indicators
    from src.phase_j._live_ex_frozen.src.b3_state import color_conflict
    g = np.ascontiguousarray(board, dtype=np.int8)
    if color_conflict(g):
        return math.nan, math.nan
    vals = dict(zip(IND_NAMES, static_indicators(g.tobytes())))
    return float(vals['immediate_fire_power']), float(vals['current_max_chain'])


def pending_with_forecast(engine: Any, steps: dict[int, tuple[int, ...]]) -> list[int]:
    """dl1k_retr.pending_with_fire を両側に広げたもの: steps にある側の連鎖の段の得点を替えて PostV47.projection と同じ射影。"""
    from src.phase_j._live_ex_frozen.src.prefire_pending_v47 import project
    from src.phase_j._live_ex_frozen.src.prefire_post_v47 import remaining_send
    snap, now = engine.state.sl_snap, engine.state.t
    first = snap['margin_origin_sec']
    first = engine.state.first_move_t() if first is None else first
    first = engine.state.game_start_t if first is None else first
    amounts, triggers, ends = [], [], []
    for i, chain in enumerate(engine.chains):
        active = chain is not None and not chain['finished']
        if active and i in steps:
            chain = dict(chain, best=dict(chain['best'], scores=list(steps[i])))
        scores = chain['best']['scores'] if active else []
        times = [chain['started']+FORMULA_SCORE_DELAY_SEC+k*STEP_SEC for k in range(len(scores))]
        amounts.append(remaining_send(chain, snap[f'generated_p{i+1}'], first))
        triggers.append(chain['started'] if active else None)
        ends.append(max(now, times[-1]) if times else None)
    return list(project(engine.state.sl.core, amounts, now, triggers, ends)['pending'])


def compute_xc(engine: Any, value: dict) -> tuple[dict, list[str]]:
    """engine の今の状態 (projection() の直後) から xc の 16 列と側ごとの理由。value は使わない (compute_retr と同じ呼び方にそろえるため
    受ける。None でよい)。"""
    extra, why = blank(), ['', '']
    now = engine.state.t
    tracker = getattr(engine, 'retr_fire', None)
    fcs: list = [None, None]
    for i in SIDES:
        chain = engine.chains[i]
        active = chain is not None and not chain['finished']
        f = tracker.current(i, now) if tracker is not None and hasattr(tracker, 'current') else None
        if active and f is not None and f.has_prediction and f.final_board is not None:
            fcs[i] = f
    net = pending_with_forecast(engine, {i: tuple(f.steps) for i, f in enumerate(fcs) if f is not None})
    for s in SIDES:
        p = NAMES[s]
        if fcs[s] is not None:
            board, why[s] = np.asarray(fcs[s].final_board, dtype=np.int64).reshape(ROWS, COLS), 'forecast'
        else:
            raw = engine.state.sides[s].board
            board = None if raw is None else np.asarray(raw, dtype=np.int64).reshape(ROWS, COLS)
            why[s] = 'no_board' if board is None else 'chaining_no_forecast' if engine._resolving(s) else 'board'
        n = int(net[s])
        extra[f'xc_net_recv_{p}'] = float(n)
        extra[f'xc_net_recv_nf_{p}'] = float(nonsat(float(n)))
        extra[f'mk_xc_{p}'] = float(fcs[s] is not None)
        if board is None:
            continue
        extra[f'xc_post_color_{p}'] = float(np.isin(board, (1, 2, 3, 4, 5)).sum())
        extra[f'xc_post_fire_{p}'], extra[f'xc_post_maxchain_{p}'] = post_indicators(board)
        extra[f'xc_lethal_{p}'], extra[f'xc_death_room_{p}'] = lethal_room(col_height(board), n)
    engine.xc_diag = dict(net=net, src=[None if f is None else f.source for f in fcs],
                          total=[None if f is None else f.total for f in fcs])
    return extra, why
