"""時間軸の打ち返し (設計 §2.2-2.4): 攻撃側 a の残り連鎖時間 → 受け側 b の置ける手数 H → 近未来火力 Q → 余地 M。

一往復のうち a→b を実装。b→a (再打ち返し) は second_leg() に口だけ置き、全行 DEFERRED で返す。
理由: 2回目の探索は b の予測連鎖数が決まってから走るため 1行あたり探索が倍になり、
v1 の目的 (行と特徴の生成・リーク検査) に対し費用が見合わない。次段で費用実測の後に埋める。
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np

from src import indicators_v2 as iv
from src.b3_column_spec import Mask, TA_VARIANTS, nonsat, signed_nonsat
from src.b3_search import HMAX_TOTAL_HANDS, IdealSearchResult, ideal_fire_search, points_to_ojama
from src.b3_state import Episode, SideState, color_conflict
from src.b3_timing import TimingTable, column_heights
from src.scoring import BASE_SCORE_PER_PUYO, MAX_BONUS_MULTIPLIER, ALL_CLEAR_BONUS, chain_power

NAN = float("nan")
CHAIN_ANIM_CALIBRATION = "empirical_table_2026_08_14"
UNCANCELLABLE_OJAMA = 72   # 12段分 (6列x12): 相殺不能の予告 (user 2026-10-05; 特徴のみ)
MIN_HANDS = 1              # 窓が0でも着弾ゲートの最後の1手は数える (設計 §2.2-3)
BOARD_CELLS = 78
PAIR_CELLS = 2
MIN_POP = 4
MAX_GROUP_BONUS = 10
MAX_COLOR_BONUS = 24
FORMULA_SLACK = 0.05
PREFIRE_EXTRA_CELLS = 4  # 発火前の確定盤面から最大2組の補完を許す。


def chain_remaining(ep: Episode, t: float, table: TimingTable) -> tuple[float, float, float]:
    """(残り秒 段時間版, 残り秒 実測テーブル版, 残り段数)。終了合図/閉鎖後は 0。"""
    if not np.isfinite(ep.pred_l):
        return NAN, NAN, NAN
    n_obs = len(ep.steps)
    length = max(ep.pred_l, float(n_obs))
    rem_stages = max(length - n_obs, 0.0)
    if not ep.active:
        return 0.0, 0.0, rem_stages
    anchor_t = ep.steps[-1][0] if n_obs else ep.first_trigger
    anchor_n = max(n_obs, 1)
    end = anchor_t + max(length - anchor_n, 0.0) * table.stage_sec + table.tail_sec
    table_end = ep.first_trigger + iv.estimate_chain_anim_duration_sec(length, CHAIN_ANIM_CALIBRATION)
    return max(end - t, 0.0), max(table_end - t, 0.0), rem_stages


def _max_height(grid: np.ndarray) -> float:
    return float(column_heights(grid).max())


def time_axis_features(a: SideState, b: SideState, t: float, elapsed: float, colors: tuple[int, ...],
                       table: TimingTable, icon_b_total: float,
                       sl_b_pending: float = NAN, sl_mask: Mask = Mask.DEFERRED,
                       observed_colors: tuple[int, ...] = (), score_ledger: object = None) -> dict[str, float]:
    """攻撃側 a についての語幹値 (接尾辞なし)。mk_ta を '__mk' で返す。"""
    out: dict[str, float] = {"ta_active": 0.0, "ta_bound": 0.0}
    if any(color_conflict(s.board) or (s.episode is not None and s.episode.active and
           any(color_conflict(f.chain_before) for f in s.episode.frags)) for s in (a, b)):
        return _blank_ta(out, Mask.RECOGNITION_CONFLICT)
    ep = a.episode
    if ep is None or not ep.active:
        if b.episode is not None and b.episode.active and b.episode.pred_final is None and sl_b_pending > 0:
            return _bound_ta(a, b, t, elapsed, colors, table, icon_b_total, sl_b_pending, score_ledger)
        return _blank_ta(out, Mask.NOT_APPLICABLE)
    a_rem, _tab, _stages = chain_remaining(ep, t, table)
    b_ep = b.episode
    if not np.isfinite(a_rem) or (b_ep is not None and b_ep.active and b_ep.pred_final is None):
        return _bound_ta(a, b, t, elapsed, colors, table, icon_b_total, sl_b_pending, score_ledger)
    out["ta_active"] = 1.0
    board, busy = b.board, 0.0
    if b_ep is not None and b_ep.active:
        if b_ep.pred_final is None:
            return _blank_ta(out, Mask.UNPREDICTABLE, keep_active=True)
        busy = chain_remaining(b_ep, t, table)[0]
        board = np.frombuffer(b_ep.pred_final, dtype=np.int8).reshape(13, 6)
    if board is None or not np.isfinite(busy):
        return _blank_ta(out, Mask.ONE_SIDE_UNOBSERVED, keep_active=True)
    window = max(a_rem - busy, 0.0)
    d_med, d_fast, _key = table.hand_sec(_max_height(board))
    h_med, h_fast = math.floor(window / d_med) + MIN_HANDS, math.floor(window / d_fast) + MIN_HANDS
    res = ideal_fire_search(board, b.queue, colors, HMAX_TOTAL_HANDS, resolve_before_death=_is_dead(board),
                            observed_colors=observed_colors)
    if res.color_conflict:
        return _blank_ta(out, Mask.UNPREDICTABLE, keep_active=True)
    at_fire = elapsed + a_rem
    (s_med, _c1, tr1), (s_fast, _c2, tr2) = res.at(h_med), res.at(h_fast)
    q_med, q_fast = float(points_to_ojama(s_med, at_fire)), float(points_to_ojama(s_fast, at_fire))
    i_pred = float(points_to_ojama(ep.pred_score, at_fire))
    i_icon = icon_b_total + i_pred if np.isfinite(icon_b_total) else NAN
    i_sl = sl_b_pending
    out.update(ta_a_rem_sec=a_rem, ta_b_busy=busy, ta_window=window, ta_hand_med_sec=d_med,
               ta_hand_fast_sec=d_fast, ta_H_med=float(h_med), ta_H_fast=float(h_fast),
               ta_req_h=float(max(h_med, h_fast)), ta_done_h=float(min(max(h_med, h_fast), res.done_hands if res.beam_died else HMAX_TOTAL_HANDS)),
               ta_trunc=float(tr1 or tr2), ta_q_med=q_med, ta_q_fast=q_fast, ta_i_pred=i_pred, ta_i_sl=i_sl, ta_i_icon=i_icon,
               ta_q_med_nf=float(nonsat(q_med)), ta_q_fast_nf=float(nonsat(q_fast)))
    _fill_margins(out, q_med, q_fast, {"pred": i_pred, "sl": i_sl, "icon": i_icon})
    out.update({f"ta_{k}": v for k, v in res.diagnostics(max(h_med, h_fast)).items()})
    out["mk_ta_trunc"] = float(Mask.OK if tr1 or tr2 else Mask.NOT_APPLICABLE)
    out["__mk"] = float(Mask.OK if np.isfinite(i_sl) else sl_mask)
    return out


@lru_cache(maxsize=16384)
def _score_upper(cells: int, stage: int = 1) -> float:
    """各段最低4個以外の配置制約を外した算術上限。探索や再帰を増やさない。"""
    if cells < MIN_POP:
        return 0.0
    best, prefix = 0, 0
    for count in range(1, cells // MIN_POP + 1):
        power = chain_power(stage + count - 1)
        largest = cells - MIN_POP * (count - 1)
        # 連鎖係数は余剰を最終段へ全配分した上限、連結係数は全段で最大群数を許す。
        chain = MIN_POP * prefix + largest * power
        bonuses = cells * (MAX_COLOR_BONUS + (largest // MIN_POP) * MAX_GROUP_BONUS)
        best = max(best, min(cells * MAX_BONUS_MULTIPLIER, chain + bonuses))
        prefix += power
    return float(BASE_SCORE_PER_PUYO * best)


def _episode_generated(ledger: object, side: SideState) -> float:
    """試合累計ではなく当該エピソードの式に帰属した台帳生成量だけを合算する。"""
    ep = side.episode
    core = getattr(ledger, "core", None)
    if core is None or ep is None:
        return NAN
    index = 0 if side.name == "1P" else 1
    records = [r for r in core.chains if r.side == index]
    current = core.st[index].chain
    if current is not None:
        records.append(current)
    return float(sum(s.generated for r in records for s in r.steps
                     if s.t_formula >= ep.first_trigger - FORMULA_SLACK))


def _origin_board(ep: Episode) -> Optional[np.ndarray]:
    """最初の断片の発火前確定盤面を優先し、途中断片の盤面へ置き換えない。"""
    if not ep.frags:
        return None
    first = ep.frags[0]
    if first.pre is not None and first.pre.get("board") is not None:
        return np.asarray(first.pre["board"], dtype=np.int8)
    return first.chain_before


def _bound_material(b: SideState, colors: tuple[int, ...], hands: int
                    ) -> tuple[int, int, float, Optional[IdealSearchResult]]:
    """発火前盤面と観測済み消去から残量を得る。起点不明は全78マスを残す。"""
    ep = b.episode
    active = ep is not None and ep.active
    origin = _origin_board(ep) if active else b.board
    cells = BOARD_CELLS if origin is None else int(np.count_nonzero((origin >= 1) & (origin <= 5)))
    if origin is not None:
        cells += int(np.count_nonzero(origin == 10))
    if active and origin is not None and ep.frags[0].pre is not None:
        cells += PREFIRE_EXTRA_CELLS
    erased = sum(s[1] for s in ep.steps) if active and origin is not None else 0
    remaining = max(cells - int(erased), 0)
    if active and (origin is None or erased > cells or
                   ep.prediction.reason in ("observation_conflict", "no_completion_candidates")):
        remaining = BOARD_CELLS  # 起点欠落・材料矛盾では消去量を引けない。
    return remaining, len(ep.steps) + 1 if active else 1, 0.0, None


def _bound_q(cells: int, stage: int, k1: float, hands: int, generated: float,
             at_fire: float, active: bool, search: Optional[IdealSearchResult] = None) -> float:
    """探索は下限なので使わず、残材料を一度だけ数えた算術上限を返す。

    予測不能な連鎖の観測段数は真の段番号を保証しないため最大係数を使う。
    k1/search は呼出互換用。全消しと換算端数は可能な発火回数分を予約する。
    """
    material = cells + PAIR_CELLS * hands
    score = (BASE_SCORE_PER_PUYO * MAX_BONUS_MULTIPLIER * material
             if active else _score_upper(material))
    fires = hands + int(active)
    return generated + float(points_to_ojama(score + fires * ALL_CLEAR_BONUS, at_fire)) + fires


def _unknown_remaining(a: SideState, t: float, table: TimingTable, hand: float, busy: float) -> float:
    """残存色数/4が最大残段数。盤面も不明のときだけ従来の窓を与える。"""
    ep = a.episode
    if ep is None or not ep.active:
        return 0.0
    origin = _origin_board(ep)
    erased = int(sum(s[1] for s in ep.steps))
    anchor = ep.steps[-1][0] if ep.steps else ep.first_trigger
    if origin is None:
        origin = a.board
        if origin is None:
            return busy + (HMAX_TOTAL_HANDS - MIN_HANDS) * hand
        # 現在盤面に対して過去の消去数を再度引かない。
        erased, anchor = 0, t
    cells = int(np.count_nonzero(((origin >= 1) & (origin <= 5)) | (origin == 10)))
    if ep.frags and ep.frags[0].pre is not None:
        cells += MIN_POP  # 起点予測と同じ最大2組の発火補完も残段数の上側へ含める。
    remaining = max(cells - erased, 0)
    return max(anchor + (remaining // MIN_POP) * table.stage_sec + table.tail_sec - t, 0.0)


def _bound_ta(a: SideState, b: SideState, t: float, elapsed: float, colors: tuple[int, ...],
              table: TimingTable, icon: float, pending: float, ledger: object) -> dict[str, float]:
    """予測不能区間だけを補完する。マスク6を維持し、上限フラグを付ける。"""
    out = _blank_ta({"ta_active": 1.0, "ta_bound": 1.0}, Mask.UNPREDICTABLE)
    b_active = b.episode is not None and b.episode.active
    if b.board is None and not b_active:
        return _blank_ta({"ta_active": 0.0, "ta_bound": 0.0}, Mask.UNPREDICTABLE)
    d_med, d_fast, _ = table.hand_sec(None if b.board is None else _max_height(b.board))
    busy = chain_remaining(b.episode, t, table)[0] if b_active else 0.0
    busy = busy if np.isfinite(busy) else 0.0
    a_active = a.episode is not None and a.episode.active
    a_rem = chain_remaining(a.episode, t, table)[0] if a_active else 0.0
    unknown_rem = not np.isfinite(a_rem)
    if unknown_rem:
        a_rem = _unknown_remaining(a, t, table, max(d_med, d_fast), busy)
    window = max(a_rem - busy, 0.0)
    hm, hf = (math.floor(window / d) + MIN_HANDS for d in (d_med, d_fast))
    if unknown_rem and _origin_board(a.episode) is None and a.board is None:
        hm, hf = max(hm, HMAX_TOTAL_HANDS), max(hf, HMAX_TOTAL_HANDS)
    cells, stage, k1, search = _bound_material(b, colors, max(hm, hf))
    generated = _episode_generated(ledger, b) if b_active else 0.0
    at_fire = elapsed + a_rem
    if b_active:
        # 式の消去が得点反映より先でも、差し引いた材料の得点を上限から落とさない。
        observed = float(points_to_ojama(sum(s[2] for s in b.episode.steps), at_fire)) + len(b.episode.steps)
        generated = max(generated if np.isfinite(generated) else 0.0, observed)
    qm, qf = (_bound_q(cells, stage, k1, h, generated, at_fire, b_active, search) for h in (hm, hf))
    pred = a.episode.pred_score if a_active else 0.0
    ip = float(points_to_ojama(pred, at_fire)) if np.isfinite(pred) else NAN
    ii = icon + ip if np.isfinite(icon) else NAN
    out.update(ta_a_rem_sec=a_rem, ta_b_busy=busy, ta_window=window, ta_hand_med_sec=d_med,
               ta_hand_fast_sec=d_fast, ta_H_med=float(hm), ta_H_fast=float(hf), ta_req_h=float(max(hm, hf)),
               ta_done_h=0.0, ta_trunc=0.0, ta_q_med=qm, ta_q_fast=qf, ta_i_pred=ip, ta_i_sl=pending,
               ta_i_icon=ii, ta_q_med_nf=float(nonsat(qm)), ta_q_fast_nf=float(nonsat(qf)), ta_full=0.0)
    _fill_margins(out, qm, qf, {"pred": ip, "sl": pending, "icon": ii})
    return out


def _is_dead(board: np.ndarray) -> bool:
    from src.board import Board
    return bool(Board.from_list(board.astype(int).tolist()).is_dead())


def _fill_margins(out: dict[str, float], q_med: float, q_fast: float, inflow: dict[str, float]) -> None:
    qs = {"med": q_med, "fast": q_fast}
    for v in TA_VARIANTS:
        kind, speed = v.split("_")
        i = inflow[kind]
        m = qs[speed] - i if np.isfinite(i) else NAN
        out[f"ta_m_{v}"] = m
        out[f"ta_mg_{v}"] = float(signed_nonsat(m)) if np.isfinite(m) else NAN
        out[f"ta_recv_{v}"] = max(-m, 0.0) if np.isfinite(m) else NAN
        out[f"ta_send_{v}"] = max(m, 0.0) if np.isfinite(m) else NAN
        out[f"ta_unc72_{v}"] = float(max(-m, 0.0) >= UNCANCELLABLE_OJAMA) if np.isfinite(m) else NAN


_TA_STEMS = ("ta_a_rem_sec", "ta_b_busy", "ta_window", "ta_hand_med_sec", "ta_hand_fast_sec", "ta_H_med",
             "ta_H_fast", "ta_req_h", "ta_done_h", "ta_trunc", "ta_q_med", "ta_q_fast", "ta_i_pred", "ta_i_sl", "ta_i_icon",
             "ta_q_med_nf", "ta_q_fast_nf", "ta_trunc_empty", "ta_trunc_empty_nf", "ta_full") + tuple(
    f"ta_{k}_{v}" for v in TA_VARIANTS for k in ("m", "mg", "recv", "send", "unc72"))


def _blank_ta(out: dict[str, float], mask: Mask, keep_active: bool = False) -> dict[str, float]:
    for k in _TA_STEMS:
        out[k] = NAN
    out["__mk"] = float(mask)
    out["mk_ta_trunc"] = float(mask)
    return out


def second_leg() -> dict[str, float]:
    """a→b→a の後半 (v1 未実装の口)。全列 NaN・mask=DEFERRED。"""
    return {"ta2_q_med": NAN, "ta2_m_med": NAN, "__mk": float(Mask.DEFERRED)}
