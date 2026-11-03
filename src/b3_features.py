"""状態 (GameState) から1行ぶんの特徴辞書を作る。全列は src/b3_column_spec の台帳と一致させる。

局面は user 定義 (色1-5+おじゃま9, row0込み, UNKNOWN除外, 両側max, <=18/19-47/>=48)。
is_dead 盤面でも火力を0にしない (resolve_before_death 探索 + is_dead フラグ)。
予告の主入力は得点台帳sl_*。旧会計台帳はr_ledger_* (R再現専用)、アイコンは補助列。
"""
from __future__ import annotations

from functools import lru_cache
import json
from typing import Optional

import numpy as np

from src import indicators_v2 as iv
from src import b3_trial_config as production_config
from src.b3_column_spec import (ICON_TYPES, IND_NAMES, K_LEVELS, RAW_COUNT_INDICATORS, Mask,
                                SIDES, nonsat)
from src.b3_net_ledger import NET_SWITCH_COLUMN, derive_net_row
from src.b3_search import HMAX_TOTAL_HANDS, SEARCH_VERSION, ideal_fire_search, points_to_ojama
from src.b3_state import Episode, GameState, SideState, color_count, color_conflict
from src.b3_static import evaluate as evaluate_static
from src.b3_timeaxis import second_leg, time_axis_features
from src.b3_timing import TimingTable, column_heights
from src.board import Board
from src.scoring import compute_effective_rate
from src.exchange_prefire_candidates import MAX_HANDS, PAIR_SIZE

NAN = float("nan")
EARLY_MAX, LATE_MIN = 18, 48        # user 定義局面 (src/exchange_event_user_phase.py と同値)
ICON_DISPLAY_MAX = 6                # 予告アイコンの表示上限 (超過分は表示落ち)
ICON_UB_SLACK = 30                  # 相手得点由来の予告上限に足す余裕 (岩1個)
BIG_ICONS = ("crown", "moon", "star")
LANDING_RECENT_SEC = 2.0
HONSEN_RATIO = 0.6
SAISOKU_MIN_SENT = 4                # 送り4個以下は整地 (催促でない)
CHAIN7 = 7
COUNT_ONLY_INDICATORS = frozenset({"board_color_puyo_total", "board_puyo_total", "max_column_height",
    "column_bumpiness", "death_margin", "death_margin_neighbor", "center_bulge_color",
    "center_bulge_ojama", "board_ojama_count", "buried_hole_count"})
STATIC_CACHE_SIZE = 16384

_IND_FUNCS = {n: getattr(iv, n) for n in IND_NAMES if n not in RAW_COUNT_INDICATORS}


@lru_cache(maxsize=STATIC_CACHE_SIZE)
def _static_indicators_cached(raw: bytes) -> tuple[float, ...]:
    """既存静止指標30種 (R1s の D 列の元)。FULL_MODEL_GRID_REGISTRY と同じ関数・同じ順。"""
    board = Board.from_list(np.frombuffer(raw, dtype=np.int8).reshape(13, 6).astype(int).tolist())
    vals = evaluate_static(board, _IND_FUNCS)
    return tuple(vals[n] for n in IND_NAMES)


static_indicators = _static_indicators_cached


@lru_cache(maxsize=STATIC_CACHE_SIZE)
def board_is_dead(raw: bytes) -> bool:
    return bool(Board.from_list(np.frombuffer(raw, dtype=np.int8).reshape(13, 6).astype(int).tolist()).is_dead())


def clear_static_caches() -> None:
    """試合境界で盤面結果を破棄する。計装による公開関数のラップにも依存しない。"""
    _static_indicators_cached.cache_clear()
    board_is_dead.cache_clear()


def _put(row: dict, stem: str, side: str, value: float) -> None:
    row[f"{stem}_{side}"] = value


def _mk(row: dict, stem: str, side: str, mask: Mask) -> None:
    row[f"{stem}_{side}"] = int(mask)


def board_group(row: dict, st: SideState, side: str, t: float, elapsed: float, colors: tuple[int, ...],
                observed_colors: tuple[int, ...] = ()) -> None:
    """盤面・静止指標・K火力・NEXT・得点。"""
    names = ("cells_color", "cells_ojama", "cells_unknown", "cells_total", "max_height", "board_age", "is_dead")
    if st.board is None:
        for n in names:
            _put(row, n, side, NAN)
        for n in IND_NAMES:
            _put(row, f"ind_{n}", side, NAN)
        for k in K_LEVELS:
            _put(row, f"k{k}_ojama", side, NAN)
            _put(row, f"k{k}_nf", side, NAN)
        _put(row, "search_done_h", side, NAN)
        for stem in ("search_trunc_empty", "search_trunc_empty_nf", "search_full"):
            _put(row, stem, side, NAN)
        for m in ("mk_board", "mk_ind", "mk_k", "mk_search_trunc"):
            _mk(row, m, side, Mask.NOT_YET_AVAILABLE)
        row[f"av_board_{side}"] = NAN
    else:
        _fill_board(row, st, side, t, elapsed, colors, observed_colors)
    _queue_score(row, st, side, t)


def _fill_board(row: dict, st: SideState, side: str, t: float, elapsed: float, colors: tuple[int, ...],
                observed_colors: tuple[int, ...] = ()) -> None:
    g = st.board
    raw = np.ascontiguousarray(g, dtype=np.int8).tobytes()
    color = float(np.isin(g, (1, 2, 3, 4, 5)).sum())
    ojama = float((g == 9).sum())
    dead = board_is_dead(raw)
    for n, v in (("cells_color", color), ("cells_ojama", ojama), ("cells_unknown", float((g == 10).sum())),
                 ("cells_total", color + ojama), ("max_height", float(column_heights(g).max())),
                 ("board_age", t - st.board_t), ("is_dead", float(dead))):
        _put(row, n, side, v)
    if color_conflict(g):
        board = Board.from_list(g.astype(int).tolist())
        vals = {n: float(f(board).score) for n, f in _IND_FUNCS.items() if n in COUNT_ONLY_INDICATORS}
        for n, v in vals.items():
            _put(row, f"ind_{n}", side, v)
        _conflicting_board(row, st, side)
        return
    for n, v in zip(IND_NAMES, static_indicators(raw)):
        _put(row, f"ind_{n}", side, v)
    res = ideal_fire_search(g, st.queue, colors, HMAX_TOTAL_HANDS, resolve_before_death=dead,
                            observed_colors=observed_colors)
    for k in K_LEVELS:
        pts, _chain, _trunc = res.at(k + iv.NEAR_FUTURE_KNOWN_HAND_SLOTS)
        o = NAN if res.color_conflict else float(points_to_ojama(pts, elapsed))
        _put(row, f"k{k}_ojama", side, o)
        _put(row, f"k{k}_nf", side, float(nonsat(o)))
    _put(row, "search_done_h", side, float(res.done_hands if res.beam_died else HMAX_TOTAL_HANDS))
    for key, value in res.diagnostics(max(K_LEVELS) + iv.NEAR_FUTURE_KNOWN_HAND_SLOTS).items():
        _put(row, f"search_{key}", side, value)
    _mk(row, "mk_search_trunc", side, Mask.OK if np.isfinite(row[f"search_trunc_empty_{side}"]) else Mask.NOT_APPLICABLE)
    _mk(row, "mk_board", side, Mask.OK)
    _mk(row, "mk_k", side, Mask.UNPREDICTABLE if res.color_conflict else Mask.OK)
    has_nan = any(not np.isfinite(row[f"ind_{n}_{side}"]) for n in IND_NAMES)
    _mk(row, "mk_ind", side, Mask.NOT_APPLICABLE if has_nan else Mask.OK)  # 低充填の saturation は未計測(NaN)
    row[f"av_board_{side}"] = st.board_t


def _conflicting_board(row: dict, st: SideState, side: str) -> None:
    """単純個数・形状は保持し、色に依存する指標と探索を欠測にする。"""
    for n in set(IND_NAMES) - COUNT_ONLY_INDICATORS:
        _put(row, f"ind_{n}", side, NAN)
    for k in K_LEVELS:
        for suffix in ("ojama", "nf"):
            _put(row, f"k{k}_{suffix}", side, NAN)
    for stem in ("search_done_h", "search_trunc_empty", "search_trunc_empty_nf", "search_full"):
        _put(row, stem, side, NAN)
    for mask in ("mk_board", "mk_ind", "mk_k", "mk_search_trunc"):
        _mk(row, mask, side, Mask.RECOGNITION_CONFLICT)
    row[f"av_board_{side}"] = st.board_t


def _queue_score(row: dict, st: SideState, side: str, t: float) -> None:
    known = any(v > 0 for v in st.queue) and np.isfinite(st.queue_t)
    _put(row, "queue_age", side, t - st.queue_t if known else NAN)
    _mk(row, "mk_queue", side, Mask.OK if known else Mask.NOT_YET_AVAILABLE)
    row[f"av_queue_{side}"] = st.queue_t if known else NAN
    _put(row, "score", side, NAN if st.score is None else st.score)
    _mk(row, "mk_score", side, Mask.NOT_YET_AVAILABLE if st.score is None else Mask.OK)
    if st.score_conflict:
        _mk(row, "mk_score", side, Mask.RECOGNITION_CONFLICT)
    row[f"lean_row_{side}"] = st.lean_row


def icon_group(row: dict, gs: GameState, i: int, side: str, rate: int, pred_active_score: float) -> float:
    """画面アイコン (主入力)。補正値の総数を返す (時間軸の I に使う)。"""
    st, opp = gs.sides[i], gs.sides[1 - i]
    ic = st.icon
    for tname in ICON_TYPES:
        _put(row, f"icon_raw_{tname}", side, ic.raw[tname])
    _put(row, "icon_raw_total", side, ic.raw_total)
    row[f"av_icon_{side}"] = ic.t
    corr = ic.corrected()
    if corr is None:
        for tname in ICON_TYPES:
            _put(row, f"icon_corr_{tname}", side, NAN)
        for n in ("icon_corr_total", "icon_corr_nf", "icon_big_clipped", "icon_ub", "icon_omitted"):
            _put(row, n, side, NAN)
        _mk(row, "mk_icon", side, Mask.NOT_YET_AVAILABLE)
        return NAN
    hist = [s for _t, s in opp.score_hist]
    gain = (max(hist[-1] - min(hist), 0.0) if hist else 0.0) + (pred_active_score if np.isfinite(pred_active_score) else 0.0)
    ub = float(gain // rate + ICON_UB_SLACK)
    clipped = 0.0
    total = sum(corr.values())
    for big in BIG_ICONS:
        if total > ub and corr[big] > 0:
            total -= corr[big]
            corr[big] = 0.0
            clipped = 1.0
    for tname in ICON_TYPES:
        _put(row, f"icon_corr_{tname}", side, corr[tname])
    _put(row, "icon_corr_total", side, total)
    _put(row, "icon_corr_nf", side, float(nonsat(total)))
    _put(row, "icon_big_clipped", side, clipped)
    _put(row, "icon_omitted", side, float(len(ic.last_icons) >= ICON_DISPLAY_MAX))
    _put(row, "icon_ub", side, ub)
    _mk(row, "mk_icon", side, Mask.OK)
    return total


def sl_group(row: dict, gs: GameState) -> None:
    """得点ベース台帳 v2 (主入力)。モジュール未完成/未出力の間は全列 NaN + mask。"""
    snap = gs.sl_snap
    from src.ledger_x_options import partial_enabled
    if partial_enabled():
        row['sl_pending_partial'] = int(bool(snap and snap.get('pending_partial')))
    mask = Mask.DEFERRED if gs.sl is None else Mask.NOT_YET_AVAILABLE
    if snap:
        mask = (Mask.HISTORY_INCOMPLETE if snap.get("history_incomplete") else
                Mask.NOT_YET_AVAILABLE if snap.get("history_waiting") else Mask.OK)
    for i, side in enumerate(SIDES):
        n = f"p{i + 1}"
        for stem, key in (("sl_pending", "pending"), ("sl_generated", "generated"), ("sl_offset", "offset"),
                          ("sl_dropped", "dropped"), ("sl_carry", "carry"),
                          ("sl_observed_landing", "observed_landing"), ("sl_board_increase", "board_increase"),
                          ("sl_board_excess", "board_excess")):
            _put(row, stem, side, float(snap[f"{key}_{n}"]) if snap else NAN)
        _put(row, "sl_pending_nf", side, float(nonsat(row[f"sl_pending_{side}"])) if snap else NAN)
        _mk(row, "mk_sl", side, mask)
    row["sl_rate"] = float(snap["rate"]) if snap else NAN
    row["mk_sl_c"] = int(Mask.OK if snap else mask)
    row["av_ledger"] = float(snap["available_at"]) if snap else NAN
    sl_inflight_group(row, gs, mask)


def sl_inflight_group(row: dict, gs: GameState, mask: Mask) -> None:
    """A有効時だけ未反映量と専用maskを追加し、OFFの行をそのまま保つ。"""
    from src.b3_column_spec import SPEC_BY_NAME
    if "sl_inflight_p1" not in SPEC_BY_NAME:
        return
    snap = gs.sl_snap or {}
    for side in SIDES:
        available = f"inflight_{side}" in snap
        for stem in ("inflight", "inflight_send"):
            value = float(snap.get(f"{stem}_{side}", NAN))
            _put(row, f"sl_{stem}", side, value)
            _put(row, f"sl_{stem}_nf", side, float(nonsat(value)))
        _mk(row, "mk_sl_inflight", side, mask if available else Mask.DEFERRED)


def land_group(row: dict, st: SideState, side: str, t: float) -> None:
    ld = st.landing
    _put(row, "land_count", side, float(st.land_count))
    if ld is None:
        for n in ("land_icon_drop", "land_board_net", "land_age"):
            _put(row, n, side, NAN)
        _mk(row, "mk_land", side, Mask.NOT_YET_AVAILABLE)
        return
    _put(row, "land_icon_drop", side, ld.icon_drop)
    _put(row, "land_board_net", side, ld.board_net)
    _put(row, "land_age", side, t - ld.finalize_t)
    ok = np.isfinite(ld.icon_drop) and np.isfinite(ld.board_net)
    _mk(row, "mk_land", side, Mask.OK if ok else Mask.UNOBSERVABLE)


def chain_group(row: dict, st: SideState, side: str, t: float, table: TimingTable, elapsed: float) -> None:
    from src.b3_timeaxis import chain_remaining
    ep = st.episode
    active = ep is not None and ep.active
    row[f"chain_pred_reason_{side}"] = ep.prediction.reason if active else "no_active_episode"
    row[f"chain_pred_final_{side}"] = (json.dumps(np.frombuffer(ep.pred_final, dtype=np.int8).reshape(13, 6).tolist())
                                        if active and ep.pred_final is not None else None)
    row[f"chain_observed_score_{side}"] = ep.prediction.observed_score if active else NAN
    row[f"av_chain_{side}"] = ep.prediction.available_at if active else NAN
    stems = ("chain_n_steps", "chain_pred_l", "chain_rem_stages", "chain_rem_sec", "chain_rem_sec_table",
             "chain_age", "chain_pred_score", "chain_pred_ojama", "endsig_stable", "endsig_next", "endsig_ojama")
    if ep is None or not ep.active:
        _put(row, "chain_active", side, 0.0)
        for n in stems:
            _put(row, n, side, NAN)
        _mk(row, "mk_chain", side, Mask.NOT_APPLICABLE)
        return
    rem, rem_tab, stages = chain_remaining(ep, t, table)
    vals = (float(len(ep.steps)), ep.pred_l, stages, rem, rem_tab, t - ep.first_trigger, ep.pred_score,
            float(points_to_ojama(ep.pred_score, elapsed)) if np.isfinite(ep.pred_score) else NAN,
            float(ep.sig_stable_t is not None), float(ep.sig_next_t is not None), float(ep.sig_ojama_t is not None))
    _put(row, "chain_active", side, 1.0)
    for n, v in zip(stems, vals):
        _put(row, n, side, v)
    _mk(row, "mk_chain", side, Mask.OK if np.isfinite(ep.pred_l) else Mask.UNPREDICTABLE)
    if ep.prediction.reason == "recognition_conflict":
        _mk(row, "mk_chain", side, Mask.RECOGNITION_CONFLICT)


def episode_group(row: dict, st: SideState, side: str, t: float, elapsed_at: float) -> None:
    """発火エピソード (閉じたもの) の本線/催促/7連鎖。式の段が完全なものだけ採用 (統括 2026-10-05)。"""
    n_closed = len(st.closed)
    _put(row, "ep_closed_n", side, float(n_closed))
    _put(row, "ep_denominator_formula", side, NAN)
    stems = ("ep_age", "ep_denominator", "ep_steps", "ep_consumed_formula", "ep_ratio_formula", "ep_consumed_board",
             "ep_ratio_board", "ep_honsen60_formula", "ep_honsen60_board", "ep_saisoku_formula", "ep_chain7plus",
             "ep_sent_ojama", "pf_color", "pf_before_color")
    for n in stems:
        _put(row, n, side, NAN)
    _put(row, "pf_available", side, 0.0)
    for m in ("mk_ep", "mk_ep_f", "mk_ep_b", "mk_pf"):
        _mk(row, m, side, Mask.NOT_APPLICABLE if n_closed == 0 else Mask.OK)
    fire_av = [x for x in (st.fire_t, st.formula_t) if np.isfinite(x)]
    row[f"av_fire_{side}"] = max(fire_av) if fire_av else NAN
    row[f"av_formula_{side}"] = st.formula_t if np.isfinite(st.formula_t) else NAN
    if n_closed == 0:
        return
    ep = st.closed[-1]
    _put(row, "ep_age", side, t - ep.closed_t)
    _put(row, "ep_denominator", side, ep.denominator)
    _mk(row, "mk_ep_b", side, Mask.OK if np.isfinite(ep.denominator) else Mask.UNOBSERVABLE)
    _formula_part(row, ep, side, elapsed_at)
    _board_part(row, ep, side)
    pre = next((f.pre for f in ep.frags if f.pre is not None), None)
    _put(row, "pf_available", side, float(pre is not None))
    _put(row, "pf_before_color", side, ep.denominator)
    if pre is not None:
        _put(row, "pf_color", side, color_count(np.asarray(pre["board"], dtype=np.int8)))
        _mk(row, "mk_pf", side, Mask.RECOGNITION_CONFLICT if
            color_conflict(np.asarray(pre["board"], dtype=np.int8)) else Mask.OK)
    else:
        _mk(row, "mk_pf", side, Mask.UNOBSERVABLE)
    prev = row[f"av_fire_{side}"]
    row[f"av_fire_{side}"] = ep.closed_t if not np.isfinite(prev) else max(prev, ep.closed_t)


def _formula_part(row: dict, ep: Episode, side: str, elapsed_at: float) -> None:
    if any(color_conflict(f.chain_before) for f in ep.frags):
        _mk(row, "mk_ep_f", side, Mask.RECOGNITION_CONFLICT)
        return
    n = len(ep.steps)
    complete = ep.steps_complete()
    if n >= CHAIN7:
        _put(row, "ep_chain7plus", side, 1.0)
    elif complete:
        _put(row, "ep_chain7plus", side, 0.0)
    if not complete:
        _mk(row, "mk_ep_f", side, Mask.UNOBSERVABLE)   # 式段0件(約47%)・段欠け・予測不能は不明層
        return
    erased = float(sum(s[1] for s in ep.steps))
    if any(s[1] < 0 for s in ep.steps):
        _mk(row, "mk_ep_f", side, Mask.RECOGNITION_CONFLICT)
        return
    points = float(sum(s[2] for s in ep.steps))
    sent = float(points_to_ojama(points, elapsed_at))
    _put(row, "ep_steps", side, float(n))
    _put(row, "ep_consumed_formula", side, erased)
    _put(row, "ep_sent_ojama", side, sent)
    denominator = ep.denominator
    if ep.pred_final is not None:
        # 同じ完走候補の残存数＋観測済み全段の消去数。発火補完分を含む材料保存則。
        denominator = erased + color_count(np.frombuffer(ep.pred_final, dtype=np.int8))
        added = denominator - ep.denominator
        if added not in range(0, PAIR_SIZE * MAX_HANDS + 1, PAIR_SIZE):
            _mk(row, "mk_ep_f", side, Mask.RECOGNITION_CONFLICT)
            return
    _put(row, "ep_denominator_formula", side, denominator)
    if np.isfinite(denominator) and denominator > 0 and 0 <= erased <= denominator:
        ratio = erased / denominator
        _put(row, "ep_ratio_formula", side, ratio)
        _put(row, "ep_honsen60_formula", side, float(ratio >= HONSEN_RATIO))
        _put(row, "ep_saisoku_formula", side, float(ratio < HONSEN_RATIO and sent > SAISOKU_MIN_SENT))
    else:
        _mk(row, "mk_ep_f", side, Mask.RECOGNITION_CONFLICT)


def _board_part(row: dict, ep: Episode, side: str) -> None:
    cons = ep.board_consumed()
    pre = next((f.pre for f in ep.frags if f.pre is not None), None)
    denominator = color_count(np.asarray(pre["board"], dtype=np.int8)) if pre else ep.denominator
    _put(row, "ep_denominator", side, denominator)
    if any(f.consumed_board is not None and f.consumed_board < 0 for f in ep.frags):
        _mk(row, "mk_ep_b", side, Mask.RECOGNITION_CONFLICT)
        return
    if cons is None or not np.isfinite(denominator) or denominator <= 0:
        _mk(row, "mk_ep_b", side, Mask.UNOBSERVABLE)
        return
    if cons > denominator:
        _mk(row, "mk_ep_b", side, Mask.RECOGNITION_CONFLICT)
        return
    ratio = cons / denominator
    _put(row, "ep_consumed_board", side, cons)
    _put(row, "ep_ratio_board", side, ratio)
    _put(row, "ep_honsen60_board", side, float(ratio >= HONSEN_RATIO))
    if pre and color_conflict(np.asarray(pre["board"], dtype=np.int8)):
        _mk(row, "mk_ep_b", side, Mask.RECOGNITION_CONFLICT)


def common_group(row: dict, gs: GameState, elapsed: float, rate: int) -> None:
    totals, unknown = [], 0.0
    for st in gs.sides:
        if st.board is not None:
            totals.append(float(np.isin(st.board, (1, 2, 3, 4, 5, 9)).sum()))
            unknown += float((st.board == 10).sum())
    tmax = max(totals) if totals else NAN
    row["phase_total_max"] = tmax
    row["phase"] = 0 if tmax <= EARLY_MAX else 1 if tmax < LATE_MIN else 2
    row["mk_phase"] = int(Mask.ONE_SIDE_UNOBSERVED if len(totals) < 2 else Mask.UNOBSERVABLE if unknown > 0 else Mask.OK)
    if any(color_conflict(st.board) for st in gs.sides):
        row["mk_phase"] = int(Mask.RECOGNITION_CONFLICT)
    row["elapsed_first_move"] = elapsed
    row["rate_now"] = float(rate)
    n_known, _ = gs.colors.known()
    row["n_known_colors"] = n_known


def mode_group(row: dict, gs: GameState, t: float, icon_totals: list[float]) -> None:
    bits = 0
    if any(s.episode is not None and s.episode.active for s in gs.sides):
        bits |= 1
    if any(s.state == "OJAMA_FALL" for s in gs.sides):
        bits |= 2
    if any(s.landing is not None and t - s.landing.finalize_t <= LANDING_RECENT_SEC for s in gs.sides):
        bits |= 4
    pending = icon_totals
    if production_config.b3_ledger_mode_enabled():
        pending = [(gs.sl_snap or {}).get(f"pending_p{i}", NAN) for i in (1, 2)]
    if any(np.isfinite(x) and x > 0 for x in pending):
        bits |= 8
    row["mode_bits"] = bits
    row["mode_exchange"] = int(bits != 0)


def ledger_group(row: dict, gs: GameState) -> None:
    """比較基準R再現専用 (特徴にも教師にも使わない)。"""
    L = gs.ledger
    for side, n in zip(SIDES, ("p1", "p2")):
        row[f"r_ledger_pending_{side}"] = float(L.get(f"pending_{n}_uncapped", NAN)) if L else NAN
        row[f"r_ledger_forecast_{side}"] = float(L.get(f"forecast_{n}", NAN)) if L else NAN
    row["r_ledger_net_balance"] = float(L.get("net_balance_capped", NAN)) if L else NAN


def diff_group(row: dict) -> None:
    """差分列。どちらか NaN なら NaN。正規化済みは中心化、生個数は生差分。"""
    def cdiff(a: float, b: float) -> float:
        return 0.5 + 0.5 * (a - b) if np.isfinite(a) and np.isfinite(b) else NAN

    def rdiff(a: float, b: float) -> float:
        return a - b if np.isfinite(a) and np.isfinite(b) else NAN
    for key, pairs in (("mk_df_ind", ("mk_ind",)), ("mk_df_k", ("mk_k",)), ("mk_df_board", ("mk_board",)),
                       ("mk_df_icon", ("mk_icon",))):
        row[key] = max(int(row[f"{pairs[0]}_p1"]), int(row[f"{pairs[0]}_p2"]))
    for n in IND_NAMES:
        f = rdiff if n in RAW_COUNT_INDICATORS else cdiff
        row[f"df_ind_{n}"] = f(row[f"ind_{n}_p1"], row[f"ind_{n}_p2"])
    for k in K_LEVELS:
        row[f"df_k{k}_nf"] = cdiff(row[f"k{k}_nf_p1"], row[f"k{k}_nf_p2"])
    row["df_cells_color"] = rdiff(row["cells_color_p1"], row["cells_color_p2"])
    row["df_cells_ojama"] = rdiff(row["cells_ojama_p1"], row["cells_ojama_p2"])
    row["df_icon_corr_total"] = rdiff(row["icon_corr_total_p1"], row["icon_corr_total_p2"])


def net_ledger_group(row: dict) -> None:
    """案A (既定OFF): 列仕様に正味の予告の列があるときだけ、同じ行の既存列から導出して足す。

    OFF (仕様に列が無い) では何もしない = 行の列も値も変わらない。導出は純関数 derive_net_row で、
    学習行の parquet に後から足す台本 (scripts/derive_b3_net_ledger_cols.py) と同じ関数を通る。
    """
    from src.b3_column_spec import SPEC_BY_NAME
    if NET_SWITCH_COLUMN not in SPEC_BY_NAME:
        return
    row.update(derive_net_row(row))


def build_features(gs: GameState, table: TimingTable) -> dict[str, object]:
    """1時点の特徴辞書 (メタ列・ラベルは b3_rows が付ける)。"""
    t = gs.t
    fm = gs.first_move_t()
    elapsed = max(t - fm, 0.0) if fm is not None else 0.0
    rate = compute_effective_rate(elapsed, from_first_move=True)
    n_known, colors = gs.colors.known()
    observed = tuple(c for c in colors if gs.colors.counts[c] > 0)
    row: dict[str, object] = {"search_version": SEARCH_VERSION}
    icon_totals = []
    for i, side in enumerate(SIDES):
        st = gs.sides[i]
        opp_ep = gs.sides[1 - i].episode
        opp_pred = opp_ep.pred_score if opp_ep is not None and opp_ep.active else NAN
        board_group(row, st, side, t, elapsed, colors, observed)
        icon_totals.append(icon_group(row, gs, i, side, rate, opp_pred))
        land_group(row, st, side, t)
        chain_group(row, st, side, t, table, elapsed)
        episode_group(row, st, side, t, elapsed)
    sl_group(row, gs)
    for i, side in enumerate(SIDES):
        receiver = SIDES[1 - i]
        sl_b = row[f"sl_pending_{receiver}"]
        sl_mask = Mask(row[f"mk_sl_{receiver}"])
        ta = time_axis_features(gs.sides[i], gs.sides[1 - i], t, elapsed, colors, table,
                                icon_totals[1 - i], sl_b, sl_mask, observed, score_ledger=gs.sl)
        mk = int(ta.pop("__mk"))
        if mk == Mask.OK and not np.isfinite(ta["ta_i_icon"]):
            mk = int(Mask.NOT_YET_AVAILABLE)  # 共通maskの対象であるアイコン由来列も欠測を反映
        for k, v in ta.items():
            _put(row, k, side, v)
        _mk(row, "mk_ta", side, Mask(mk))
        for k, v in second_leg().items():
            if k == "__mk":
                _mk(row, "mk_ta2", side, Mask(int(v)))
            else:
                _put(row, k, side, v)
    common_group(row, gs, elapsed, rate)
    mode_group(row, gs, t, icon_totals)
    ledger_group(row, gs)
    diff_group(row)
    net_ledger_group(row)
    return row
