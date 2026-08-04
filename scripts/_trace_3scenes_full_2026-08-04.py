"""user新規指摘3件「一気に振れて急回復」の完全トレース (2026-08-04)。

render_delta_winprob_demo.py と完全に同一の経路 (chain_windows込みで
_build_stable_timeline を呼ぶ、compute_display_state で表示値を得る) で
0.2秒刻みの内訳表を出す。user注意事項: chain_windows なしで _build_stable_
timeline を単体呼びすると別物の滑らかな系列が出るため、本スクリプトは
render_delta_winprob_demo.main() と同じ呼び方 (chain_windows/activity_windows
を必ず渡す) を厳守する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.chain import ChainSimulator
from scripts.compute_exchange_delta_winprob import (
    DEFAULT_LABELED_WIN_CSV,
    _VideoNpzCache,
    _aggregate_known_pending_net_ojama,
    _board_from_grid,
    _clamp_severity,
    _find_active_chain_window,
    _load_video_npz,
    _net_pending_after_cancellation,
    _realizable_counter_ojama,
    _safe_next_array,
    build_chain_in_progress_windows,
    build_event_activity_windows,
    train_winprob_models,
)
from scripts.render_delta_winprob_demo import build_fire_event_views, compute_display_state
from scripts.visualize_advantage_overlay import board_room

NPZ_DIR = Path("data/indicators_v2/boards_lean_olRyxDGacbg_2026-08-03")
DELTA_CSV = Path("data/verify/delta_winprob_olRyxDGacbg_2026-08-03/exchange_delta_winprob.csv")
GRID_STEP_SEC = 0.2

# (シーン名, game_idx, t0, t1)
# 2026-08-04 最終確認: scene5はmain指摘のt=3163 (色ぷよ量差+14相当、J-1実測で
# 説明済みの正当値) を含むよう開始点を3163.0に前倒しした。
SCENES = [
    ("scene1(match_01)", 1, 2925.0, 2945.0),
    ("scene2(match_02)", 2, 2977.0, 2995.0),
    ("scene3(match_03)", 3, 3020.0, 3040.0),
    ("scene4(match_04)", 4, 3090.0, 3105.0),
    ("scene5(match_05)", 5, 3163.0, 3172.0),
]


def _events_causing_pending(
    activity_windows: list, t: float, fire_side_of_pending: str,
) -> list:
    """時刻tで残存予告(remaining>0)に寄与しているfire_side側のイベント一覧
    (相打ち適用有無の突合用、_aggregate_known_pending_net_ojama と同じ判定)。
    """
    result = []
    for w in activity_windows:
        if w.fire_side != fire_side_of_pending or w.ignition_sec > t:
            continue
        result.append(w)
    return result


def trace_scene(
    name: str, game_idx: int, t0: float, t1: float,
    cache: _VideoNpzCache, models: dict, sim: ChainSimulator,
    delta_df: pd.DataFrame,
) -> pd.DataFrame:
    """1シーン分の完全トレース表を作る (render と同一経路)。"""
    events_df = delta_df.loc[(delta_df["game_idx"] == game_idx) & (~delta_df["match_failed"])].copy()
    chain_windows = build_chain_in_progress_windows(events_df, cache, sim)
    activity_windows = build_event_activity_windows(events_df, cache)
    from scripts.compute_exchange_delta_winprob import _build_stable_timeline
    timeline_df = _build_stable_timeline(cache, game_idx, models, sim, chain_windows, activity_windows)
    timeline_t = timeline_df["t_sec"].values.astype(float)
    timeline_v = timeline_df["winprob_1p"].values.astype(float)
    timeline_uncertain = timeline_df["is_uncertain"].values.astype(bool)
    events = build_fire_event_views(events_df, cache)

    mask1, mask2 = cache.r1p.game_idx == game_idx, cache.r2p.game_idx == game_idx
    t1_arr, g1 = cache.r1p.t_sec[mask1], cache.r1p.grids[mask1]
    t2_arr, g2 = cache.r2p.t_sec[mask2], cache.r2p.grids[mask2]
    # 修正H2 (2026-08-04): _build_stable_timeline と同じ既知ツモ抽出
    # (診断列と実際の表示値を完全一致させるため必須、以前ここが欠落して
    # 診断列が古い値のまま表示値だけ修正H2反映という不整合があった)。
    n1a, n1b = _safe_next_array(cache.r1p.next1_a, mask1), _safe_next_array(cache.r1p.next1_b, mask1)
    d1a, d1b = _safe_next_array(cache.r1p.dnext_a, mask1), _safe_next_array(cache.r1p.dnext_b, mask1)
    n2a, n2b = _safe_next_array(cache.r2p.next1_a, mask2), _safe_next_array(cache.r2p.next1_b, mask2)
    d2a, d2b = _safe_next_array(cache.r2p.dnext_a, mask2), _safe_next_array(cache.r2p.dnext_b, mask2)

    rows = []
    grid = np.arange(t0, t1, GRID_STEP_SEC)
    clamp_was_active = False  # ヒステリシス状態 (_build_stable_timeline と同じ意味論を再現)
    for t in grid:
        idx1 = int(np.searchsorted(t1_arr, t, side="right")) - 1
        idx2 = int(np.searchsorted(t2_arr, t, side="right")) - 1
        if idx1 < 0 or idx2 < 0:
            continue
        b1, b2 = _board_from_grid(g1[idx1]), _board_from_grid(g2[idx2])
        active = _find_active_chain_window(chain_windows, float(t))
        if active is not None:
            if active.fire_side == "1P":
                b1 = active.board_after
            else:
                b2 = active.board_after
        active_desc = f"{active.fire_side}仮想盤面" if active is not None else "無し(live)"

        attack_1p, attack_2p, chain_1p, chain_2p = _aggregate_known_pending_net_ojama(
            activity_windows, float(t), b1, b2)
        pending_on_1p, pending_on_2p = _net_pending_after_cancellation(attack_1p, attack_2p)

        next_1p, dnext_1p = (int(n1a[idx1]), int(n1b[idx1])), (int(d1a[idx1]), int(d1b[idx1]))
        next_2p, dnext_2p = (int(n2a[idx2]), int(n2b[idx2])), (int(d2a[idx2]), int(d2b[idx2]))

        clamp_state, p_prime, room_val, ef_raw, mutual_flag = "無効", np.nan, np.nan, np.nan, "-"
        severity = 0.0
        if pending_on_1p > 0.0 or pending_on_2p > 0.0:
            if pending_on_1p > 0.0:
                threatened, pending, attacker_chain, attacker_side = b1, pending_on_1p, chain_2p, "2P"
                next_pair, dnext_pair = next_1p, dnext_1p
            else:
                threatened, pending, attacker_chain, attacker_side = b2, pending_on_2p, chain_1p, "1P"
                next_pair, dnext_pair = next_2p, dnext_2p
            ef_raw = _realizable_counter_ojama(threatened, attacker_chain, next_pair, dnext_pair)
            room_val = board_room(threatened)
            raw_excess = pending - ef_raw - room_val
            p_prime = pending - ef_raw
            severity, clamp_was_active = _clamp_severity(raw_excess, clamp_was_active)
            clamp_state = (f"severity={severity:.2f}({attacker_side}有利)" if severity > 0.0
                           else "無効(severity=0)")
            causing = _events_causing_pending(activity_windows, float(t), attacker_side)
            mutual_flag = ",".join(
                str(bool(events_df.loc[events_df["t_sec"] == w.t_sec, "is_mutual_exchange"].iloc[0]))
                if (events_df["t_sec"] == w.t_sec).any() else "?"
                for w in causing
            ) or "-"

        state = compute_display_state(events, timeline_t, timeline_v, float(t), timeline_uncertain)
        display = np.nan if state.waiting else state.winprob_1p
        threatened_next = f"next={next_1p if pending_on_1p > 0 else next_2p}"
        rows.append({
            "t_sec": float(t), "有効窓": active_desc,
            "pending_1p": pending_on_1p, "pending_2p": pending_on_2p,
            "realizable_counter": ef_raw, "P_prime": p_prime, "room": room_val,
            "クランプ": clamp_state, "既知ネクスト(受け側)": threatened_next,
            "相打ち適用": mutual_flag,
            "判定保留": bool(state.uncertain_frozen), "表示値": display,
        })
    df = pd.DataFrame(rows)
    df.insert(0, "シーン", name)
    return df


def _describe_jumps(df: pd.DataFrame) -> None:
    """表示値が前行から大きく変化した行を検出し、跨いだ閾値を1行で説明する。"""
    JUMP_THRESHOLD = 20.0
    vals = df["表示値"].values
    for i in range(1, len(df)):
        if np.isnan(vals[i]) or np.isnan(vals[i - 1]):
            continue
        delta = vals[i] - vals[i - 1]
        if abs(delta) < JUMP_THRESHOLD:
            continue
        row, prev = df.iloc[i], df.iloc[i - 1]
        print(f"  [急変] t={prev['t_sec']:.1f}->{row['t_sec']:.1f}s: {vals[i-1]:.1f}->{vals[i]:.1f}"
              f" (Δ{delta:+.1f})")
        print(f"    直前: クランプ={prev['クランプ']} P'={prev['P_prime']:.1f} room={prev['room']:.1f}"
              if not pd.isna(prev['P_prime']) else f"    直前: クランプ={prev['クランプ']} (pending無し)")
        print(f"    直後: クランプ={row['クランプ']} P'={row['P_prime']:.1f} room={row['room']:.1f}"
              if not pd.isna(row['P_prime']) else f"    直後: クランプ={row['クランプ']} (pending無し)")


def main() -> None:
    print("=== 勝率モデル学習 ===")
    models = train_winprob_models(DEFAULT_LABELED_WIN_CSV)
    cache = _load_video_npz("olRyxDGacbg", NPZ_DIR)
    delta_df = pd.read_csv(DELTA_CSV)
    sim = ChainSimulator()

    out_dir = Path("data/verify/trace_3scenes_2026-08-04")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, game_idx, t0, t1 in SCENES:
        print(f"\n\n========== {name} (game_idx={game_idx}, {t0}-{t1}s) ==========")
        df = trace_scene(name, game_idx, t0, t1, cache, models, sim, delta_df)
        pd.set_option("display.width", 200)
        print(df.to_string(index=False))
        print(f"\n--- {name} 急変行の説明 ---")
        _describe_jumps(df)
        out_path = out_dir / f"{name.split('(')[0]}_trace.csv"
        df.to_csv(out_path, index=False)
        print(f"[保存] {out_path}")


if __name__ == "__main__":
    main()
