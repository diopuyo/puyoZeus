"""match_02 終盤 (2995-3007s) の1%超過小評価 真因分析 (2026-08-03 main指摘)。

user指摘: 「連鎖は発火した瞬間に火力が確定する」(progressive開示は不要、
現行の着火時全量計上タイミングは正しい)。今回は別の疑い=受け切れ判定
(_lethal_readout_clamp) の k_hands 過小評価による誤クランプの可能性を検証する。

計装1-4 (main指示) をそのまま数表出力する。修正は行わない (この段階は
実測確認のみ)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import src.indicators_v2 as iv
from src.chain import ChainSimulator
from scripts.compute_exchange_delta_winprob import (
    DEFAULT_LABELED_WIN_CSV,
    _VideoNpzCache,
    _aggregate_known_pending_net_ojama,
    _board_from_grid,
    _find_active_chain_window,
    _load_video_npz,
    _net_pending_after_cancellation,
    build_chain_in_progress_windows,
    build_event_activity_windows,
    compute_board_only_features,
    train_winprob_models,
    winprob_attacker,
    winprob_to_score100,
)
from scripts.measure_exchange_effectiveness import (
    MAX_SUPPORTED_K_HANDS,
    estimate_available_hands,
    estimate_landing_delay_sec,
)
from scripts.visualize_advantage_overlay import board_room

NPZ_DIR = Path("data/indicators_v2/boards_lean_olRyxDGacbg_2026-08-03")
DELTA_CSV = Path("data/verify/delta_winprob_olRyxDGacbg_2026-08-03/exchange_delta_winprob.csv")
GAME_IDX = 2
T_START, T_END = 2980.0, 3011.0

# main実測アンカー (実効アニメ時間の物差し比較用、user実測14.5秒/8連鎖)
REAL_ANIM_SEC_PER_8CHAIN = 14.5


def _color_puyo_count(board) -> int:
    """空/おじゃま以外の色ぷよ数 (1P盤面の成長確認用、簡易カウント)。"""
    grid = board.to_dict()["grid"]
    return int(sum(1 for row in grid for c in row if c not in (0, 9)))


def main() -> None:
    print("=== 勝率モデル学習 ===")
    models = train_winprob_models(DEFAULT_LABELED_WIN_CSV)
    cache = _load_video_npz("olRyxDGacbg", NPZ_DIR)
    delta_df = pd.read_csv(DELTA_CSV)
    sim = ChainSimulator()

    events_df = delta_df.loc[
        (delta_df["game_idx"] == GAME_IDX) & (~delta_df["match_failed"])].copy()
    print(f"\n=== game_idx={GAME_IDX} 発火イベント一覧 ===")
    print(events_df[["t_sec", "fire_side", "net_ojama_after", "approx_fire_chains"]].to_string())

    chain_windows = build_chain_in_progress_windows(events_df, cache, sim)
    activity_windows = build_event_activity_windows(events_df, cache)
    print("\n=== activity_windows (t_sec, ignition_sec, fire_side, net_ojama_after, fire_chain_count) ===")
    for w in activity_windows:
        print(f"  fire_side={w.fire_side} ignition={w.ignition_sec:.2f} t_sec={w.t_sec:.2f}"
              f" net_ojama_after={w.net_ojama_after:.1f} chain_count={w.fire_chain_count:.1f}"
              f" receiver_baseline_ojama={w.receiver_baseline_ojama:.1f}")

    # --- 計装2: estimate_available_hands の現行値 vs 実効アニメ時間版 ---
    print("\n=== 計装2: k_hands (現行 CHAIN_ANIM_PER_STEP_SEC=0.4 版 vs 実測14.5秒/8連鎖版) ===")
    for chain in (5.0, 8.0):
        cur_delay = estimate_landing_delay_sec(chain)
        cur_hands_raw = int(cur_delay // iv.SEC_PER_HAND) + 1
        cur_hands = estimate_available_hands(chain)
        real_delay = REAL_ANIM_SEC_PER_8CHAIN * (chain / 8.0)  # 線形換算 (概算)
        real_hands_raw = int(real_delay // iv.SEC_PER_HAND) + 1
        real_hands = min(MAX_SUPPORTED_K_HANDS, real_hands_raw)
        print(f"  chain={chain:.0f}: 現行delay={cur_delay:.2f}s hands(素)={cur_hands_raw}"
              f" ->clamp後k_hands={cur_hands}  |  実測換算delay={real_delay:.2f}s"
              f" hands(素)={real_hands_raw} ->clamp後k_hands={real_hands}"
              f" (MAX_SUPPORTED_K_HANDS={MAX_SUPPORTED_K_HANDS})")

    # --- 両サイドのSTABLE時刻和集合 (実際の評価点) を対象窓で抽出 ---
    mask1, mask2 = cache.r1p.game_idx == GAME_IDX, cache.r2p.game_idx == GAME_IDX
    t1, g1 = cache.r1p.t_sec[mask1], cache.r1p.grids[mask1]
    t2, g2 = cache.r2p.t_sec[mask2], cache.r2p.grids[mask2]
    puyo_totals = np.array([int((g != 0).sum()) for g in g1], dtype=float)
    q_low, q_high = np.quantile(puyo_totals, 0.33), np.quantile(puyo_totals, 0.67)
    eval_times = np.union1d(t1, t2)
    eval_times = eval_times[(eval_times >= T_START) & (eval_times <= T_END)]

    print(f"\n=== 計装1/3/4: t={T_START}-{T_END}s の実評価点 ({len(eval_times)}点) 内訳表 ===")
    print(f"  (2Pの直近STABLE index2 が変化しない区間 = 2P側が長時間フリーズしていることの直接証拠)")
    header = (f"{'t_sec':>9} {'idx2':>5} {'1P色ぷよ':>7} {'1Pおじゃま':>9} {'2P色ぷよ':>7} {'2Pおじゃま':>9} {'room':>5}"
              f" {'pend1p':>7} {'pend2p':>7} {'attacker_chain':>14} {'k_hands':>7}"
              f" {'ef_raw':>7} {'P_prime':>8} {'clamp':>6} {'model_raw':>9} {'display':>8}")
    print(header)
    for t in eval_times:
        idx1 = int(np.searchsorted(t1, t, side="right")) - 1
        idx2 = int(np.searchsorted(t2, t, side="right")) - 1
        if idx1 < 0 or idx2 < 0:
            continue
        phase = "序" if puyo_totals[idx1] <= q_low else ("終" if puyo_totals[idx1] > q_high else "中")
        if phase not in models:
            continue
        b1, b2 = _board_from_grid(g1[idx1]), _board_from_grid(g2[idx2])
        active = _find_active_chain_window(chain_windows, float(t))
        if active is not None:
            if active.fire_side == "1P":
                b1 = active.board_after
            else:
                b2 = active.board_after

        f1, f2 = compute_board_only_features(b1, sim), compute_board_only_features(b2, sim)
        p1 = winprob_attacker(models, phase, f1, f2)
        model_raw = winprob_to_score100(p1)

        attack_from_1p, attack_from_2p, chain_from_1p, chain_from_2p = (
            _aggregate_known_pending_net_ojama(activity_windows, float(t), b1, b2))
        pending_on_1p, pending_on_2p = _net_pending_after_cancellation(attack_from_1p, attack_from_2p)

        if pending_on_1p > 0.0:
            threatened_board, pending, attacker_chain = b1, pending_on_1p, chain_from_2p
        elif pending_on_2p > 0.0:
            threatened_board, pending, attacker_chain = b2, pending_on_2p, chain_from_1p
        else:
            threatened_board, pending, attacker_chain = None, 0.0, 0.0

        if threatened_board is not None:
            k_hands = estimate_available_hands(int(round(attacker_chain)))
            ef_result = iv.expected_fire_power(threatened_board, k_levels=(k_hands,))
            ef_raw = float(ef_result.values[k_hands].raw)
            room = board_room(threatened_board)
            p_prime = pending - ef_raw
            clamp = "?"
            if p_prime <= 0.0:
                clamp = "no(P'<=0)"
                display = model_raw
            elif p_prime <= room:
                clamp = "no(room内)"
                display = model_raw
            else:
                clamp = "YES"
                if pending_on_1p > 0.0:
                    display = min(model_raw, 5.0)
                else:
                    display = max(model_raw, 95.0)
        else:
            k_hands, ef_raw, room, p_prime, clamp, display = 0, 0.0, 0.0, 0.0, "no(pend=0)", model_raw

        color_1p = _color_puyo_count(b1)
        ojama_1p = float(iv.board_ojama_count(b1).raw)
        color_2p = _color_puyo_count(b2)
        ojama_2p = float(iv.board_ojama_count(b2).raw)
        room_val = room if threatened_board is not None else board_room(b1)
        print(f"{t:9.2f} {idx2:5d} {color_1p:7d} {ojama_1p:9.1f} {color_2p:7d} {ojama_2p:9.1f} {room_val:5.0f}"
              f" {pending_on_1p:7.1f} {pending_on_2p:7.1f} {attacker_chain:14.1f} {k_hands:7d}"
              f" {ef_raw:7.1f} {p_prime:8.1f} {clamp:>6} {model_raw:9.2f} {display:8.2f}")


if __name__ == "__main__":
    main()
