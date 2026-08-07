"""K拡張MC (scripts.mc_counter_estimator) の全域バックテスト (パイロット版、2026-08-04)。

scripts/_backtest_fix_h2kai_2026-08-04.py と同じ枠組み (既存資産の再利用のみ:
reconstruct_event_board_pair、board_room、sim_expected_counter_ojama列、
immediate_fire_power、severity_flag/precision_recall/calibration_by_severity_bin
は再実装せずそのまま import する) で、修正H (immediate/expected の max) と
「修正H+MC拡張 (p75チャネル)」の precision/recall/F1 を「回避不能死」ラベルで
比較する。

⚠️ 正直な注記 (コスト、本タスク検収項目): MC (n_rollouts=200既定) は
1イベントあたり実測1-3.3秒 (scripts/mc_counter_estimator.py 呼び出し側の
case (a)/(b) 実測)。18,057件全域×200本は現実的な時間で終わらないため、
本パイロットは (1) n_rollouts を落とす、(2) video層化サンプルを縮小する、
の2つで実行時間を抑える (数値はコマンドライン引数、既定値は本ファイル
下部のCLI既定を参照)。全域拡張は本パイロットの結果を見てから判断する
(過学習ガード feedback_overfitting_awareness_2026-08-04)。
"""
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

import src.indicators_v2 as iv

# ファイル名に日付ハイフンを含むため通常の import 文では読めない (project既存の
# scripts/_measure_freeze_threshold_2026-08-03.py と同じ回避策、再実装しない)。
_h2kai_mod = importlib.import_module("scripts._backtest_fix_h2kai_2026-08-04")
calibration_by_severity_bin = _h2kai_mod.calibration_by_severity_bin
precision_recall = _h2kai_mod.precision_recall
severity_flag = _h2kai_mod.severity_flag

from scripts.compute_exchange_delta_winprob import (
    OJAMA_MAX_DROP_PER_TURN,
    _load_video_npz,
    reconstruct_event_board_pair,
)
from scripts.measure_exchange_effectiveness import estimate_landing_delay_sec
from scripts.mc_counter_estimator import estimate_counter_distribution
from scripts.visualize_advantage_overlay import board_room

AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv")
REGEN_NPZ_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
OUT_DIR = Path("data/verify/mc_counter_backtest_2026-08-04")


def _opp_next_pair_at(cache, receiver_side: str, game_idx: int, t_sec: float):
    """reconstruct_event_board_pair と同じ「時刻最近傍」規則でopp側の
    next_pair/dnext_pairを取得する (scripts/_backtest_fix_h2kai_2026-08-04.py
    の同名関数と同一実装、再実装ではなく重複回避のためこのファイル内にも
    薄くコピー: importで日付付きファイル名の import 制約を避けるため)。
    """
    rec = cache.r2p if receiver_side == "2P" else cache.r1p
    mask = rec.game_idx == game_idx
    t_arr = rec.t_sec[mask]
    if len(t_arr) == 0:
        return (-1, -1), (-1, -1)
    nearest = int(np.argmin(np.abs(t_arr - t_sec)))
    n1a, n1b = rec.next1_a[mask], rec.next1_b[mask]
    d1a, d1b = rec.dnext_a[mask], rec.dnext_b[mask]
    return (int(n1a[nearest]), int(n1b[nearest])), (int(d1a[nearest]), int(d1b[nearest]))


def compute_room_immediate_and_mc(
    df: pd.DataFrame, n_rollouts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """行ごとに room / immediate_fire_power / MC(p75) を npz から計算する
    (video単位でキャッシュ再利用)。10分おき進捗ログ + 合計時間を返す。
    """
    n = len(df)
    room = np.full(n, np.nan)
    immediate = np.full(n, np.nan)
    mc_p75 = np.full(n, np.nan)
    t_start = time.time()
    done = 0
    for video_id, sub in df.groupby("video_id"):
        stem = video_id[len("video_"):] if video_id.startswith("video_") else video_id
        cache = _load_video_npz(stem, REGEN_NPZ_DIR)
        if cache is None:
            continue
        for idx, row in sub.iterrows():
            game_idx, t_sec, fire_side = int(row["game_idx"]), float(row["t_sec"]), str(row["fire_side"])
            pair = reconstruct_event_board_pair(cache, game_idx, t_sec, fire_side)
            if pair is None:
                continue
            _fire_board, opp_board = pair
            i = df.index.get_loc(idx)
            room[i] = board_room(opp_board)
            immediate[i] = iv.immediate_fire_power(opp_board).raw
            receiver_side = "2P" if fire_side == "1P" else "1P"
            next_pair, dnext_pair = _opp_next_pair_at(cache, receiver_side, game_idx, t_sec)
            known_pairs = tuple(p for p in (next_pair, dnext_pair) if all(c >= 1 for c in p))
            time_budget = estimate_landing_delay_sec(float(row["approx_fire_chains"]))
            dist = estimate_counter_distribution(
                opp_board, time_budget, known_pairs=known_pairs, n_rollouts=n_rollouts,
            )
            mc_p75[i] = dist.p75
            done += 1
            if done % 200 == 0:
                elapsed_min = (time.time() - t_start) / 60.0
                remaining_min = elapsed_min / done * (n - done)
                print(f"[PROGRESS] {done}/{n} 完了 (経過{elapsed_min:.1f}分、残り約{remaining_min:.1f}分)", flush=True)
    total_sec = time.time() - t_start
    return room, immediate, mc_p75, total_sec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=300, help="video層化サンプルの総行数目安 (既定300、パイロット)")
    parser.add_argument("--n-rollouts", type=int, default=30, help="MCロールアウト本数 (既定30、本番既定200より軽量)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_full = pd.read_csv(AUG_CSV)
    frac = min(1.0, args.sample_n / len(df_full))
    rng = np.random.RandomState(args.seed)
    parts = [g.sample(frac=frac, random_state=rng) for _, g in df_full.groupby("video_id")]
    df = pd.concat(parts).sort_index().reset_index(drop=True)
    print(f"=== パイロット入力: 全{len(df_full)}行のうちvideo層化サンプル frac={frac:.4f}"
          f" = {len(df)}行、n_rollouts={args.n_rollouts} ===")

    room, immediate, mc_p75, total_sec = compute_room_immediate_and_mc(df, args.n_rollouts)
    n_ok = int(np.sum(~np.isnan(room)))
    print(f"盤面復元成功: {n_ok}/{len(df)} ({n_ok / len(df):.1%})")
    print(f"[コスト実測] 合計 {total_sec:.1f}秒 / 成功{n_ok}件 = {total_sec / max(1, n_ok):.3f}秒/件"
          f" (n_rollouts={args.n_rollouts})")
    full_scale_hr = total_sec / max(1, n_ok) * len(df_full) / 3600.0
    print(f"[コスト外挿] 全域{len(df_full)}件 (同n_rollouts) なら概算 {full_scale_hr:.1f}時間")

    valid = ~np.isnan(room)
    df_v = df.loc[valid].reset_index(drop=True)
    room_v, immediate_v, mc_v = room[valid], immediate[valid], mc_p75[valid]
    pending = df_v["net_ojama_after"].astype(float).values
    counter_h_v = np.maximum(df_v["sim_expected_counter_ojama"].astype(float).values, immediate_v)
    counter_mc_v = np.maximum(counter_h_v, mc_v)

    truth_unavoidable = (
        (df_v["opp_buried"].astype(int).values == 1) & (df_v["returned"].astype(int).values == 0)
    ).astype(int)
    n_unavoidable = int(truth_unavoidable.sum())
    print(f"回避不能死の件数: {n_unavoidable}/{len(df_v)} ({n_unavoidable / len(df_v):.1%})")

    flag_h, sev_h = severity_flag(pending, counter_h_v, room_v)
    flag_mc, sev_mc = severity_flag(pending, counter_mc_v, room_v)

    pr_h = precision_recall(flag_h, truth_unavoidable)
    pr_mc = precision_recall(flag_mc, truth_unavoidable)
    print(f"\n=== [主指標] precision/recall/F1 (回避不能死ラベル, n={len(df_v)}) ===")
    print(f"  修正H       : {pr_h}")
    print(f"  修正H+MC拡張: {pr_mc}")
    print(f"\n  [検収] precision>=H({pr_h['precision']:.4f})? {pr_mc['precision'] >= pr_h['precision']}"
          f" / recall>=H({pr_h['recall']:.4f})? {pr_mc['recall'] >= pr_h['recall']}")

    n_helped = int((counter_mc_v > counter_h_v).sum())
    print(f"\n  MCがcounterを押し上げた行: {n_helped}/{len(df_v)} ({n_helped / len(df_v):.1%})")

    summary = pd.DataFrame([{"版": "修正H", **pr_h}, {"版": "修正H+MC拡張", **pr_mc}])
    summary.to_csv(OUT_DIR / "precision_recall_summary_pilot.csv", index=False)
    calib_h = calibration_by_severity_bin(sev_h, truth_unavoidable)
    calib_mc = calibration_by_severity_bin(sev_mc, truth_unavoidable)
    calib_h.to_csv(OUT_DIR / "calibration_h_pilot.csv", index=False)
    calib_mc.to_csv(OUT_DIR / "calibration_mc_pilot.csv", index=False)
    print(f"\n[保存] {OUT_DIR}")


if __name__ == "__main__":
    main()
