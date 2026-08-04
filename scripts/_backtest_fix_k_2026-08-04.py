"""K-2 (2026-08-04 main発注): 修正K の全域検証。

発火イベント時点の実践値 (winprob_after、K適用後の評価盤面ベース) vs
最終勝敗 (won、fire_side視点) の AUC を、K適用前後・位相別・動画クラスタ
考慮で比較する。既存資産のみ再利用:
- build_event_activity_windows / _aggregate_known_pending_net_ojama /
  _net_pending_after_cancellation / _apply_pending_ojama_virtual_landing
  (本番 _build_stable_timeline と全く同じ関数、再実装しない)
- reconstruct_event_board_pair (盤面復元)
- train_winprob_models (実モデル、fakeでない)

活動窓 (activity_windows) は「その試合(video_id, game_idx)の全発火イベント」
から構築する (サンプリングで抜けた行があっても台帳の完全性を保つため、
台帳構築はdf_full、AUC評価はサンプル済みdf_sampleで行う)。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.compute_exchange_delta_winprob import (
    DEFAULT_LABELED_WIN_CSV,
    _aggregate_known_pending_net_ojama,
    _apply_pending_ojama_virtual_landing,
    _load_video_npz,
    _net_pending_after_cancellation,
    build_event_activity_windows,
    compute_board_only_features,
    reconstruct_event_board_pair,
    train_winprob_models,
    winprob_attacker,
)
from src.chain import ChainSimulator

AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv")
REGEN_NPZ_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
OUT_DIR = Path("data/verify/fix_k_backtest_2026-08-04")
SAMPLE_FRAC = 0.30  # 実行時間対策 (video単位層化サンプリング、J-1と同じ方針)


def _cluster_bootstrap_auc_ci(
    p: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = 500, seed: int = 42,
) -> tuple[float, float]:
    """動画クラスタ単位のブートストラップでAUCの95%CIを返す。"""
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, p[idx]))
    if len(aucs) < 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def compute_before_after_winprob(
    df_sample: pd.DataFrame, df_full: pd.DataFrame, models: dict, sim: ChainSimulator,
) -> pd.DataFrame:
    """サンプル行ごとに、K適用前(生盤面)/K適用後(pending仮想着弾済み盤面)の
    fire_side視点勝率(p_before_k/p_after_k)を計算する。台帳は各(video_id,
    game_idx)の全イベント(df_full)から構築する。10分おき進捗ログ。
    """
    n = len(df_sample)
    p_no_k = np.full(n, np.nan)
    p_with_k = np.full(n, np.nan)
    t_start = time.time()
    done = 0
    cache_by_video: dict[str, object] = {}
    windows_by_game: dict[tuple[str, int], list] = {}
    for video_id, sub in df_sample.groupby("video_id"):
        stem = video_id[len("video_"):] if video_id.startswith("video_") else video_id
        if stem not in cache_by_video:
            cache_by_video[stem] = _load_video_npz(stem, REGEN_NPZ_DIR)
        cache = cache_by_video[stem]
        if cache is None:
            continue
        for idx, row in sub.iterrows():
            game_idx, t_sec, fire_side, phase = (
                int(row["game_idx"]), float(row["t_sec"]), str(row["fire_side"]), str(row["phase"]))
            if phase not in models:
                continue
            key = (video_id, game_idx)
            if key not in windows_by_game:
                game_events = df_full.loc[
                    (df_full["video_id"] == video_id) & (df_full["game_idx"] == game_idx)]
                windows_by_game[key] = build_event_activity_windows(game_events, cache)
            windows = windows_by_game[key]

            pair = reconstruct_event_board_pair(cache, game_idx, t_sec, fire_side)
            if pair is None:
                continue
            fire_board, opp_board = pair
            # fire_side視点に統一 (1P/2Pどちらが発火側でも「発火側=board_1p側」
            # として評価する、winprob_attackerは常に第1引数側の勝率を返す)。
            b_attacker, b_opp = fire_board, opp_board

            feats_a_raw = compute_board_only_features(b_attacker, sim)
            feats_o_raw = compute_board_only_features(b_opp, sim)
            i = df_sample.index.get_loc(idx)
            p_no_k[i] = winprob_attacker(models, phase, feats_a_raw, feats_o_raw)

            # 修正K: 台帳(activity_windows)からこの時点のpendingを計算し、
            # attacker/opp それぞれの盤面に仮想着弾させてから再評価する。
            # ledgerは1P/2P基準のため、fire_side="2P"の場合は attacker=board_2p
            # として渡す必要がある (_aggregate_known_pending_net_ojama の
            # 引数順は (board_1p, board_2p) 固定)。
            if fire_side == "1P":
                board_1p, board_2p = b_attacker, b_opp
            else:
                board_1p, board_2p = b_opp, b_attacker
            attack_1p, attack_2p, _c1, _c2 = _aggregate_known_pending_net_ojama(
                windows, t_sec, board_1p, board_2p)
            pending_1p, pending_2p = _net_pending_after_cancellation(attack_1p, attack_2p)
            board_1p_k = _apply_pending_ojama_virtual_landing(board_1p, pending_1p, sim)
            board_2p_k = _apply_pending_ojama_virtual_landing(board_2p, pending_2p, sim)
            if fire_side == "1P":
                b_attacker_k, b_opp_k = board_1p_k, board_2p_k
            else:
                b_attacker_k, b_opp_k = board_2p_k, board_1p_k
            feats_a_k = compute_board_only_features(b_attacker_k, sim)
            feats_o_k = compute_board_only_features(b_opp_k, sim)
            p_with_k[i] = winprob_attacker(models, phase, feats_a_k, feats_o_k)

            done += 1
            if done % 1000 == 0:
                elapsed_min = (time.time() - t_start) / 60.0
                remaining_min = elapsed_min / done * (n - done)
                print(f"[PROGRESS] {done}/{n} 完了 (経過{elapsed_min:.1f}分、残り約{remaining_min:.1f}分)",
                      flush=True)
    out = df_sample.copy()
    out["p_no_k"] = p_no_k
    out["p_with_k"] = p_with_k
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== モデル学習 (board-only, 実モデル) ===")
    models = train_winprob_models(DEFAULT_LABELED_WIN_CSV)
    sim = ChainSimulator()

    df_full = pd.read_csv(AUG_CSV)
    rng = np.random.RandomState(42)
    parts = [g.sample(frac=SAMPLE_FRAC, random_state=rng) for _, g in df_full.groupby("video_id")]
    df_sample = pd.concat(parts).sort_index().reset_index(drop=True)
    print(f"=== 入力: 全{len(df_full)}行のうちvideo層化サンプリング{SAMPLE_FRAC:.0%}"
          f" = {len(df_sample)}行 (台帳は各試合の全イベントdf_fullから構築) ===")

    out = compute_before_after_winprob(df_sample, df_full, models, sim)
    valid = out["p_no_k"].notna() & out["p_with_k"].notna()
    out_v = out.loc[valid].reset_index(drop=True)
    n_ok = len(out_v)
    print(f"\n盤面復元・評価成功: {n_ok}/{len(df_sample)} ({n_ok / len(df_sample):.1%})")

    y = out_v["won"].astype(int).values
    groups = out_v["video_id"].values
    print("\n=== AUC (fire_side視点勝率 vs 最終勝敗、K適用前後比較) ===")
    rows = []
    for phase in ("序", "中", "終", "全体"):
        if phase == "全体":
            mask = np.ones(len(out_v), dtype=bool)
        else:
            mask = (out_v["phase"] == phase).values
        n = int(mask.sum())
        if n < 30 or len(np.unique(y[mask])) < 2:
            print(f"  {phase}: n={n} データ不足 -> skip")
            continue
        auc_before = roc_auc_score(y[mask], out_v.loc[mask, "p_no_k"].values)
        auc_after = roc_auc_score(y[mask], out_v.loc[mask, "p_with_k"].values)
        lo_b, hi_b = _cluster_bootstrap_auc_ci(
            out_v.loc[mask, "p_no_k"].values, y[mask], groups[mask])
        lo_a, hi_a = _cluster_bootstrap_auc_ci(
            out_v.loc[mask, "p_with_k"].values, y[mask], groups[mask])
        print(f"  {phase}: n={n}  K適用前 AUC={auc_before:.4f} [{lo_b:.4f},{hi_b:.4f}]"
              f"  K適用後 AUC={auc_after:.4f} [{lo_a:.4f},{hi_a:.4f}]"
              f"  Δ={auc_after - auc_before:+.4f}")
        rows.append({"位相": phase, "n": n, "AUC_K前": auc_before, "CI下_K前": lo_b, "CI上_K前": hi_b,
                     "AUC_K後": auc_after, "CI下_K後": lo_a, "CI上_K後": hi_a,
                     "delta": auc_after - auc_before})
    pd.DataFrame(rows).to_csv(OUT_DIR / "auc_before_after_by_phase.csv", index=False)
    out_v.to_csv(OUT_DIR / "raw_predictions.csv", index=False)
    print(f"\n[保存] {OUT_DIR}")


if __name__ == "__main__":
    main()
