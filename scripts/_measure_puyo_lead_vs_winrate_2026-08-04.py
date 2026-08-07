"""J-1 (2026-08-04 main発注): ぷよ量差 vs 実勝率の層別測定 (モデル非介入)。

labeled_win_combined66 の中盤/序盤/終盤ペア (対称化前の生ペア、
train_winprob_models と同じ pair_sides_for_win + 位相3分位を再利用) で、
board_color_puyo_total_diff (色ぷよ量差) をバケット化し、各バケットの
実勝率 (won_1p率) と n・動画クラスタ考慮の95%CI (クラスタブートストラップ)
を算出する。おじゃま数差 (board_ojama_count_diff) での層別版も出し、
「量リードが色ぷよ由来かおじゃま押し付けられ由来か」の交絡を確認する。

このスクリプトはモデルを一切学習・変更しない (main指示「モデル変更の前に
実測」、過学習ガード)。既存の pair_sides_for_win / _assign_phase_by_puyo_
tertile をそのまま再利用し、再実装しない。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compute_exchange_delta_winprob import (
    DEFAULT_LABELED_WIN_CSV,
    PAIR_MAX_TDIFF_SEC,
    _assign_phase_by_puyo_tertile,
)
from scripts.model_indicator_win import load_labeled_csv, pair_sides_for_win

# 量差バケットの境界 (main指定、色ぷよ個数差・おじゃま数差の両方に共用)。
BUCKET_EDGES: tuple[float, ...] = (-np.inf, -15.0, -8.0, -3.0, 3.0, 8.0, 15.0, np.inf)
BUCKET_LABELS: tuple[str, ...] = (
    "<=-15", "-15~-8", "-8~-3", "-3~+3(ほぼ同量)", "+3~+8", "+8~+15", ">+15",
)

N_BOOTSTRAP: int = 2000
BOOTSTRAP_SEED: int = 42


def _cluster_bootstrap_ci(
    values: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """動画クラスタ (video_id) 単位のブートストラップで実勝率の95%CIを返す。

    通常の二項分布CIは同一動画内の行の相関を無視して過信 (CI幅が狭すぎる)
    になるため、動画単位でリサンプリングするクラスタブートストラップを使う
    (feedback_stratify_before_pooling の考え方: 動画は層別軸の一つ)。
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        means[b] = float(values[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _bucketize(diff: np.ndarray) -> pd.Categorical:
    return pd.cut(diff, bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=True)


def _summarize_by_bucket(
    df: pd.DataFrame, bucket_col: str, other_diff_col: "str | None" = None,
) -> pd.DataFrame:
    """バケット別に n・実勝率(won_1p率)・95%CI・(交絡チェック用)他方の量差平均を出す。"""
    rows = []
    for label in BUCKET_LABELS:
        sub = df.loc[df[bucket_col] == label]
        n = len(sub)
        if n == 0:
            rows.append({"バケット": label, "n": 0, "実勝率": float("nan"),
                         "CI下": float("nan"), "CI上": float("nan"),
                         "他方量差平均(交絡チェック)": float("nan")})
            continue
        won = sub["won_1p"].astype(int).values
        groups = sub["video_id_1p"].values
        win_rate = float(won.mean())
        lo, hi = _cluster_bootstrap_ci(won.astype(float), groups)
        other_mean = float(sub[other_diff_col].mean()) if other_diff_col else float("nan")
        rows.append({"バケット": label, "n": n, "実勝率": win_rate,
                     "CI下": lo, "CI上": hi, "他方量差平均(交絡チェック)": other_mean})
    return pd.DataFrame(rows)


def main() -> None:
    print(f"[J-1] labeled_win読込: {DEFAULT_LABELED_WIN_CSV}")
    df = load_labeled_csv(str(DEFAULT_LABELED_WIN_CSV))
    paired = pair_sides_for_win(df, PAIR_MAX_TDIFF_SEC)
    print(f"[J-1] ペア成立 (対称化前の生ペア): {len(paired)}行")

    # 位相判定: train_winprob_models と同じ「1P+2P合計の3分位」(対称化前でも
    # phase_metric自体は不変量のため境界値は学習時と一致する)。
    phase_metric = (paired["board_puyo_total_1p"].astype(float).values
                    + paired["board_puyo_total_2p"].astype(float).values)
    phase_labels, q_low, q_high = _assign_phase_by_puyo_tertile(phase_metric)
    paired = paired.copy()
    paired["phase"] = phase_labels
    print(f"[J-1] 位相境界: 序<={q_low:.3f} 終>{q_high:.3f}")

    # 量差 (生カウント、_raw列を使う。0-1正規化スコアでなく実個数差で
    # バケット化するのがmain指定の意図に合う)。
    paired["color_puyo_diff_raw"] = (
        paired["board_color_puyo_total_raw_1p"].astype(float)
        - paired["board_color_puyo_total_raw_2p"].astype(float)
    )
    paired["ojama_count_diff_raw"] = (
        paired["board_ojama_count_raw_1p"].astype(float)
        - paired["board_ojama_count_raw_2p"].astype(float)
    )
    paired["color_bucket"] = _bucketize(paired["color_puyo_diff_raw"].values)
    paired["ojama_bucket"] = _bucketize(paired["ojama_count_diff_raw"].values)

    for phase in ("序", "中", "終"):
        sub = paired.loc[paired["phase"] == phase]
        print(f"\n=== 位相={phase} (n={len(sub)}) ===")
        print("--- 色ぷよ量差 (1P-2P) バケット別 実勝率 (won_1p率) ---")
        print(_summarize_by_bucket(sub, "color_bucket", other_diff_col="ojama_count_diff_raw")
              .to_string(index=False))
        print("--- おじゃま数差 (1P-2P) バケット別 実勝率 (交絡チェック用) ---")
        print(_summarize_by_bucket(sub, "ojama_bucket", other_diff_col="color_puyo_diff_raw")
              .to_string(index=False))

    out_dir = Path("data/verify/puyo_lead_vs_winrate_2026-08-04")
    out_dir.mkdir(parents=True, exist_ok=True)
    for phase in ("序", "中", "終"):
        sub = paired.loc[paired["phase"] == phase]
        _summarize_by_bucket(sub, "color_bucket", "ojama_count_diff_raw").to_csv(
            out_dir / f"color_bucket_{phase}.csv", index=False)
        _summarize_by_bucket(sub, "ojama_bucket", "color_puyo_diff_raw").to_csv(
            out_dir / f"ojama_bucket_{phase}.csv", index=False)
    print(f"\n[保存] {out_dir}")


if __name__ == "__main__":
    main()
