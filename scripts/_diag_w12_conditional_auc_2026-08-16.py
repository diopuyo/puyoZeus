"""W12根治 P5: 予告帯限定の条件付きAUC+校正比較 (A vs B、2026-08-16 追加依頼)。

## 背景 (coordinator指摘、2026-08-16)
全体AUCでのA(旧47列)vs B(A+W12新5列)比較は判定材料にならない。理由:
`ojama_forecast_uncapped>0` の行は全体の約8.56%、`>=216` は約1.1%しかなく、
そこで大幅改善しても全体AUC (dAUC=+0.0005) には現れない。

本スクリプトは同一85本CSV・同一GroupKFold分割で構成A/Bのみを再学習し
(permutation importance等は計算しない、疎通済みの `_retrain85_w12_
2026-08-16.py` の A/B相当を軽量に再現)、以下を**行を絞った条件**で測る:

1. forecast>0 / forecast>=72 / forecast>=216 の各部分集合でのAUC (全体+位相別)
2. 予告bucket別の 平均予測勝率(A/B) vs 実勝率 の校正テーブル (`_diag_w12_
   quantify_2026-08-16.py` と同一bucket定義)
3. forecast>0 部分集合でのECE (A/B比較)

判定基準 (coordinator指定): 予告帯 (特に216+・終盤) で予測が実勝率に
近づいていれば「W12が効いた」。全体AUCが動かなくても構わない。
"""
from __future__ import annotations

import os

_CPU_LIMIT: str = "3"
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, _CPU_LIMIT)

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

DEFAULT_CSV: str = "data/verify/labeled_win_w12_85_2026-08-16/labeled_win_w12_85.csv"
OUT_DIR = Path("data/verify/retrain85_w12_2026-08-16")

N_FOLDS: int = 5
GBC_PARAMS: dict = {
    "max_iter": 300, "max_depth": 4, "learning_rate": 0.05,
    "min_samples_leaf": 20, "random_state": 42, "early_stopping": False,
}

META_COLS = frozenset({"video_id", "game_idx", "t_sec", "frame", "tsumo", "side", "won"})
SOURCE_COLS = frozenset({"all_clear_source", "ojama_source"})
W12_NEW_COLS: tuple[str, ...] = (
    "ojama_net_balance_uncapped", "ojama_forecast_uncapped",
    "ojama_forecast_log", "ojama_forecast_progress_interaction",
    "color_forecast_ratio_own",
)

# `_diag_w12_quantify_2026-08-16.py` と同一定義 (実勝率の既報値との直接比較用)。
FORECAST_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (-0.5, 0.5, "0"), (0.5, 11.5, "1-11"), (11.5, 29.5, "12-29"),
    (29.5, 71.5, "30-71"), (71.5, 143.5, "72-143"),
    (143.5, 215.5, "144-215"), (215.5, 1e9, "216+"),
)

CALIB_N_BINS: int = 10


def bucket_of(val: float, edges) -> str:
    for lo, hi, name in edges:
        if lo < val <= hi:
            return name
    return "?"


def resolve_feature_cols(df: pd.DataFrame, exclude_extra: frozenset = frozenset()) -> list[str]:
    exclude = META_COLS | SOURCE_COLS | exclude_extra
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def compute_match_progress(df: pd.DataFrame) -> np.ndarray:
    max_tsumo = df.groupby(["video_id", "game_idx", "side"])["tsumo"].transform("max")
    max_tsumo = max_tsumo.replace(0, np.nan)
    return (df["tsumo"] / max_tsumo).fillna(0.0).clip(0.0, 1.0).values


def run_oof(X, y, groups, n_folds):
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_folds)
    for i, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        m = HistGradientBoostingClassifier(**GBC_PARAMS)
        t0 = time.time()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        print(f"    fold {i+1}/{n_folds} 学習{time.time()-t0:.1f}秒")
    return oof


def auc_or_nan(y, p) -> tuple[float, int]:
    valid = ~np.isnan(p)
    y_v, p_v = y[valid], p[valid]
    if len(np.unique(y_v)) < 2:
        return float("nan"), int(valid.sum())
    return float(roc_auc_score(y_v, p_v)), int(valid.sum())


def conditional_auc_table(won, oof_a, oof_b, forecast, progress) -> pd.DataFrame:
    """予告条件別 (all/forecast>0/>=72/>=216) x 位相別 (全体/序盤/中盤/終盤) のAUC。"""
    phase_masks = {
        "全体": np.ones(len(won), dtype=bool),
        "序盤": progress <= 1 / 3,
        "中盤": (progress > 1 / 3) & (progress <= 2 / 3),
        "終盤": progress > 2 / 3,
    }
    cond_masks = {
        "forecast_all": np.ones(len(won), dtype=bool),
        "forecast>0": forecast > 0,
        "forecast>=72": forecast >= 72,
        "forecast>=216": forecast >= 216,
    }
    rows = []
    for cond_name, cmask in cond_masks.items():
        for phase_name, pmask in phase_masks.items():
            mask = cmask & pmask
            auc_a, n_a = auc_or_nan(won[mask], oof_a[mask])
            auc_b, n_b = auc_or_nan(won[mask], oof_b[mask])
            rows.append({
                "condition": cond_name, "phase": phase_name,
                "n": int(mask.sum()), "n_valid_a": n_a, "n_valid_b": n_b,
                "auc_A_旧47列": auc_a, "auc_B_旧47列+W12新5列": auc_b,
                "delta_auc_B_minus_A": (
                    auc_b - auc_a if not (np.isnan(auc_a) or np.isnan(auc_b)) else float("nan")
                ),
            })
    return pd.DataFrame(rows)


def calibration_bucket_table(won, oof_a, oof_b, forecast, progress) -> pd.DataFrame:
    """予告bucket別 (全体+位相別) の 平均予測勝率(A/B) vs 実勝率。W12の核心判定用。"""
    order = [b[2] for b in FORECAST_BUCKETS]
    work = pd.DataFrame({
        "won": won, "forecast": forecast, "pred_a": oof_a, "pred_b": oof_b,
        "progress": progress,
    })
    work["forecast_bucket"] = work["forecast"].apply(lambda v: bucket_of(v, FORECAST_BUCKETS))
    work["phase"] = pd.cut(
        work["progress"], bins=[-0.01, 1 / 3, 2 / 3, 1.01], labels=["序盤", "中盤", "終盤"],
    )

    overall = work.groupby("forecast_bucket").agg(
        actual_win_rate=("won", "mean"),
        pred_A_mean=("pred_a", "mean"),
        pred_B_mean=("pred_b", "mean"),
        n=("won", "size"),
    ).reindex(order)
    overall.insert(0, "phase", "全体")
    overall = overall.reset_index()

    by_phase = work.groupby(["phase", "forecast_bucket"], observed=True).agg(
        actual_win_rate=("won", "mean"),
        pred_A_mean=("pred_a", "mean"),
        pred_B_mean=("pred_b", "mean"),
        n=("won", "size"),
    ).reindex(pd.MultiIndex.from_product(
        [["序盤", "中盤", "終盤"], order], names=["phase", "forecast_bucket"],
    )).reset_index()

    combined = pd.concat([overall, by_phase], ignore_index=True)
    combined["gap_A"] = (combined["pred_A_mean"] - combined["actual_win_rate"]).abs()
    combined["gap_B"] = (combined["pred_B_mean"] - combined["actual_win_rate"]).abs()
    combined["gap_improved_B_vs_A"] = combined["gap_A"] - combined["gap_B"]
    return combined


def compute_ece(y, p, n_bins=CALIB_N_BINS) -> float:
    valid = ~np.isnan(p)
    y_v, p_v = y[valid], p[valid]
    if len(y_v) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_v, bins) - 1, 0, n_bins - 1)
    total = len(y_v)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        gap = abs(float(p_v[mask].mean()) - float(y_v[mask].mean()))
        ece += (n / total) * gap
    return float(ece)


def main() -> int:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("  W12 P5: forecast>0/>=72/>=216 条件付きAUC + 校正比較 (A vs B)")
    print("=" * 80)

    print("\n=== 1. データ読み込み ===")
    df = pd.read_csv(DEFAULT_CSV)
    n0 = len(df)
    df = df.dropna(subset=["won"]).reset_index(drop=True)
    df["won"] = df["won"].astype(int)
    print(f"  {n0}行 -> won欠損除外後 {len(df)}行 (動画数={df['video_id'].nunique()})")

    forecast = df["ojama_forecast_uncapped"].values
    n_f0 = int((forecast > 0).sum())
    n_f72 = int((forecast >= 72).sum())
    n_f216 = int((forecast >= 216).sum())
    print(f"  forecast>0: n={n_f0} ({100*n_f0/len(df):.2f}%)"
          f"  forecast>=72: n={n_f72} ({100*n_f72/len(df):.2f}%)"
          f"  forecast>=216: n={n_f216} ({100*n_f216/len(df):.2f}%)")

    progress = compute_match_progress(df)
    y = df["won"].values.astype(int)
    groups = df["video_id"].values

    cols_a = resolve_feature_cols(df, exclude_extra=frozenset(W12_NEW_COLS))
    cols_b = resolve_feature_cols(df)
    print(f"  構成A: {len(cols_a)}列 / 構成B: {len(cols_b)}列")

    print("\n=== 2. 構成A (旧47列) 学習 ===")
    X_a = df[cols_a].fillna(0.0).values.astype(np.float32)
    oof_a = run_oof(X_a, y, groups, N_FOLDS)

    print("\n=== 3. 構成B (旧47列+W12新5列) 学習 ===")
    X_b = df[cols_b].fillna(0.0).values.astype(np.float32)
    oof_b = run_oof(X_b, y, groups, N_FOLDS)

    print("\n=== 4. 条件付きAUC (forecast>0 / >=72 / >=216 x 全体/位相別) ===")
    cond_df = conditional_auc_table(y, oof_a, oof_b, forecast, progress)
    cond_df.to_csv(OUT_DIR / "conditional_auc_by_forecast.csv", index=False)
    print(cond_df.to_string(index=False))

    print("\n=== 5. 予告bucket別 校正テーブル (実勝率 vs 予測A/B平均) ===")
    calib_df = calibration_bucket_table(y, oof_a, oof_b, forecast, progress)
    calib_df.to_csv(OUT_DIR / "calibration_by_forecast_bucket.csv", index=False)
    print(calib_df.to_string(index=False))

    print("\n=== 6. ECE (forecast>0 部分集合、A vs B) ===")
    mask_f0 = forecast > 0
    ece_a = compute_ece(y[mask_f0], oof_a[mask_f0])
    ece_b = compute_ece(y[mask_f0], oof_b[mask_f0])
    print(f"  ECE (forecast>0, n={int(mask_f0.sum())}): A={ece_a:.4f}  B={ece_b:.4f}"
          f"  (改善={ece_a - ece_b:+.4f})")

    summary = {
        "n_rows": len(df), "n_videos": int(df["video_id"].nunique()),
        "n_forecast_gt0": n_f0, "n_forecast_ge72": n_f72, "n_forecast_ge216": n_f216,
        "pct_forecast_gt0": 100 * n_f0 / len(df),
        "pct_forecast_ge72": 100 * n_f72 / len(df),
        "pct_forecast_ge216": 100 * n_f216 / len(df),
        "ece_forecast_gt0_A": ece_a, "ece_forecast_gt0_B": ece_b,
        "ece_improvement_B_vs_A": ece_a - ece_b,
        "total_wall_seconds": time.time() - t_start,
    }
    with open(OUT_DIR / "conditional_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"  完了。総所要時間: {(time.time()-t_start)/60:.1f}分")
    print("P5_CONDITIONAL_AUC_DONE")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
