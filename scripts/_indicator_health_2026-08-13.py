# -*- coding: utf-8 -*-
"""死に指標 実測診断スクリプト (2026-08-13、user依頼)。

data/indicators_v2/boards_lean_phase_l_2026-08-11/ の npz (収集走行中のため
mtime が古い安定分のみ) から build_labeled_win_from_npz (efdcd22以降) の
light profile で labeled_win CSV を再構築し、以下を診断する:
  1. 無情報判定 (分散ゼロ/欠損率/全体・位相別単変量AUC)
  2. 冗長判定 (|Spearman|>0.85 ペア、>0.95 クラスタ)
  3. 交互作用列 (color_ojama_ratio_own 等) の単変量/層別AUC
  4. 多変量 (HistGBC, 学習側 scripts/model_indicator_win.py と同型パラメータ)
     permutation importance
  5. 動画別中央値との乖離、8/12 63本測定 (data/verify/npz_light_smoke_2026-08-12/)
     との比較
別途、64本目以降の npz にのみ存在する真値列 (tsumo_count/all_clear_pending/
ojama_net_balance/ojama_forecast) の分布・欠損率・妥当性を集計する。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_labeled_win_from_npz import (
    _final_fieldnames,
    _resolve_indicator_registry,
    convert_one_npz,
)

NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
OUT_DIR = Path("data/verify/indicator_health_2026-08-13")
OUT_CSV = OUT_DIR / "labeled_win_light_current.csv"

STABLE_CUTOFF_SEC = 60 * 60

META_COLS = frozenset(
    ["video_id", "game_idx", "t_sec", "frame", "tsumo", "side", "won"],
)

GBC_PARAMS: dict = {
    "max_iter": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "random_state": 42,
    "early_stopping": False,
}
N_FOLDS = 5
PERM_N_REPEATS = 20
PERM_RANDOM_STATE = 42

CORR_THRESHOLD_PAIR = 0.85
CORR_THRESHOLD_CLUSTER = 0.95

PHASE_LABELS = ["early", "mid", "late"]

BASELINE_EVIDENCE_CSV = Path(
    "data/verify/npz_light_smoke_2026-08-12/_reorg_evidence_summary_2026-08-12.csv",
)
BASELINE_DIFF_CSV = Path(
    "data/verify/npz_light_smoke_2026-08-12/_reorg_diff_comparison_2026-08-12.csv",
)

TRUE_VALUE_KEYS = ("tsumo_count", "all_clear_pending", "ojama_net_balance", "ojama_forecast")


def select_stable_npz() -> tuple[list[Path], list[Path]]:
    now = time.time()
    files = sorted(NPZ_DIR.glob("*.npz"))
    stable, excluded = [], []
    for p in files:
        age = now - p.stat().st_mtime
        (stable if age >= STABLE_CUTOFF_SEC else excluded).append(p)
    print(f"[info] npz total={len(files)} stable={len(stable)} excluded(new)={len(excluded)}")
    for p in excluded:
        age_min = (now - p.stat().st_mtime) / 60.0
        print(f"  excluded (writing suspect): {p.name} (age {age_min:.1f}min)")
    return stable, excluded


def build_current_csv(stable_files: list[Path]) -> pd.DataFrame:
    if OUT_CSV.exists():
        print(f"[info] reuse existing csv: {OUT_CSV}")
        return pd.read_csv(OUT_CSV)
    registry = _resolve_indicator_registry("light")
    all_rows: list[dict] = []
    t0 = time.time()
    for i, p in enumerate(stable_files):
        rows = convert_one_npz(p, registry)
        all_rows.extend(rows)
        print(
            f"[{i + 1}/{len(stable_files)}] {p.name}: {len(rows)} rows "
            f"(total {len(all_rows)}, {time.time() - t0:.1f}s)",
        )
    fieldnames = _final_fieldnames("light")
    df = pd.DataFrame(all_rows)
    out_cols = [c for c in fieldnames if c in df.columns]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, columns=out_cols)
    print(f"[done] {len(df)} rows -> {OUT_CSV} ({time.time() - t0:.1f}s)")
    return df[out_cols]


def compute_progress(df: pd.DataFrame) -> pd.Series:
    grp = df.groupby(["video_id", "game_idx"])["t_sec"]
    t_min = grp.transform("min")
    t_max = grp.transform("max")
    rng = (t_max - t_min).replace(0, np.nan)
    progress = (df["t_sec"] - t_min) / rng
    return progress.fillna(0.5).clip(0.0, 1.0)


def add_phase(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["progress"] = compute_progress(df)
    df["phase"] = pd.cut(
        df["progress"], bins=[-0.001, 1 / 3, 2 / 3, 1.001], labels=PHASE_LABELS,
    )
    return df


def safe_auc(y: np.ndarray, x: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return None
    mask = ~np.isnan(x)
    if mask.sum() < 10:
        return None
    xm = x[mask]
    if np.nanstd(xm) == 0 or len(np.unique(y[mask])) < 2:
        return None
    try:
        return float(roc_auc_score(y[mask], xm))
    except ValueError:
        return None


def univariate_report(df: pd.DataFrame, indicator_cols: list[str]) -> pd.DataFrame:
    y_all = df["won"].to_numpy()
    rows_out = []
    for col in indicator_cols:
        x_all = df[col].to_numpy(dtype=float)
        missing_pct = float(df[col].isna().mean()) * 100.0
        std_all = float(np.nanstd(x_all))
        zero_var = std_all == 0.0

        raw_auc = safe_auc(y_all, x_all)
        if raw_auc is None:
            direction, overall_auc = "N/A", None
        else:
            direction = "+" if raw_auc >= 0.5 else "-"
            overall_auc = raw_auc if raw_auc >= 0.5 else 1.0 - raw_auc

        phase_aucs = {}
        phase_flip = {}
        for ph in PHASE_LABELS:
            sub = df[df["phase"] == ph]
            a = safe_auc(sub["won"].to_numpy(), sub[col].to_numpy(dtype=float))
            if a is None:
                phase_aucs[ph], phase_flip[ph] = None, False
            else:
                phase_aucs[ph] = a if direction == "+" else (1.0 - a)
                phase_flip[ph] = (a >= 0.5) != (direction == "+")

        video_aucs = []
        for _vid, sub in df.groupby("video_id"):
            a = safe_auc(sub["won"].to_numpy(), sub[col].to_numpy(dtype=float))
            if a is None:
                continue
            video_aucs.append(a if direction == "+" else (1.0 - a))
        video_median = float(np.median(video_aucs)) if video_aucs else None

        all_phase_flat = all(
            (phase_aucs[ph] is not None and abs(phase_aucs[ph] - 0.5) <= 0.01)
            for ph in PHASE_LABELS
        )
        overall_flat = overall_auc is not None and abs(overall_auc - 0.5) <= 0.01
        is_uninformative = zero_var or missing_pct >= 99.0 or (overall_flat and all_phase_flat)

        rows_out.append({
            "col": col, "direction": direction, "overall_auc": overall_auc,
            "early_auc": phase_aucs["early"], "early_flip": phase_flip["early"],
            "mid_auc": phase_aucs["mid"], "mid_flip": phase_flip["mid"],
            "late_auc": phase_aucs["late"], "late_flip": phase_flip["late"],
            "video_median_auc": video_median, "video_n": len(video_aucs),
            "missing_pct": missing_pct, "zero_var": zero_var, "std": std_all,
            "is_uninformative": is_uninformative,
        })
    return pd.DataFrame(rows_out).sort_values("overall_auc", ascending=False, na_position="last")


def redundancy_report(df, indicator_cols, uni):
    corr = df[indicator_cols].corr(method="spearman")
    cols = corr.columns.tolist()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.isna(r):
                continue
            if abs(r) > CORR_THRESHOLD_PAIR:
                pairs.append({"col_a": cols[i], "col_b": cols[j], "rho": round(float(r), 4)})
    if pairs:
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.reindex(pairs_df["rho"].abs().sort_values(ascending=False).index)
    else:
        pairs_df = pd.DataFrame(columns=["col_a", "col_b", "rho"])

    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.isna(r):
                continue
            if abs(r) > CORR_THRESHOLD_CLUSTER:
                union(cols[i], cols[j])

    clusters = {}
    for c in cols:
        clusters.setdefault(find(c), []).append(c)
    cluster_list = [members for members in clusters.values() if len(members) > 1]
    return pairs_df, cluster_list


def recommend_cluster_representative(cluster, uni):
    sub = uni[uni["col"].isin(cluster)].copy()
    sub["video_median_auc"] = sub["video_median_auc"].fillna(-1.0)
    return sub.sort_values("video_median_auc", ascending=False).iloc[0]["col"]


def multivariate_report(df, indicator_cols):
    X = df[indicator_cols].to_numpy(dtype=float)
    y = df["won"].to_numpy(dtype=int)
    groups = df["video_id"].to_numpy()

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.full(len(y), np.nan)
    importances_per_fold = []
    t0 = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        model = HistGradientBoostingClassifier(**GBC_PARAMS)
        model.fit(X_tr, y_tr)
        oof_proba[test_idx] = model.predict_proba(X_te)[:, 1]
        perm = permutation_importance(
            model, X_te, y_te, n_repeats=PERM_N_REPEATS,
            random_state=PERM_RANDOM_STATE, scoring="roc_auc", n_jobs=1,
        )
        importances_per_fold.append(perm.importances_mean)
        print(
            f"    fold {fold_idx + 1}/{N_FOLDS}: train={len(train_idx)} test={len(test_idx)} "
            f"({time.time() - t0:.1f}s elapsed)",
        )

    valid = ~np.isnan(oof_proba)
    oof_auc = float(roc_auc_score(y[valid], oof_proba[valid]))
    imp_arr = np.array(importances_per_fold)
    imp_df = pd.DataFrame({
        "col": indicator_cols,
        "perm_importance_mean": imp_arr.mean(axis=0),
        "perm_importance_std": imp_arr.std(axis=0),
    }).sort_values("perm_importance_mean", ascending=False)
    return oof_auc, imp_df


def true_value_report(stable_files):
    print("")
    print("=== true-value columns (tsumo_count/all_clear_pending/ojama_net_balance/ojama_forecast) ===")
    rows = []
    for p in stable_files:
        d = np.load(str(p), allow_pickle=True)
        if "tsumo_count" not in d.files:
            continue
        n = len(d["grids"])
        sides = d["side"]
        rec = {"video_id": p.stem, "n_rows": n}
        for key in TRUE_VALUE_KEYS:
            if key not in d.files:
                rec[key + "_missing_pct"] = 100.0
                continue
            arr = np.asarray(d[key], dtype=float)
            rec[key + "_missing_pct"] = float(np.isnan(arr).mean()) * 100.0
            rec[key + "_mean"] = float(np.nanmean(arr))
            rec[key + "_std"] = float(np.nanstd(arr))
            rec[key + "_min"] = float(np.nanmin(arr))
            rec[key + "_max"] = float(np.nanmax(arr))
            if key == "ojama_net_balance":
                p1_mask = sides == "1P"
                p2_mask = sides == "2P"
                if p1_mask.sum() > 10 and p2_mask.sum() > 10:
                    rec["ojama_net_balance_1p_mean"] = float(np.nanmean(arr[p1_mask]))
                    rec["ojama_net_balance_2p_mean"] = float(np.nanmean(arr[p2_mask]))
        rows.append(rec)
    if not rows:
        print("[info] no stable npz with true-value columns")
        return
    tv_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "true_value_report.csv"
    tv_df.to_csv(out_path, index=False)
    print(f"[info] true-value report {len(tv_df)} videos -> {out_path}")
    for key in TRUE_VALUE_KEYS:
        col = key + "_missing_pct"
        if col in tv_df.columns:
            print(f"  {key}: missing rate mean={tv_df[col].mean():.2f}% max={tv_df[col].max():.2f}%")


def main() -> None:
    stable_files, _excluded = select_stable_npz()
    df = build_current_csv(stable_files)
    n_videos = df["video_id"].nunique()
    n_games = df[["video_id", "game_idx"]].drop_duplicates().shape[0]
    print(f"[info] rows={len(df)} videos={n_videos} games={n_games}")

    # won ラベル整合性チェック (2026-08-13 実データで発覚: 0/1 以外の値=NaN が
    # 一部videoで100%を占め、roc_auc_score が multi_class エラーで落ちる)。
    # AUC診断は labeled subset のみで行う (won not in {0,1} は除外し、除外内訳を報告)。
    n_total_rows = len(df)
    unlabeled_mask = ~df["won"].isin([0.0, 1.0])
    n_unlabeled = int(unlabeled_mask.sum())
    if n_unlabeled > 0:
        bad_videos = df.loc[unlabeled_mask, "video_id"].value_counts()
        print(
            f"[WARN] won label unlabeled rows = {n_unlabeled} "
            f"({100.0 * n_unlabeled / n_total_rows:.2f}% of {n_total_rows})",
        )
        print("[WARN] unlabeled-won video breakdown:")
        print(bad_videos.to_string())
        df = df[~unlabeled_mask].copy()
        print(f"[info] rows after label filter = {len(df)}")

    df = add_phase(df)
    indicator_cols = [c for c in df.columns if c not in META_COLS and c not in ("progress", "phase")]
    print(f"[info] indicator_cols({len(indicator_cols)}) = {indicator_cols}")

    print("")
    print("=== univariate AUC diagnosis ===")
    uni = univariate_report(df, indicator_cols)
    uni_out = OUT_DIR / "univariate_report.csv"
    uni.to_csv(uni_out, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(uni[[
        "col", "direction", "overall_auc", "early_auc", "mid_auc", "late_auc",
        "video_median_auc", "missing_pct", "zero_var", "is_uninformative",
    ]].to_string(index=False))
    print(f"[info] univariate csv -> {uni_out}")

    print("")
    print("=== redundancy (Spearman) ===")
    pairs_df, clusters = redundancy_report(df, indicator_cols, uni)
    pairs_out = OUT_DIR / "redundant_pairs.csv"
    pairs_df.to_csv(pairs_out, index=False)
    print(f"|rho|>{CORR_THRESHOLD_PAIR} pairs = {len(pairs_df)} -> {pairs_out}")
    print(pairs_df.to_string(index=False))
    print("")
    print(f"|rho|>{CORR_THRESHOLD_CLUSTER} clusters = {len(clusters)}")
    cluster_rows = []
    for cl in clusters:
        rep = recommend_cluster_representative(cl, uni)
        print(f"  cluster={cl} -> recommended={rep}")
        cluster_rows.append({"cluster": ",".join(cl), "recommended": rep})
    pd.DataFrame(cluster_rows).to_csv(OUT_DIR / "redundant_clusters.csv", index=False)

    print("")
    print("=== interaction / new columns individual check ===")
    special_cols = [
        c for c in [
            "color_ojama_ratio_own", "color_diff_x_ojama_diff",
            "all_clear_bonus_pending", "opp_all_clear_bonus_pending",
            "center_bulge_color", "center_bulge_ojama",
        ] if c in df.columns
    ]
    print(uni[uni["col"].isin(special_cols)][[
        "col", "direction", "overall_auc", "early_auc", "mid_auc", "late_auc", "missing_pct",
    ]].to_string(index=False))
    if "all_clear_bonus_pending" in df.columns:
        rate = df["all_clear_bonus_pending"].mean()
        print(f"[info] all_clear_bonus_pending ON rate = {rate * 100:.3f}%")

    print("")
    print("=== multivariate (HistGBC + permutation importance) ===")
    oof_auc, imp_df = multivariate_report(df, indicator_cols)
    imp_out = OUT_DIR / "permutation_importance.csv"
    imp_df.to_csv(imp_out, index=False)
    print(f"[info] OOF AUC (all cols) = {oof_auc:.4f}")
    print(imp_df.to_string(index=False))
    print(f"[info] importance csv -> {imp_out}")

    if BASELINE_EVIDENCE_CSV.exists():
        print("")
        print("=== compare vs 8/12 63-video baseline (overall_auc) ===")
        base = pd.read_csv(BASELINE_EVIDENCE_CSV)[["col", "overall_auc", "video_median_auc"]]
        base = base.rename(columns={
            "overall_auc": "overall_auc_0812",
            "video_median_auc": "video_median_auc_0812",
        })
        cmp_df = uni[["col", "overall_auc", "video_median_auc"]].merge(base, on="col", how="left")
        cmp_df["delta_overall"] = cmp_df["overall_auc"] - cmp_df["overall_auc_0812"]
        cmp_out = OUT_DIR / "compare_vs_0812.csv"
        cmp_df.to_csv(cmp_out, index=False)
        print(cmp_df.to_string(index=False))
        print(f"[info] compare csv -> {cmp_out}")

    true_value_report(stable_files)
    print("")
    print("[all done]")


if __name__ == "__main__":
    raise SystemExit(main())
