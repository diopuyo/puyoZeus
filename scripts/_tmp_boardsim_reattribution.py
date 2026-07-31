"""board sim reattribution analysis script."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.model_indicator_win import (
    load_labeled_csv,
    pair_sides_for_win,
    META_COLS,
    REDUNDANT_COLS,
    GBC_PARAMS,
    N_FOLDS,
    TSUMO_EARLY_RATIO,
    TSUMO_LATE_RATIO,
    DEFAULT_MAX_TDIFF,
)

LABELED_CSV = "data/indicators_v2/study/labeled_win.csv"
OUT_AUC_CSV = "data/indicators_v2/boardsim_reattribution_auc.csv"
OUT_PERM_CSV = "data/indicators_v2/boardsim_reattribution_perm.csv"
OUT_CORR_CSV = "data/indicators_v2/boardsim_reattribution_corr.csv"

BOARD_SIM_NAMES = [
    "saturated_chain_count",
    "ignition_point_count",
    "multi_color_ignition",
    "sub_chain_count",
    "simultaneous_pop_richness",
]
NEW4_NAMES = [
    "ignition_point_count",
    "multi_color_ignition",
    "sub_chain_count",
    "simultaneous_pop_richness",
]

PERM_N_REPEATS = 20
PERM_RANDOM_STATE = 42


def get_baseline_cols(paired):
    all_exclude = META_COLS | REDUNDANT_COLS | frozenset(BOARD_SIM_NAMES)
    result = []
    for col in paired.columns:
        if not col.endswith("_1p"):
            continue
        base = col[:-3]
        if base in all_exclude:
            continue
        if base.endswith("_raw") or base.endswith("_source"):
            continue
        if base == "reach_fire_power_max_chain":
            continue
        if pd.api.types.is_numeric_dtype(paired[col]):
            result.append(base)
    return result


def build_X(paired, bases):
    feat = {}
    for base in bases:
        c1, c2 = base + "_1p", base + "_2p"
        if c1 in paired.columns:
            feat[c1] = paired[c1].astype(float)
        if c2 in paired.columns:
            feat[c2] = paired[c2].astype(float)
        if c1 in paired.columns and c2 in paired.columns:
            feat[base + "_diff"] = paired[c1].astype(float) - paired[c2].astype(float)
    fdf = pd.DataFrame(feat, index=paired.index).fillna(0.0)
    return fdf.values.astype(float), list(fdf.columns)


def oof_auc(X, y, groups, n_folds):
    n_uni = len(np.unique(groups))
    folds = min(n_folds, max(2, n_uni))
    proba = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(**GBC_PARAMS)
        m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    valid = ~np.isnan(proba)
    yv, pv = y[valid], proba[valid]
    if len(np.unique(yv)) < 2:
        return float("nan")
    return float(roc_auc_score(yv, pv))
def phase_masks(paired):
    ts = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(ts, TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(ts, TSUMO_LATE_RATIO))
    n = len(ts)
    return {
        "zentai": np.ones(n, dtype=bool),
        "jyoban": ts <= q33,
        "chuban": (ts > q33) & (ts <= q67),
        "shuban": ts > q67,
    }


def run_condition(paired, y, groups, masks, bases, n_folds):
    X, _ = build_X(paired, bases)
    result = {}
    for phase, mask in masks.items():
        Xp, yp, gp = X[mask], y[mask], groups[mask]
        if len(Xp) < 20 or len(np.unique(yp)) < 2:
            result[phase] = float("nan")
            continue
        result[phase] = oof_auc(Xp, yp, gp, n_folds)
    return result


def compute_perm_importance(X, y, groups, feat_names, n_folds):
    n_uni = len(np.unique(groups))
    folds = min(n_folds, max(2, n_uni))
    imp_list = []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups=groups):
        m = HistGradientBoostingClassifier(**GBC_PARAMS)
        m.fit(X[tr], y[tr])
        perm = permutation_importance(
            m, X[te], y[te], n_repeats=PERM_N_REPEATS,
            random_state=PERM_RANDOM_STATE, scoring="roc_auc",
        )
        imp_list.append(perm.importances_mean)
    imp = np.array(imp_list)
    out = pd.DataFrame({
        "feature": feat_names,
        "importance_mean": imp.mean(axis=0),
        "importance_std": imp.std(axis=0, ddof=1),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out
def main():
    print("[main] labeled=" + LABELED_CSV, flush=True)
    df = load_labeled_csv(LABELED_CSV)
    paired = pair_sides_for_win(df, DEFAULT_MAX_TDIFF)
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    masks = phase_masks(paired)
    for ph, m in masks.items():
        print("  phase " + ph + ": n=" + str(int(m.sum())), flush=True)

    baseline_cols = get_baseline_cols(paired)
    print("[baseline] tier1 base col count (excl XII): " + str(len(baseline_cols)), flush=True)
    print("  cols: " + str(baseline_cols), flush=True)

    conditions = {"baseline": baseline_cols}
    for name in NEW4_NAMES:
        conditions["+" + name] = baseline_cols + [name]
    conditions["+all4"] = baseline_cols + NEW4_NAMES
    conditions["+all5_with_saturated"] = baseline_cols + BOARD_SIM_NAMES

    records = []
    print("\n=== condition-wise OOF AUC by phase ===", flush=True)
    for cond, bases in conditions.items():
        aucs = run_condition(paired, y, groups, masks, bases, N_FOLDS)
        row = {"condition": cond, "n_bases": len(bases)}
        for ph, v in aucs.items():
            row["auc_" + ph] = v
        records.append(row)
        msg = "  [" + cond + "] "
        for ph, v in aucs.items():
            msg += ph + "=" + format(v, ".4f") + "  "
        print(msg, flush=True)

    result_df = pd.DataFrame(records)
    base_row = result_df[result_df["condition"] == "baseline"].iloc[0]
    for ph in masks.keys():
        result_df["delta_" + ph] = result_df["auc_" + ph] - base_row["auc_" + ph]
    result_df.to_csv(OUT_AUC_CSV, index=False)
    print("\n[save] " + OUT_AUC_CSV, flush=True)

    print("\n=== marginal delta chuban (vs baseline) ===", flush=True)
    for _, r in result_df.iterrows():
        print("  " + str(r["condition"]).ljust(28) + " delta_chuban=" + format(r["delta_chuban"], "+.4f") + "  auc_chuban=" + format(r["auc_chuban"], ".4f"), flush=True)
    print("\n=== Permutation Importance (+all5, zentai) ===", flush=True)
    X5_all, feat5_names = build_X(paired, baseline_cols + BOARD_SIM_NAMES)
    perm_all = compute_perm_importance(X5_all, y, groups, feat5_names, N_FOLDS)
    perm_all["subset"] = "zentai"

    print("\n=== Permutation Importance (+all5, chuban) ===", flush=True)
    mid_mask = masks["chuban"]
    X5_mid = X5_all[mid_mask]
    y_mid = y[mid_mask]
    g_mid = groups[mid_mask]
    perm_mid = compute_perm_importance(X5_mid, y_mid, g_mid, feat5_names, N_FOLDS)
    perm_mid["subset"] = "chuban"

    perm_combined = pd.concat([perm_all, perm_mid], ignore_index=True)
    perm_combined.to_csv(OUT_PERM_CSV, index=False)
    print("[save] " + OUT_PERM_CSV, flush=True)

    print("\n--- zentai top20 ---", flush=True)
    for _, r in perm_all.head(20).iterrows():
        mark = "*" if any(nm in r["feature"] for nm in BOARD_SIM_NAMES) else " "
        print("  " + str(int(r["rank"])).rjust(3) + mark + " " + str(r["feature"]).ljust(40) + " " + format(r["importance_mean"], "+.6f"), flush=True)

    print("\n--- chuban top20 ---", flush=True)
    for _, r in perm_mid.head(20).iterrows():
        mark = "*" if any(nm in r["feature"] for nm in BOARD_SIM_NAMES) else " "
        print("  " + str(int(r["rank"])).rjust(3) + mark + " " + str(r["feature"]).ljust(40) + " " + format(r["importance_mean"], "+.6f"), flush=True)

    print("\n=== corr with current_max_chain (1p/2p raw values pooled) ===", flush=True)
    corr_records = []
    cmc_raw = pd.concat([paired["current_max_chain_raw_1p"], paired["current_max_chain_raw_2p"]], ignore_index=True).astype(float)
    for name in BOARD_SIM_NAMES:
        c1 = name + "_raw_1p"
        c2 = name + "_raw_2p"
        if c1 not in paired.columns or c2 not in paired.columns:
            continue
        vals = pd.concat([paired[c1], paired[c2]], ignore_index=True).astype(float)
        corr = float(cmc_raw.corr(vals))
        corr_records.append({"indicator": name, "corr_with_current_max_chain": corr})
        print("  " + name.ljust(28) + " corr=" + format(corr, "+.4f"), flush=True)
    corr_df = pd.DataFrame(corr_records)
    corr_df.to_csv(OUT_CORR_CSV, index=False)
    print("[save] " + OUT_CORR_CSV, flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
