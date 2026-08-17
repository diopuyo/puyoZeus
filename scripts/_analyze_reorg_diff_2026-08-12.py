# -*- coding: utf-8 -*-
"""
Diff-version and PCA reduction evidence script (2026-08-12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CSV_PATH = "data/verify/npz_light_smoke_2026-08-12/labeled_win_light63.csv"

BASE_COLS = [
    "board_color_puyo_total",
    "board_puyo_total",
    "max_column_height",
    "column_bumpiness",
    "death_margin",
    "death_margin_neighbor",
    "center_bulge",
    "board_ojama_count",
    "conn_pair_count",
    "conn_triple_count",
    "conn_max_group_size",
]

FILL_FAMILY_COLS = [
    "board_puyo_total",
    "max_column_height",
    "death_margin",
    "death_margin_neighbor",
    "board_color_puyo_total",
]

PHASE_LABELS = ["early", "mid", "late"]


def compute_progress(df):
    grp = df.groupby(["video_id", "game_idx"])["t_sec"]
    t_min = grp.transform("min")
    t_max = grp.transform("max")
    rng = (t_max - t_min).replace(0, np.nan)
    progress = (df["t_sec"] - t_min) / rng
    return progress.fillna(0.5).clip(0.0, 1.0)


def safe_auc(y, x):
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


def oriented_auc_table(df, col):
    y_all = df["won"].to_numpy()
    x_all = df[col].to_numpy(dtype=float)
    raw_auc = safe_auc(y_all, x_all)
    if raw_auc is None:
        return {"col": col, "direction": "N/A", "overall_auc": None}
    direction = "+" if raw_auc >= 0.5 else "-"
    overall_auc = raw_auc if direction == "+" else 1.0 - raw_auc

    out = {"col": col, "direction": direction, "overall_auc": overall_auc}
    for ph in PHASE_LABELS:
        sub = df[df["phase"] == ph]
        a = safe_auc(sub["won"].to_numpy(), sub[col].to_numpy(dtype=float))
        out[ph + "_auc"] = (a if direction == "+" else 1.0 - a) if a is not None else None

    video_aucs = []
    for _vid, sub in df.groupby("video_id"):
        a = safe_auc(sub["won"].to_numpy(), sub[col].to_numpy(dtype=float))
        if a is None:
            continue
        video_aucs.append(a if direction == "+" else 1.0 - a)
    out["video_median_auc"] = float(np.median(video_aucs)) if video_aucs else None
    out["video_n"] = len(video_aucs)
    return out


def _grouped_opp_dict(opp_src, group_cols):
    out = {}
    for key, g in opp_src.groupby(group_cols, sort=False):
        gs = g.sort_values("t_sec").drop(columns=group_cols).reset_index(drop=True)
        out[key] = gs
    return out


def _asof_join_by_game(self_df, opp_dict, group_cols, opp_col_names):
    parts = []
    for key, g in self_df.groupby(group_cols, sort=False):
        gs = g.sort_values("t_sec").reset_index(drop=True)
        opp_g = opp_dict.get(key)
        if opp_g is None or len(opp_g) == 0:
            gs2 = gs.copy()
            for c in opp_col_names:
                gs2[c] = np.nan
            parts.append(gs2)
            continue
        merged = pd.merge_asof(gs, opp_g, on="t_sec", direction="backward")
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def build_paired_diff(df):
    keep_meta = ["video_id", "game_idx", "t_sec", "frame", "won", "phase", "progress"]
    group_cols = ["video_id", "game_idx"]
    df1 = df[df["side"] == "1P"][keep_meta + BASE_COLS].reset_index(drop=True)
    df2 = df[df["side"] == "2P"][keep_meta + BASE_COLS].reset_index(drop=True)

    opp_cols = {c: "opp_" + c for c in BASE_COLS}
    opp_col_names = list(opp_cols.values())
    df2_as_opp = df2[group_cols + ["t_sec"] + BASE_COLS].rename(columns=opp_cols)
    df1_as_opp = df1[group_cols + ["t_sec"] + BASE_COLS].rename(columns=opp_cols)

    groups2 = _grouped_opp_dict(df2_as_opp, group_cols)
    groups1 = _grouped_opp_dict(df1_as_opp, group_cols)

    merged_1p = _asof_join_by_game(df1, groups2, group_cols, opp_col_names)
    merged_1p["side"] = "1P"
    merged_2p = _asof_join_by_game(df2, groups1, group_cols, opp_col_names)
    merged_2p["side"] = "2P"

    n_1p_total, n_2p_total = len(merged_1p), len(merged_2p)
    n_1p_matched = merged_1p["opp_" + BASE_COLS[0]].notna().sum()
    n_2p_matched = merged_2p["opp_" + BASE_COLS[0]].notna().sum()

    merged_1p = merged_1p.dropna(subset=["opp_" + BASE_COLS[0]]).copy()
    merged_2p = merged_2p.dropna(subset=["opp_" + BASE_COLS[0]]).copy()

    combined = pd.concat([merged_1p, merged_2p], ignore_index=True)
    for c in BASE_COLS:
        combined["diff_" + c] = combined[c] - combined["opp_" + c]

    pairing_info = {
        "n_1p_total": int(n_1p_total),
        "n_1p_matched": int(n_1p_matched),
        "n_2p_total": int(n_2p_total),
        "n_2p_matched": int(n_2p_matched),
        "n_total": int(n_1p_total + n_2p_total),
        "n_matched": int(n_1p_matched + n_2p_matched),
    }
    return combined, pairing_info


def main():
    df = pd.read_csv(CSV_PATH)
    df["progress"] = compute_progress(df)
    df["phase"] = pd.cut(
        df["progress"], bins=[-0.001, 1 / 3, 2 / 3, 1.001], labels=PHASE_LABELS
    )

    combined, pairing_info = build_paired_diff(df)
    print("[info] pairing success rate:")
    r1 = 100.0 * pairing_info["n_1p_matched"] / pairing_info["n_1p_total"]
    r2 = 100.0 * pairing_info["n_2p_matched"] / pairing_info["n_2p_total"]
    r3 = 100.0 * pairing_info["n_matched"] / pairing_info["n_total"]
    print("  1P: " + str(pairing_info["n_1p_matched"]) + "/" + str(pairing_info["n_1p_total"]) + " (" + str(round(r1, 2)) + "%)")
    print("  2P: " + str(pairing_info["n_2p_matched"]) + "/" + str(pairing_info["n_2p_total"]) + " (" + str(round(r2, 2)) + "%)")
    print("  all: " + str(pairing_info["n_matched"]) + "/" + str(pairing_info["n_total"]) + " (" + str(round(r3, 2)) + "%)")
    print("[info] combined subset rows: " + str(len(combined)))

    rows_out = []
    for c in BASE_COLS:
        own = oriented_auc_table(combined, c)
        diff = oriented_auc_table(combined, "diff_" + c)
        rows_out.append(
            {
                "col": c,
                "own_direction": own["direction"],
                "own_overall_same_subset": own["overall_auc"],
                "own_early": own.get("early_auc"),
                "own_mid": own.get("mid_auc"),
                "own_late": own.get("late_auc"),
                "own_video_median": own.get("video_median_auc"),
                "diff_direction": diff["direction"],
                "diff_overall": diff["overall_auc"],
                "diff_early": diff.get("early_auc"),
                "diff_mid": diff.get("mid_auc"),
                "diff_late": diff.get("late_auc"),
                "diff_video_median": diff.get("video_median_auc"),
            }
        )

    res = pd.DataFrame(rows_out)
    res["gain_overall"] = res["diff_overall"] - res["own_overall_same_subset"]
    res = res.sort_values("gain_overall", ascending=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print("")
    print("=== own-board (same subset) vs diff comparison ===")
    print(res.to_string(index=False))

    res.to_csv(
        "data/verify/npz_light_smoke_2026-08-12/_reorg_diff_comparison_2026-08-12.csv",
        index=False,
    )

    print("")
    print("=== fill family 5 cols (diff version) PCA reduction check (late phase) ===")
    fill_diff_cols = ["diff_" + c for c in FILL_FAMILY_COLS]
    fit_data = combined[fill_diff_cols].dropna()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(fit_data.to_numpy())
    pca = PCA(n_components=1, random_state=0)
    pc1_all = pca.fit_transform(x_scaled).ravel()
    print("[info] PC1 explained variance ratio: " + str(round(float(pca.explained_variance_ratio_[0]), 4)))
    loadings = dict(zip(fill_diff_cols, [round(float(v), 4) for v in pca.components_[0]]))
    print("[info] PC1 loadings: " + str(loadings))

    combined2 = combined.loc[fit_data.index].copy()
    combined2["pc1_fill_family"] = pc1_all

    end_only = combined2[combined2["phase"] == "late"]
    pc1_end_auc = safe_auc(end_only["won"].to_numpy(), end_only["pc1_fill_family"].to_numpy())
    if pc1_end_auc is not None and pc1_end_auc < 0.5:
        pc1_end_auc = 1.0 - pc1_end_auc
    print("[info] PC1 late-phase AUC = " + str(pc1_end_auc))

    best_single = None
    best_single_col = None
    for c in fill_diff_cols:
        a = safe_auc(end_only["won"].to_numpy(), end_only[c].to_numpy(dtype=float))
        if a is None:
            continue
        a_oriented = a if a >= 0.5 else 1.0 - a
        print("  " + c + ": late-phase AUC (single) = " + str(round(a_oriented, 6)))
        if best_single is None or a_oriented > best_single:
            best_single = a_oriented
            best_single_col = c

    print("")
    print("[info] best single of 5: " + str(best_single_col) + " AUC=" + str(round(best_single, 6)))
    print("[info] PC1 combined:      AUC=" + str(round(pc1_end_auc, 6)))
    if best_single is not None and pc1_end_auc is not None:
        print("[info] drop = " + str(round(best_single - pc1_end_auc, 6)) + " (best_single - PC1)")


if __name__ == "__main__":
    main()
