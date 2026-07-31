"""prescreen_boardsim_auc.csv を再利用し opp_buried/won の中盤で saturated分離テスト。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

DATA_CSV = "data/indicators_v2/prescreen_boardsim_auc.csv"
N_FOLDS = 5
BASELINE_COLS = [
    "fire_current_max_chain", "fire_death_margin", "fire_absorption_capacity",
    "fire_dig_resistance", "fire_potential_fire_power", "fire_immediate_fire_power",
    "fire_board_ojama_count", "fire_max_column_height", "fire_second_chain_potential",
    "opp_current_max_chain", "opp_death_margin", "opp_absorption_capacity",
    "opp_dig_resistance", "opp_potential_fire_power", "opp_immediate_fire_power",
    "opp_board_ojama_count", "opp_max_column_height", "opp_second_chain_potential",
    "diff_current_max_chain", "diff_death_margin", "diff_absorption_capacity",
    "diff_dig_resistance", "diff_potential_fire_power", "diff_immediate_fire_power",
    "diff_board_ojama_count", "diff_max_column_height", "diff_second_chain_potential",
    "net_ojama",
]
NEW4 = ["ignition_point_count", "multi_color_ignition", "sub_chain_count", "simultaneous_pop_richness"]
ALL5 = NEW4 + ["saturated_chain_count"]


def cols_for(names):
    return [p + n for n in names for p in ("fire_", "opp_", "diff_")]


def fill_bsim_nan(df):
    out = df.copy()
    for nm in ALL5:
        for pfx in ("fire_", "opp_", "diff_"):
            c = pfx + nm
            if c in out.columns:
                out[c] = out[c].fillna(-1.0)
    return out


def oof_auc(X, y, groups):
    n_uni = len(np.unique(groups))
    folds = min(N_FOLDS, max(2, n_uni))
    proba = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31, learning_rate=0.05, random_state=42)
        m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    valid = ~np.isnan(proba)
    yv, pv = y[valid], proba[valid]
    if len(np.unique(yv)) < 2:
        return float("nan"), int(valid.sum())
    return float(roc_auc_score(yv, pv)), int(valid.sum())


def main():
    df = pd.read_csv(DATA_CSV)
    df = fill_bsim_nan(df)
    conditions = {
        "baseline": BASELINE_COLS,
        "+new4(saturated除く)": BASELINE_COLS + cols_for(NEW4),
        "+all5(saturated込み)": BASELINE_COLS + cols_for(ALL5),
        "+saturated_only": BASELINE_COLS + cols_for(["saturated_chain_count"]),
    }
    for target in ["won", "opp_buried"]:
        for phase_label, phase_val in [("zentai", None), ("chuban", "中")]:
            sub = df if phase_val is None else df[df["phase"] == phase_val]
            y = sub[target].values.astype(int)
            groups = sub["video_id"].values
            print("==== " + target + " / " + phase_label + " (n=" + str(len(sub)) + ") ====")
            base_auc = None
            for cond, cols in conditions.items():
                valid_cols = [c for c in cols if c in sub.columns]
                X = sub[valid_cols].fillna(sub[valid_cols].median()).values.astype(float)
                auc, n = oof_auc(X, y, groups)
                if cond == "baseline":
                    base_auc = auc
                delta = auc - base_auc if (base_auc == base_auc and auc == auc) else float("nan")
                print("  " + cond.ljust(28) + " AUC=" + format(auc, ".4f") + " delta=" + format(delta, "+.4f") + " n=" + str(n))


if __name__ == "__main__":
    main()
