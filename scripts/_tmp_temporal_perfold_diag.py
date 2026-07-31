"""proto_temporal_winprob の中盤 AUC 差 (static_diff vs temporal_K3) が
video ごとにどれだけ安定しているかを LeaveOneGroupOut (10動画=10 fold) で
診断する一時スクリプト (使い捨て・非正式)。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from scripts.model_indicator_win import (  # noqa: E402
    GBC_PARAMS, build_features, load_labeled_csv, pair_sides_for_win,
    _get_indicator_cols,
)
from scripts.proto_temporal_winprob import (  # noqa: E402
    K_LIST, K_MAX, CURATED_INDICATORS, LABELED_CSV, TSUMO_EARLY_MAX,
    TSUMO_LATE_MIN, assign_match_segments, build_variant_column_sets,
    compute_momentum_features, compute_sign_flip_features,
    compute_slope_var_features, drop_inconsistent_segments,
    filter_min_history,
)


def _logo_auc_per_video(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """LeaveOneGroupOut で video ごとの held-out AUC を返す。"""
    logo = LeaveOneGroupOut()
    result: dict[str, float] = {}
    for tr_idx, te_idx in logo.split(X, y, groups=groups):
        vid = groups[te_idx][0]
        if len(np.unique(y[te_idx])) < 2:
            result[vid] = float("nan")
            continue
        model = HistGradientBoostingClassifier(**GBC_PARAMS)
        model.fit(X[tr_idx], y[tr_idx])
        p = model.predict_proba(X[te_idx])[:, 1]
        result[vid] = float(roc_auc_score(y[te_idx], p))
    return result


def main() -> None:
    df = load_labeled_csv(str(LABELED_CSV))
    paired = pair_sides_for_win(df, 1.0)
    seg = assign_match_segments(paired)
    seg = drop_inconsistent_segments(seg)
    indicator_cols = _get_indicator_cols(seg)
    diff_cols = [f"{c}_diff" for c in indicator_cols]
    curated_cols = [f"{c}_diff" for c in indicator_cols if c in CURATED_INDICATORS]
    feat_df = build_features(seg, indicator_cols)
    diff_only = feat_df[[c for c in feat_df.columns if c.endswith("_diff")]]
    seg = pd.concat([seg.reset_index(drop=True), diff_only.reset_index(drop=True)], axis=1)
    seg = compute_momentum_features(seg, diff_cols, K_LIST)
    seg = compute_slope_var_features(seg, curated_cols, K_LIST)
    seg = compute_sign_flip_features(seg, curated_cols, K_LIST)
    dff = filter_min_history(seg, K_MAX)

    tcr = dff["tsumo_count_rate_1p"].astype(float)
    mid_mask = (tcr > TSUMO_EARLY_MAX) & (tcr <= TSUMO_LATE_MIN)
    mid = dff[mid_mask].reset_index(drop=True)
    y = mid["won_1p"].astype(int).values
    groups = mid["video_id_1p"].values

    variants = build_variant_column_sets(diff_cols, curated_cols, K_LIST)
    for name in ["static_diff", "temporal_K3", "temporal_K8"]:
        cols = [c for c in variants[name] if c in mid.columns]
        X = mid[cols].fillna(0.0).values.astype(float)
        per_video = _logo_auc_per_video(X, y, groups)
        vals = [v for v in per_video.values() if not np.isnan(v)]
        print(f"\n[{name}] n_features={len(cols)}  video別AUC:")
        for vid, auc in sorted(per_video.items()):
            n_v = int((groups == vid).sum())
            print(f"    {vid}: AUC={auc:.4f}  n={n_v}")
        print(f"  平均={np.mean(vals):.4f}  中央値={np.median(vals):.4f}  std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
