"""148本フルデータ再学習 + 評価 (タスク#4本丸、2026-08-14)。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

DEFAULT_CSV: str = "data/verify/labeled_win_full148_2026-08-14/labeled_win_full148.csv"
DEFAULT_OUT_DIR: str = "data/verify/retrain148_2026-08-14"

N_FOLDS: int = 5

GBC_PARAMS: dict = {
    "max_iter": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "random_state": 42,
    "early_stopping": False,
}

PERM_N_REPEATS: int = 15
PERM_RANDOM_STATE: int = 42

META_COLS: frozenset = frozenset({
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side", "won",
})

SOURCE_COLS: frozenset = frozenset({"all_clear_source", "ojama_source"})

LIGHT63_EQUIV_COLS = (
    "board_color_puyo_total",
    "board_puyo_total",
    "diff_max_column_height",
    "diff_column_bumpiness",
    "diff_death_margin",
    "diff_death_margin_neighbor",
    "center_bulge_color",
    "center_bulge_ojama",
    "board_ojama_count",
    "diff_conn_pair_count",
    "conn_triple_count",
    "diff_conn_max_group_size",
)

OFFSET_COL: str = "prior_win_rate_offset"
OFFSET_N_COL: str = "prior_n_games_offset"

CALIB_N_BINS: int = 10


def load_data(csv_path: str, quick_n_videos):
    """CSVを読み込み、won欠損行を除外する。"""
    t0 = time.time()
    df = pd.read_csv(csv_path)
    n0 = len(df)
    df = df.dropna(subset=["won"]).reset_index(drop=True)
    df["won"] = df["won"].astype(int)
    if quick_n_videos is not None:
        keep_videos = sorted(df["video_id"].unique())[:quick_n_videos]
        df = df[df["video_id"].isin(keep_videos)].reset_index(drop=True)
    print(f"  読込 {time.time()-t0:.1f}秒: {n0}行 -> won欠損除外後 {len(df)}行"
          f" (動画数={df['video_id'].nunique()})")
    return df


def resolve_full_feature_cols(df):
    exclude = META_COLS | SOURCE_COLS
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def compute_match_progress(df):
    max_tsumo = df.groupby(["video_id", "game_idx", "side"])["tsumo"].transform("max")
    max_tsumo = max_tsumo.replace(0, np.nan)
    progress = (df["tsumo"] / max_tsumo).fillna(0.0).clip(0.0, 1.0)
    return progress


def compute_strength_offset(df):
    game_level = (
        df.drop_duplicates(["video_id", "game_idx", "side"])
        [["video_id", "game_idx", "side", "won"]]
        .sort_values(["video_id", "side", "game_idx"])
        .reset_index(drop=True)
    )
    grp = game_level.groupby(["video_id", "side"])
    cum_wins_incl = grp["won"].cumsum()
    cum_n_incl = grp.cumcount() + 1
    cum_wins_prior = cum_wins_incl - game_level["won"]
    cum_n_prior = cum_n_incl - 1
    game_level[OFFSET_COL] = np.where(
        cum_n_prior > 0, cum_wins_prior / cum_n_prior.replace(0, np.nan), 0.5,
    )
    game_level[OFFSET_COL] = game_level[OFFSET_COL].fillna(0.5)
    game_level[OFFSET_N_COL] = cum_n_prior
    return game_level[["video_id", "game_idx", "side", OFFSET_COL, OFFSET_N_COL]]


def run_oof(X, y, groups, n_folds):
    """GroupKFoldでOOF確率(1列, P(won=1))を返す。fold_id配列も返す。"""
    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    gkf = GroupKFold(n_splits=n_folds)
    models = []
    for i, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        m = HistGradientBoostingClassifier(**GBC_PARAMS)
        t0 = time.time()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        fold_id[te] = i
        models.append(m)
        n_v = len(np.unique(groups[te]))
        print(f"    fold {i+1}/{n_folds}: train={len(tr)} test={len(te)}"
              f" (動画{n_v}本) 学習{time.time()-t0:.1f}秒")
    return oof, fold_id, models


def eval_oof(y, p, label):
    valid = ~np.isnan(p)
    y_v, p_v = y[valid], p[valid]
    auc = float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else float("nan")
    ll = float(log_loss(y_v, np.column_stack([1 - p_v, p_v])))
    brier = float(np.mean((p_v - y_v) ** 2))
    print(f"    [{label}] AUC={auc:.4f} logloss={ll:.4f} brier={brier:.4f} n={valid.sum()}")
    return {"label": label, "auc": auc, "logloss": ll, "brier": brier, "n": int(valid.sum())}


def per_video_auc(y, p, groups):
    rows = []
    for vid in np.unique(groups):
        mask = groups == vid
        y_v, p_v = y[mask], p[mask]
        valid = ~np.isnan(p_v)
        y_v, p_v = y_v[valid], p_v[valid]
        if len(np.unique(y_v)) < 2:
            rows.append({"video_id": vid, "auc": np.nan, "n": len(y_v)})
            continue
        auc = float(roc_auc_score(y_v, p_v))
        rows.append({"video_id": vid, "auc": auc, "n": len(y_v)})
    return pd.DataFrame(rows)


def phase_auc_report(y, p, progress, groups):
    """試合内相対進行率の3分位(プール全体で0.333/0.667固定)で層別AUC。"""
    bounds = (1.0 / 3.0, 2.0 / 3.0)
    masks = {
        "序盤": progress <= bounds[0],
        "中盤": (progress > bounds[0]) & (progress <= bounds[1]),
        "終盤": progress > bounds[1],
    }
    result = {}
    for name, mask in masks.items():
        y_p, p_p = y[mask], p[mask]
        valid = ~np.isnan(p_p)
        y_v, p_v = y_p[valid], p_p[valid]
        auc = float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else float("nan")
        vids = groups[mask][valid]
        pv_df = per_video_auc(y_v, p_v, vids)
        med = float(pv_df["auc"].median()) if pv_df["auc"].notna().any() else float("nan")
        print(f"    {name}: n={valid.sum()} プールAUC={auc:.4f} 動画別中央値={med:.4f}"
              f" (有効動画{pv_df['auc'].notna().sum()}本)")
        result[name] = {"pooled_auc": auc, "video_median_auc": med, "n": int(valid.sum())}
    return result


def compute_perm_importance(X, y, groups, feature_names, n_folds, n_repeats):
    gkf = GroupKFold(n_splits=n_folds)
    per_fold = []
    for i, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        m = HistGradientBoostingClassifier(**GBC_PARAMS)
        m.fit(X[tr], y[tr])
        t0 = time.time()
        perm = permutation_importance(
            m, X[te], y[te], n_repeats=n_repeats,
            random_state=PERM_RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
        )
        per_fold.append(perm.importances_mean)
        print(f"    perm fold {i+1}/{n_folds} 完了 ({time.time()-t0:.1f}秒)")
    mat = np.array(per_fold)
    out = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": mat.mean(axis=0),
        "importance_std": mat.std(axis=0, ddof=1) if n_folds > 1 else np.zeros(len(feature_names)),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def calibration_table(y, p, n_bins=CALIB_N_BINS):
    valid = ~np.isnan(p)
    y_v, p_v = y[valid], p[valid]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_v, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": b, "pred_mean": np.nan, "actual_rate": np.nan, "n": 0})
            continue
        rows.append({
            "bin": b, "pred_mean": float(p_v[mask].mean()),
            "actual_rate": float(y_v[mask].mean()), "n": n,
        })
    return pd.DataFrame(rows)


def compute_ece(calib_df, n_total):
    df = calib_df.dropna(subset=["pred_mean", "actual_rate"])
    if len(df) == 0 or n_total == 0:
        return float("nan")
    weights = df["n"] / n_total
    return float((weights * (df["pred_mean"] - df["actual_rate"]).abs()).sum())


def main():
    parser = argparse.ArgumentParser(description="148本フル学習し直し+評価")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--perm-repeats", type=int, default=PERM_N_REPEATS)
    parser.add_argument("--skip-perm", action="store_true")
    parser.add_argument("--quick", type=int, default=None, help="先頭N動画のみで疎通確認")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 80)
    print("  148本フルデータ再学習・評価 (タスク#4本丸)")
    print("=" * 80)

    print("\n=== 1. データ読み込み ===")
    df = load_data(args.csv, args.quick)

    print("\n=== 2. 試合内相対進行率 + 実力差オフセット計算 ===")
    df["match_progress"] = compute_match_progress(df)
    offset_tbl = compute_strength_offset(df)
    df = df.merge(offset_tbl, on=["video_id", "game_idx", "side"], how="left")
    print(f"  match_progress 分布: min={df['match_progress'].min():.3f}"
          f" max={df['match_progress'].max():.3f} mean={df['match_progress'].mean():.3f}")
    print(f"  {OFFSET_COL} 分布: mean={df[OFFSET_COL].mean():.3f}"
          f" std={df[OFFSET_COL].std():.3f}"
          f" (0.5からの乖離平均={df[OFFSET_COL].sub(0.5).abs().mean():.3f})")

    y = df["won"].values.astype(int)
    groups = df["video_id"].values
    progress = df["match_progress"].values

    full_cols = resolve_full_feature_cols(df)
    full_cols_no_meta_extra = [c for c in full_cols if c not in (
        "match_progress", OFFSET_COL, OFFSET_N_COL,
    )]
    light_cols = [c for c in LIGHT63_EQUIV_COLS if c in df.columns]
    missing_light = [c for c in LIGHT63_EQUIV_COLS if c not in df.columns]
    if missing_light:
        print(f"  [WARN] light63相当列で欠落: {missing_light}")

    print(f"\n  新フル列構成: {len(full_cols_no_meta_extra)}列")
    print(f"  旧light63相当列構成: {len(light_cols)}列")

    summary = {"n_rows": len(df), "n_videos": int(df["video_id"].nunique())}

    print("\n=== 3. ベース学習 (新フル列構成, HistGBC, GroupKFold) ===")
    X_full = df[full_cols_no_meta_extra].fillna(0.0).values.astype(np.float32)
    oof_full, fold_full, _ = run_oof(X_full, y, groups, args.n_folds)
    res_full = eval_oof(y, oof_full, "full148_フル列構成")
    summary["full_features"] = res_full
    pv_full = per_video_auc(y, oof_full, groups)
    summary["full_features"]["video_median_auc"] = float(pv_full["auc"].median())
    summary["full_features"]["video_auc_iqr"] = [
        float(pv_full["auc"].quantile(0.25)), float(pv_full["auc"].quantile(0.75)),
    ]
    print(f"    動画別AUC中央値={summary['full_features']['video_median_auc']:.4f}"
          f" IQR={summary['full_features']['video_auc_iqr']}")
    pv_full.to_csv(out_dir / "per_video_auc_full.csv", index=False)

    print("\n  --- 位相別 (フル列構成) ---")
    summary["full_features"]["phase"] = phase_auc_report(y, oof_full, progress, groups)

    print("\n=== 4. 旧light63相当 vs 新フル列構成 (同一144動画データ) ===")
    X_light = df[light_cols].fillna(0.0).values.astype(np.float32)
    oof_light, fold_light, _ = run_oof(X_light, y, groups, args.n_folds)
    res_light = eval_oof(y, oof_light, "full148_light63相当列")
    summary["light_equiv_features"] = res_light
    pv_light = per_video_auc(y, oof_light, groups)
    summary["light_equiv_features"]["video_median_auc"] = float(pv_light["auc"].median())
    print("\n  --- 位相別 (light63相当列) ---")
    summary["light_equiv_features"]["phase"] = phase_auc_report(y, oof_light, progress, groups)

    delta_auc = res_full["auc"] - res_light["auc"]
    print(f"\n  新列群の純増分 dAUC (プール) = {delta_auc:+.4f}")
    summary["delta_auc_new_columns"] = delta_auc

    print("\n=== 5. 実力差オフセット A(無) / B(有) ===")
    full_cols_with_offset = full_cols_no_meta_extra + [OFFSET_COL]
    X_offset = df[full_cols_with_offset].fillna(0.0).values.astype(np.float32)
    oof_offset, fold_offset, _ = run_oof(X_offset, y, groups, args.n_folds)
    res_offset = eval_oof(y, oof_offset, "full148_フル列+実力差オフセット(B)")
    summary["offset_b"] = res_offset
    summary["offset_a"] = res_full
    delta_offset = res_offset["auc"] - res_full["auc"]
    print(f"  オフセットB - オフセットA dAUC = {delta_offset:+.4f}")
    summary["delta_auc_offset"] = delta_offset

    calib_a = calibration_table(y, oof_full)
    calib_b = calibration_table(y, oof_offset)
    ece_a = compute_ece(calib_a, int((~np.isnan(oof_full)).sum()))
    ece_b = compute_ece(calib_b, int((~np.isnan(oof_offset)).sum()))
    print(f"  ECE: A(無)={ece_a:.4f}  B(有)={ece_b:.4f}")
    summary["ece_offset_a"] = ece_a
    summary["ece_offset_b"] = ece_b
    calib_a.to_csv(out_dir / "calibration_offset_a.csv", index=False)
    calib_b.to_csv(out_dir / "calibration_offset_b.csv", index=False)

    if not args.skip_perm:
        print("\n=== 6. permutation importance (フル列構成, HistGBC) ===")
        t0 = time.time()
        perm_df = compute_perm_importance(
            X_full, y, groups, full_cols_no_meta_extra, args.n_folds, args.perm_repeats,
        )
        print(f"  perm importance 完了 ({time.time()-t0:.1f}秒)")
        perm_df.to_csv(out_dir / "permutation_importance_full.csv", index=False)
        print("\n  上位20列:")
        for _, r in perm_df.head(20).iterrows():
            print(f"    {int(r['rank']):>3}. {r['feature']:<40} "
                  f"{r['importance_mean']:+.6f} (+-{r['importance_std']:.6f})")
        new_cols = [c for c in full_cols_no_meta_extra if c not in light_cols]
        new_rank = perm_df[perm_df["feature"].isin(new_cols)]
        print(f"\n  新列群 ({len(new_cols)}列) の中央値rank="
              f"{new_rank['rank'].median():.0f} / 全{len(perm_df)}列")
        dead_new = new_rank[new_rank["importance_mean"] <= 0.0]
        print(f"  新列群のうち importance<=0 (死んでいる疑い): {len(dead_new)}列")
        if len(dead_new) > 0:
            print(f"    -> {list(dead_new['feature'])}")
        summary["perm_new_cols_median_rank"] = float(new_rank["rank"].median())
        summary["perm_dead_new_cols"] = list(dead_new["feature"])
    else:
        print("\n=== 6. permutation importance: --skip-perm によりスキップ ===")

    print("\n=== 7. 採用候補モデルの保存 (全144動画データで最終学習) ===")
    t0 = time.time()
    final_model = HistGradientBoostingClassifier(**GBC_PARAMS)
    final_model.fit(X_full, y)
    train_sec = time.time() - t0
    print(f"  最終学習時間: {train_sec:.1f}秒 (n={len(y)}, 列={len(full_cols_no_meta_extra)})")
    try:
        import joblib
        joblib.dump(final_model, out_dir / "model_full148_full_features.joblib")
        print(f"  モデル保存: {out_dir / 'model_full148_full_features.joblib'}")
    except ImportError:
        print("  [WARN] joblib 未導入のためモデル保存スキップ")
    with open(out_dir / "feature_cols_full.json", "w", encoding="utf-8") as f:
        json.dump(full_cols_no_meta_extra, f, ensure_ascii=False, indent=2)
    summary["final_train_seconds"] = train_sec
    summary["n_features_full"] = len(full_cols_no_meta_extra)
    summary["n_features_light_equiv"] = len(light_cols)

    total_sec = time.time() - t_start
    summary["total_wall_seconds"] = total_sec
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"  全工程完了。総所要時間: {total_sec/60:.1f}分")
    print(f"  結果一式: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
