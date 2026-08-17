"""W12根治 P5先行検証: 真値85本のみで旧列 vs 新5列 (W12) + 指標整理アブレーション。

`scripts/_retrain148_2026-08-14.py` の構造を流用しつつ、以下を85本サブセット
向けに変更する:

- **同一85本・同一分割**で「旧列のみ」「旧列+W12新5列」の2モデルを比較する
  (148本の旧モデルと比べるとサンプル差が交絡するため、user指示により
  85本ベースラインを別途学習し直す)。
- 位相別 (序盤/中盤/終盤) AUC を両モデルで比較する。
- permutation importance は新5列モデルのみで計算し、新5列の順位を報告する。
- W12の核心 (予告216個以上での予測勝率の実勝率への近づき) を、
  `scripts/_diag_w12_quantify_2026-08-16.py` と同じ bucket 定義
  (`FORECAST_BUCKETS`) ・同じ位相定義 (progress三分位) で直接検証する。

## 指標整理アブレーション (2026-08-16 追加依頼、user決定「収集は続けるが判定に
使わない (除外リスト方式)」)
148本モデルの permutation importance (`data/verify/retrain148_2026-08-14/
permutation_importance_full.csv`) で下位が死んでいた列を、同じ85本・同じ分割
で段階的に除外し AUC (全体+位相別) が落ちないか実測する:

| 構成 | 内容 |
|---|---|
| A | 旧47列 (ベースライン) |
| B | 旧47列 + W12新5列 |
| C | B から DEAD_ZERO_OR_NEG_5 (貢献ゼロ/負の5列) を除外 |
| D | C から NEARLY_DEAD_10 (ほぼ死の10列) もさらに除外 (計15列削減) |
| E | B から `ojama_margin` だけを単独で除外 (W12容量交互作用の代替列のため
    C/Dとは独立に個別判定できるよう分離、user指示「余力があれば」) |

「消さずに判定に使わないようにする」= 除外リスト方式のため、収集パイプライン
(`build_labeled_win_from_npz.py` 側の列生成) 自体はどの列も一切削除しない
(このスクリプトは学習時の特徴量選択のみを扱う)。

入力CSVは `scripts/_p5_convert85_2026-08-16.py` が生成する
`data/verify/labeled_win_w12_85_2026-08-16/labeled_win_w12_85.csv`
(真値npz85本限定、既存148本CSVは一切変更しない)。
"""
from __future__ import annotations

import os

# 63本再収集ジョブ (10並列、既に高負荷) との競合を避けるため、numpy/sklearn の
# 内部スレッド数を明示的に絞る (import前に環境変数で指定する必要がある、
# HistGradientBoostingClassifier はデフォルトで全コアのOpenMPスレッドを使う)。
# CPU_LIMIT_ENV_VARS 直下のコメント参照。
_CPU_LIMIT: str = "3"
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, _CPU_LIMIT)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

DEFAULT_CSV: str = "data/verify/labeled_win_w12_85_2026-08-16/labeled_win_w12_85.csv"
DEFAULT_OUT_DIR: str = "data/verify/retrain85_w12_2026-08-16"

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
# 63本再収集ジョブ (10並列) と競合しないよう permutation_importance の内部
# 並列度を明示的に絞る (n_jobs=-1=全コア利用は既存 _retrain148 スクリプトの
# 値をそのまま流用すると学習データ量が少ない85本でもCPU競合で著しく低速化
# することを実測 [4000行のsmokeテストで13分ハングを確認、n_jobs=-1が原因]。
# 本ファイルはP5専用の新規スクリプトのため既存互換は不要、3に固定する)。
PERM_N_JOBS: int = 3

META_COLS: frozenset = frozenset({
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side", "won",
})
SOURCE_COLS: frozenset = frozenset({"all_clear_source", "ojama_source"})

# W12根治 (2026-08-16) で追加した新5列 (build_labeled_win_from_npz.py の
# OJAMA_TRUTH_COLUMNS/PAIR_INTERACTION_COLUMNS 末尾追加分と同一)。
W12_NEW_COLS: tuple[str, ...] = (
    "ojama_net_balance_uncapped", "ojama_forecast_uncapped",
    "ojama_forecast_log", "ojama_forecast_progress_interaction",
    "color_forecast_ratio_own",
)

# ============================
# 指標整理アブレーション (2026-08-16 追加依頼)
# ============================
# 148本モデル (`retrain148_2026-08-14/permutation_importance_full.csv`) で
# 貢献ゼロまたは負だった5列 (削除の有力候補)。saturation_chain_upper は
# 「飽和天井は空き空間量と同じで無相関」撤退済みなのに列だけ残っている
# もの (project_saturation_ceiling_untrustworthy_2026-07-22)。
DEAD_ZERO_OR_NEG_5: tuple[str, ...] = (
    "dig_resistance", "saturation_chain_upper", "main_linked_ratio",
    "multi_color_ignition", "isolated_pair_count",
)

# 同モデルで「ほぼ死」だった10列。ojama_margin は W12 の容量交互作用の代替
# として残す判断をした列のため、C/D をひとまとめに除外するだけでなく E で
# 単独除外も測る (削除するとW12対応と矛盾しうるため個別判断が必要)。
NEARLY_DEAD_10: tuple[str, ...] = (
    "ojama_margin", "simultaneous_pop_richness", "immediate_fire_power",
    "conn_triple_count", "main_linked_pair_count", "diff_conn_pair_count",
    "min_puyos_to_ignite", "opp_all_clear_bonus_pending",
    "buried_hole_count", "ignition_point_count",
)

# `_diag_w12_quantify_2026-08-16.py` と同一のbucket定義 (直接比較のため)。
FORECAST_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (-0.5, 0.5, "0"), (0.5, 11.5, "1-11"), (11.5, 29.5, "12-29"),
    (29.5, 71.5, "30-71"), (71.5, 143.5, "72-143"),
    (143.5, 215.5, "144-215"), (215.5, 1e9, "216+"),
)


def bucket_of(val: float, edges) -> str:
    for lo, hi, name in edges:
        if lo < val <= hi:
            return name
    return "?"


def load_data(csv_path: str) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(csv_path)
    n0 = len(df)
    df = df.dropna(subset=["won"]).reset_index(drop=True)
    df["won"] = df["won"].astype(int)
    print(f"  読込 {time.time()-t0:.1f}秒: {n0}行 -> won欠損除外後 {len(df)}行"
          f" (動画数={df['video_id'].nunique()})")
    return df


def resolve_feature_cols(df: pd.DataFrame, exclude_extra: frozenset = frozenset()) -> list[str]:
    exclude = META_COLS | SOURCE_COLS | exclude_extra
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def compute_match_progress(df: pd.DataFrame) -> np.ndarray:
    max_tsumo = df.groupby(["video_id", "game_idx", "side"])["tsumo"].transform("max")
    max_tsumo = max_tsumo.replace(0, np.nan)
    progress = (df["tsumo"] / max_tsumo).fillna(0.0).clip(0.0, 1.0)
    return progress.values


def run_oof(X, y, groups, n_folds):
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_folds)
    for i, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        m = HistGradientBoostingClassifier(**GBC_PARAMS)
        t0 = time.time()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        n_v = len(np.unique(groups[te]))
        print(f"    fold {i+1}/{n_folds}: train={len(tr)} test={len(te)}"
              f" (動画{n_v}本) 学習{time.time()-t0:.1f}秒")
    return oof


def eval_oof(y, p, label):
    valid = ~np.isnan(p)
    y_v, p_v = y[valid], p[valid]
    auc = float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else float("nan")
    print(f"    [{label}] AUC={auc:.4f} n={valid.sum()}")
    return {"label": label, "auc": auc, "n": int(valid.sum())}


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
        rows.append({"video_id": vid, "auc": float(roc_auc_score(y_v, p_v)), "n": len(y_v)})
    return pd.DataFrame(rows)


def phase_auc_report(y, p, progress, groups):
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
            random_state=PERM_RANDOM_STATE, scoring="roc_auc", n_jobs=PERM_N_JOBS,
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


def forecast_bin_report(df: pd.DataFrame, oof_baseline, oof_new, progress: np.ndarray) -> pd.DataFrame:
    """W12の核心検証: 予告bucket別に 実勝率 vs 旧モデル予測 vs 新モデル予測 を並べる。

    `_diag_w12_quantify_2026-08-16.py` table 1a/1b と同一のbucket・位相定義。
    """
    order = [b[2] for b in FORECAST_BUCKETS]
    work = pd.DataFrame({
        "won": df["won"].values,
        "forecast_uncapped": df["ojama_forecast_uncapped"].values,
        "pred_baseline": oof_baseline,
        "pred_new": oof_new,
        "progress": progress,
    })
    work = work.dropna(subset=["forecast_uncapped"]).copy()
    work["forecast_bucket"] = work["forecast_uncapped"].apply(lambda v: bucket_of(v, FORECAST_BUCKETS))
    work["phase"] = pd.cut(work["progress"], bins=[-0.01, 1 / 3, 2 / 3, 1.01],
                            labels=["序盤", "中盤", "終盤"])

    overall = work.groupby("forecast_bucket").agg(
        actual_win_rate=("won", "mean"),
        pred_baseline_mean=("pred_baseline", "mean"),
        pred_new_mean=("pred_new", "mean"),
        n=("won", "size"),
    ).reindex(order)
    overall.insert(0, "phase", "全体")
    overall = overall.reset_index()

    by_phase = work.groupby(["phase", "forecast_bucket"], observed=True).agg(
        actual_win_rate=("won", "mean"),
        pred_baseline_mean=("pred_baseline", "mean"),
        pred_new_mean=("pred_new", "mean"),
        n=("won", "size"),
    ).reindex(pd.MultiIndex.from_product(
        [["序盤", "中盤", "終盤"], order], names=["phase", "forecast_bucket"],
    )).reset_index()

    combined = pd.concat([overall, by_phase], ignore_index=True)
    return combined


def run_config(label, cols, df, y, groups, progress, n_folds):
    """1構成分 (特徴量リスト固定) の OOF学習・評価・位相別評価をまとめて行う。"""
    print(f"\n=== 構成 {label}: {len(cols)}列 ===")
    X = df[cols].fillna(0.0).values.astype(np.float32)
    oof = run_oof(X, y, groups, n_folds)
    res = eval_oof(y, oof, label)
    pv = per_video_auc(y, oof, groups)
    res["video_median_auc"] = float(pv["auc"].median())
    res["n_features"] = len(cols)
    print(f"  --- 位相別 ({label}) ---")
    res["phase"] = phase_auc_report(y, oof, progress, groups)
    return res, oof


def main():
    parser = argparse.ArgumentParser(
        description="85本 (真値限定) 旧列 vs W12新5列 + 指標整理アブレーション A/B/C/D/E",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--perm-repeats", type=int, default=PERM_N_REPEATS)
    parser.add_argument("--skip-perm", action="store_true")
    parser.add_argument(
        "--skip-e", action="store_true",
        help="構成E (ojama_margin単独除外) をスキップする (時間節約用)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 80)
    print("  W12 P5先行検証 + 指標整理アブレーション (A/B/C/D/E)")
    print("=" * 80)

    print("\n=== 1. データ読み込み ===")
    df = load_data(args.csv)
    missing_new = [c for c in W12_NEW_COLS if c not in df.columns]
    if missing_new:
        print(f"  [FATAL] W12新列が見つからない: {missing_new}")
        return 1
    missing_dead = [c for c in DEAD_ZERO_OR_NEG_5 + NEARLY_DEAD_10 if c not in df.columns]
    if missing_dead:
        print(f"  [FATAL] アブレーション対象列が見つからない: {missing_dead}")
        return 1

    print("\n=== 2. 試合内相対進行率 ===")
    progress = compute_match_progress(df)
    y = df["won"].values.astype(int)
    groups = df["video_id"].values

    cols_a = resolve_feature_cols(df, exclude_extra=frozenset(W12_NEW_COLS))
    cols_b = resolve_feature_cols(df)  # W12新5列を含む全数値列 (旧47列+新5列)
    cols_c = [c for c in cols_b if c not in DEAD_ZERO_OR_NEG_5]
    cols_d = [c for c in cols_c if c not in NEARLY_DEAD_10]
    cols_e = [c for c in cols_b if c != "ojama_margin"]

    configs = [
        ("A_旧47列", cols_a),
        ("B_旧47列+W12新5列", cols_b),
        ("C_Bから貢献ゼロ負5列除外", cols_c),
        ("D_Cからほぼ死10列も除外", cols_d),
    ]
    if not args.skip_e:
        configs.append(("E_Bからojama_margin単独除外", cols_e))

    summary = {
        "n_rows": len(df), "n_videos": int(df["video_id"].nunique()),
        "n_features": {label: len(cols) for label, cols in configs},
    }

    results = {}
    oofs = {}
    for label, cols in configs:
        res, oof = run_config(label, cols, df, y, groups, progress, args.n_folds)
        results[label] = res
        oofs[label] = oof
    summary["configs"] = results

    delta_auc_w12 = results["B_旧47列+W12新5列"]["auc"] - results["A_旧47列"]["auc"]
    print(f"\n  W12新5列の純増分 dAUC (B-A, プール) = {delta_auc_w12:+.4f}")
    summary["delta_auc_w12_b_minus_a"] = delta_auc_w12

    delta_auc_c = results["C_Bから貢献ゼロ負5列除外"]["auc"] - results["B_旧47列+W12新5列"]["auc"]
    print(f"  貢献ゼロ/負5列削除の影響 dAUC (C-B) = {delta_auc_c:+.4f}")
    summary["delta_auc_c_minus_b"] = delta_auc_c

    delta_auc_d = results["D_Cからほぼ死10列も除外"]["auc"] - results["C_Bから貢献ゼロ負5列除外"]["auc"]
    print(f"  ほぼ死10列追加削除の影響 dAUC (D-C) = {delta_auc_d:+.4f}")
    summary["delta_auc_d_minus_c"] = delta_auc_d

    if "E_Bからojama_margin単独除外" in results:
        delta_auc_e = results["E_Bからojama_margin単独除外"]["auc"] - results["B_旧47列+W12新5列"]["auc"]
        print(f"  ojama_margin単独除外の影響 dAUC (E-B) = {delta_auc_e:+.4f}")
        summary["delta_auc_e_minus_b"] = delta_auc_e

    if not args.skip_perm:
        print("\n=== permutation importance (構成B: 新列モデル) ===")
        perm_df = compute_perm_importance(
            df[cols_b].fillna(0.0).values.astype(np.float32),
            y, groups, cols_b, args.n_folds, args.perm_repeats,
        )
        perm_df.to_csv(out_dir / "permutation_importance_new.csv", index=False)
        print("\n  上位15列:")
        for _, r in perm_df.head(15).iterrows():
            print(f"    {int(r['rank']):>3}. {r['feature']:<40} "
                  f"{r['importance_mean']:+.6f} (+-{r['importance_std']:.6f})")
        w12_rank = perm_df[perm_df["feature"].isin(W12_NEW_COLS)]
        print(f"\n  W12新5列の順位 (全{len(perm_df)}列中):")
        for _, r in w12_rank.sort_values("rank").iterrows():
            print(f"    {int(r['rank']):>3}. {r['feature']:<40} {r['importance_mean']:+.6f}")
        summary["w12_col_ranks"] = w12_rank.set_index("feature")["rank"].to_dict()
        summary["w12_col_importance"] = w12_rank.set_index("feature")["importance_mean"].to_dict()
    else:
        print("\n=== permutation importance: --skip-perm によりスキップ ===")

    print("\n=== W12核心検証: 予告bucket別 実勝率 vs 構成A/B モデル予測 ===")
    fb_report = forecast_bin_report(df, oofs["A_旧47列"], oofs["B_旧47列+W12新5列"], progress)
    fb_report.to_csv(out_dir / "forecast_bin_report.csv", index=False)
    print(fb_report.to_string(index=False))

    total_sec = time.time() - t_start
    summary["total_wall_seconds"] = total_sec
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"  全工程完了。総所要時間: {total_sec/60:.1f}分")
    print(f"  結果一式: {out_dir}")
    print("P5_RETRAIN85_DONE")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
