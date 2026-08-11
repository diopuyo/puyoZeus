"""序盤の予測ターゲット 2方式比較 (2026-08-11、読み取り専用検証)。

## 背景
現行 (構成A) は全位相で「最終勝敗」を直接予測し、序盤 AUC は 0.58 と弱い。
代替案 (構成B、ブートストラップ方式) は「序盤特徴量→中盤時点の有利確率
(M_mid の予測値)」をソフトターゲットとして学習し、その予測値で最終勝敗の
AUC を測る。本スクリプトは **どちらが有望かの数字出し** のみを行う
(採否は user が判断する。既存ファイルは一切変更しない、新規スクリプトのみ)。

## 構成A (現行と同等の再現)
序盤ペアで「最終勝敗」を直接学習 (HistGBC、動画固定 fold の OOF)。

## 構成B (ブートストラップ)
1. 中盤ペアで「最終勝敗」を学習した M_mid を OOF で作る。
2. 各「試合」の中盤帯の最初のペアにおける M_mid の OOF 予測値を、
   同じ試合の序盤ペア全てにソフトターゲットとして付与する。
3. 序盤ペアで「そのソフトターゲット」を回帰学習する (HistGBR)。
4. 予測値 (連続値) で **最終勝敗** の AUC を測る (自己参照にしない)。

## 試合区切りについて (重要な注意)
既存 CSV の game_idx は full窓/mid窓が重複するため試合特定に使えない
(scripts/model_indicator_win.py の pair_sides_for_win docstring 参照)。
本スクリプトは検証専用として、tsumo (手数カウンタ) が大きい値から
0近辺へ急落する箇所を「新しい試合の開始」とみなす簡易ヒューリスティック
で試合区切りを推定する (RESET_TSUMO_LOW / RESET_TSUMO_PREV_MIN、実データの
目視観測に基づく定数、本番の WIN★パネル方式の試合境界検知とは異なる近似)。
このヒューリスティックの精度は未検証であり、本スクリプトの限界として
レポートに明記する。

## リーク防止 (最重要)
video_id の全体集合を KFold(shuffle) で N_FOLDS 個に固定分割し
(_build_video_fold_map)、構成A・M_mid・構成B の3モデル全てに
**同一の fold 割当て** を使う。これにより:
  - M_mid はある動画を学習に使ったfoldでは、その動画の中盤帯を
    絶対に予測しない (OOF)。
  - 構成Bのソフトターゲットは常に「その動画を学習に使っていない
    M_mid」から得られる (試合単位ではなく動画単位で徹底)。
  - 構成Bの学習自体も、最終評価fold の動画を学習に使わない。
最終評価軸は必ず実勝敗 (won_1p) の AUC とし、ソフトターゲート値との
自己整合 (自己参照) では評価しない。

## 出力
- コンソールレポート (構成A vs B の序盤AUC、動画クラスタブートストラップCI)
- data/verify/early_bootstrap_2026-08-11/ 配下に TSV 保存
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold  # noqa: E402

from src.console_init import init_console  # noqa: E402
from scripts.exchange_meter_eval_harness import (  # noqa: E402
    bootstrap_diff_ci_by_video,
    exact_auc,
)
from scripts.model_indicator_win import (  # noqa: E402
    GBC_PARAMS,
    META_COLS,
    N_FOLDS,
    REDUNDANT_COLS,
    TSUMO_EARLY_RATIO,
    TSUMO_LATE_RATIO,
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
)

# =============================================================================
# 定数
# =============================================================================

# 最新 combined66 (193,623行、66動画) を使用 (指示された固定パス)
LABELED_CSV_PATH: str = (
    "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"
)

# ペアリング最大時刻差 (秒、model_indicator_win.DEFAULT_MAX_TDIFF と同値)
MAX_TDIFF_SEC: float = 1.0

# 試合区切りヒューリスティック定数 (実データ目視: リセット時は必ず tsumo<=4、
# 直前値は常に>=14 だった。安全マージンを取って LOW=5 / PREV_MIN=10 とする)
RESET_TSUMO_LOW: float = 5.0
RESET_TSUMO_PREV_MIN: float = 10.0

# 動画 fold 分割 (KFold shuffle、乱数固定で再現性確保)
FOLD_RANDOM_STATE: int = 42

# HistGradientBoostingRegressor パラメータ (GBC_PARAMS と同構成、回帰版)
REG_PARAMS: dict = {
    "max_iter": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "random_state": 42,
}

# 出力先ディレクトリ
OUT_DIR: Path = Path("data/verify/early_bootstrap_2026-08-11")


# =============================================================================
# 試合区切り推定 (検証専用の簡易ヒューリスティック)
# =============================================================================

def assign_match_id(paired: pd.DataFrame) -> pd.Series:
    """tsumo の急落 (>=PREV_MIN から <=LOW へ) を試合開始とみなし、
    "{video_id}::{試合連番}" 形式の match_id を返す (paired の index に整列)。
    """
    match_ids: np.ndarray = np.empty(len(paired), dtype=object)
    order = paired.sort_values(["video_id_1p", "t_sec_1p"]).index.to_numpy()
    current_video = None
    current_match_no = -1
    prev_tsumo: float | None = None
    for pos in order:
        vid = paired.at[pos, "video_id_1p"]
        tsumo = float(paired.at[pos, "tsumo_1p"])
        if vid != current_video:
            current_video = vid
            current_match_no = 0
        elif (
            prev_tsumo is not None
            and tsumo <= RESET_TSUMO_LOW
            and prev_tsumo >= RESET_TSUMO_PREV_MIN
        ):
            current_match_no += 1
        match_ids[pos] = f"{vid}::{current_match_no}"
        prev_tsumo = tsumo
    return pd.Series(match_ids, index=paired.index, name="match_id")


# =============================================================================
# 位相マスク (model_indicator_win.run_phase_models と同じ手数三分位)
# =============================================================================

def phase_masks_and_bounds(
    paired: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """序盤/中盤の bool マスクと分位境界 (q33, q67) を返す。"""
    tsumo_vals = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(tsumo_vals, TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(tsumo_vals, TSUMO_LATE_RATIO))
    early_mask = tsumo_vals <= q33
    mid_mask = (tsumo_vals > q33) & (tsumo_vals <= q67)
    return early_mask, mid_mask, q33, q67


# =============================================================================
# 動画 fold 割当て (構成A / M_mid / 構成B で共通利用、リーク防止の要)
# =============================================================================

def build_video_fold_map(
    unique_videos: list[str], n_folds: int, random_state: int,
) -> dict[str, int]:
    """動画単位で N_FOLDS に固定分割する (KFold shuffle、決定論的)。"""
    videos_sorted = sorted(unique_videos)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_map: dict[str, int] = {}
    for fold_idx, (_, test_pos) in enumerate(kf.split(videos_sorted)):
        for pos in test_pos:
            fold_map[videos_sorted[pos]] = fold_idx
    return fold_map


# =============================================================================
# 外部固定 fold での OOF 予測 (分類 / 回帰)
# =============================================================================

def oof_classifier_fixed_folds(
    X: np.ndarray, y: np.ndarray, fold_assign: np.ndarray, n_folds: int,
) -> np.ndarray:
    """外部で確定した fold 割当てで OOF 確率 (1P勝ち) を返す。"""
    oof = np.full(len(y), np.nan)
    for k in range(n_folds):
        test_idx = np.where(fold_assign == k)[0]
        train_idx = np.where(fold_assign != k)[0]
        if len(test_idx) == 0 or len(np.unique(y[train_idx])) < 2:
            continue
        model = HistGradientBoostingClassifier(**GBC_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof


def oof_regressor_fixed_folds(
    X: np.ndarray, y: np.ndarray, fold_assign: np.ndarray, n_folds: int,
) -> np.ndarray:
    """外部で確定した fold 割当てで OOF 回帰予測 (連続ソフトターゲット) を返す。"""
    oof = np.full(len(y), np.nan)
    for k in range(n_folds):
        test_idx = np.where(fold_assign == k)[0]
        train_idx = np.where(fold_assign != k)[0]
        if len(test_idx) == 0 or len(train_idx) < 10:
            continue
        model = HistGradientBoostingRegressor(**REG_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X[test_idx])
    return oof


# =============================================================================
# 指標列名の抽出 (model_indicator_win._get_indicator_cols と同ロジック、
# private シンボルを跨いで import しないためここでは重複を許容)
# =============================================================================

def indicator_cols_for(paired: pd.DataFrame) -> list[str]:
    """分析対象の指標列名 (接尾辞なし) を返す。"""
    all_exclude = META_COLS | REDUNDANT_COLS
    result: list[str] = []
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


# =============================================================================
# ソフトターゲット付与 (試合ごとの中盤帯 最初のペア)
# =============================================================================

def build_soft_target_map(
    paired: pd.DataFrame, mid_mask: np.ndarray, oof_mid: np.ndarray,
) -> dict[str, float]:
    """試合ごとに「中盤帯の最初のペア」の M_mid OOF 予測値を辞書で返す。"""
    mid_idx = paired.index[mid_mask]
    mid_df = pd.DataFrame({
        "match_id": paired.loc[mid_idx, "match_id"].values,
        "t_sec": paired.loc[mid_idx, "t_sec_1p"].values,
        "oof": oof_mid,
    }, index=mid_idx)
    mid_df = mid_df.dropna(subset=["oof"])
    mid_df_sorted = mid_df.sort_values("t_sec")
    earliest = mid_df_sorted.groupby("match_id").first()
    return earliest["oof"].to_dict()


# =============================================================================
# メイン実験
# =============================================================================

def main() -> int:
    init_console()
    print("=" * 80)
    print("  序盤の予測ターゲット 2方式比較 (2026-08-11、読み取り専用検証)")
    print("=" * 80)
    print(f"  データ: {LABELED_CSV_PATH}")
    print()

    # 1. 読み込み + ペアリング (既存資産をそのまま再利用)
    print("=== 1. データ読み込み + 1P/2P ペアリング ===")
    df = load_labeled_csv(LABELED_CSV_PATH)
    n_videos_raw = df["video_id"].nunique()
    paired = pair_sides_for_win(df, MAX_TDIFF_SEC)
    if len(paired) == 0:
        print("[ERROR] ペアが成立しなかった。")
        return 1

    # 2. 試合区切り推定 (簡易ヒューリスティック、限界は docstring 参照)
    print("\n=== 2. 試合区切り推定 (簡易ヒューリスティック) ===")
    paired["match_id"] = assign_match_id(paired)
    n_matches = paired["match_id"].nunique()
    match_sizes = paired.groupby("match_id").size()
    print(f"  推定試合数: {n_matches} (動画 {n_videos_raw} 本)")
    print(f"  試合あたりペア数: 中央値={match_sizes.median():.0f}"
          f"  最小={match_sizes.min()}  最大={match_sizes.max()}")
    print("  [注意] tsumo急落ベースの近似であり本番の試合境界検知とは異なる"
          "(精度未検証、本スクリプト限定の簡易法)")

    # 3. 位相マスク (手数三分位、既存モデルと同一の定義)
    print("\n=== 3. 位相マスク (手数三分位) ===")
    early_mask, mid_mask, q33, q67 = phase_masks_and_bounds(paired)
    print(f"  序盤<=手数{q33:.0f}  中盤 手数{q33:.0f}-{q67:.0f}")
    print(f"  序盤ペア数={early_mask.sum()}  中盤ペア数={mid_mask.sum()}")

    # 4. 動画 fold 固定割当て (構成A/M_mid/構成B で共通利用、リーク防止の要)
    print("\n=== 4. 動画 fold 固定割当て ===")
    unique_videos = list(paired["video_id_1p"].unique())
    fold_map = build_video_fold_map(unique_videos, N_FOLDS, FOLD_RANDOM_STATE)
    fold_assign = np.array(
        [fold_map[v] for v in paired["video_id_1p"].values], dtype=int
    )
    for k in range(N_FOLDS):
        n_v = sum(1 for v in fold_map.values() if v == k)
        print(f"  fold {k}: 動画 {n_v} 本")

    # 5. 特徴量構築 (全ペア共通、位相ごとにマスクで切り出す)
    print("\n=== 5. 特徴量構築 ===")
    indicator_cols = indicator_cols_for(paired)
    feat_df = build_features(paired, indicator_cols)
    X_all = feat_df.fillna(0.0).values.astype(float)
    y_all = paired["won_1p"].astype(int).values
    print(f"  指標ベース列数={len(indicator_cols)}  特徴量数={X_all.shape[1]}")

    # 6. 構成A: 序盤ペアで最終勝敗を直接学習 (現行と同等の再現)
    print("\n=== 6. 構成A (現行): 序盤ペアで最終勝敗を直接学習 ===")
    oof_a_early = oof_classifier_fixed_folds(
        X_all[early_mask], y_all[early_mask], fold_assign[early_mask], N_FOLDS
    )
    valid_a = ~np.isnan(oof_a_early)
    auc_a_all_early = exact_auc(y_all[early_mask][valid_a], oof_a_early[valid_a])
    print(f"  構成A 序盤AUC (全序盤ペア対象) = {auc_a_all_early:.4f}"
          f"  (n={int(valid_a.sum())})")

    # 7. M_mid: 中盤ペアで最終勝敗を学習 (OOF)
    print("\n=== 7. M_mid: 中盤ペアで最終勝敗を学習 (OOF) ===")
    oof_mid = oof_classifier_fixed_folds(
        X_all[mid_mask], y_all[mid_mask], fold_assign[mid_mask], N_FOLDS
    )
    valid_mid = ~np.isnan(oof_mid)
    auc_mid_quality = exact_auc(y_all[mid_mask][valid_mid], oof_mid[valid_mid])
    print(f"  M_mid 中盤AUC (ソフトターゲットの質) = {auc_mid_quality:.4f}"
          f"  (n={int(valid_mid.sum())})")

    # 8. ソフトターゲット付与 (試合ごとに中盤帯 最初のペアの M_mid OOF値)
    print("\n=== 8. ソフトターゲット付与 (試合単位) ===")
    soft_target_map = build_soft_target_map(paired, mid_mask, oof_mid)
    early_idx = paired.index[early_mask]
    early_match_id = paired.loc[early_idx, "match_id"]
    soft_target_early = early_match_id.map(soft_target_map)
    coverage_mask = soft_target_early.notna().values
    coverage_rate = float(coverage_mask.mean())
    print(f"  序盤ペアの中盤ソフトターゲット適用可能率 = {coverage_rate:.1%}"
          f"  ({int(coverage_mask.sum())}/{len(coverage_mask)})")

    # 9. 構成B: 序盤特徴量でソフトターゲットを回帰学習 → 最終勝敗AUCで評価
    print("\n=== 9. 構成B (ブートストラップ): ソフトターゲット回帰 ===")
    X_early = X_all[early_mask]
    fold_early = fold_assign[early_mask]
    y_final_early = y_all[early_mask]

    X_covered = X_early[coverage_mask]
    fold_covered = fold_early[coverage_mask]
    y_final_covered = y_final_early[coverage_mask]
    soft_covered = soft_target_early.values[coverage_mask].astype(float)
    oof_a_covered = oof_a_early[coverage_mask]

    oof_b_soft = oof_regressor_fixed_folds(
        X_covered, soft_covered, fold_covered, N_FOLDS
    )
    valid_b = ~np.isnan(oof_b_soft)
    auc_b_covered = exact_auc(y_final_covered[valid_b], oof_b_soft[valid_b])
    valid_a_covered = ~np.isnan(oof_a_covered)
    auc_a_covered = exact_auc(
        y_final_covered[valid_a_covered], oof_a_covered[valid_a_covered]
    )
    print(f"  構成A 序盤AUC (適用可能subset限定、フェア比較用) = "
          f"{auc_a_covered:.4f}")
    print(f"  構成B 序盤AUC (適用可能subset、最終勝敗基準)     = "
          f"{auc_b_covered:.4f}")

    # 10. 動画クラスタブートストラップ CI (同一 held-out population で A vs B)
    print("\n=== 10. 動画クラスタブートストラップ CI (構成B − 構成A) ===")
    both_valid = valid_a_covered & valid_b
    video_ids_covered = paired.loc[early_idx[coverage_mask], "video_id_1p"].values
    ci = bootstrap_diff_ci_by_video(
        metric_fn=exact_auc,
        video_ids=video_ids_covered[both_valid],
        arrays_a={
            "y_true": y_final_covered[both_valid],
            "y_score": oof_b_soft[both_valid],
        },
        arrays_b={
            "y_true": y_final_covered[both_valid],
            "y_score": oof_a_covered[both_valid],
        },
    )
    print(f"  差分 (構成B − 構成A) = {ci.point:+.4f}"
          f"  95%CI [{ci.ci_low:+.4f}, {ci.ci_high:+.4f}]"
          f"  (n_resamples={ci.n_resamples})")

    # 11. 結論
    print("\n=== 11. 結論 ===")
    if ci.ci_low > 0.0:
        verdict = (
            "構成B が構成A を有意に上回る (CI が 0 を跨がない)。"
            "投資価値ありと見立てられる。"
        )
    elif ci.ci_high < 0.0:
        verdict = (
            "構成A が構成B を有意に上回る (CI が 0 を跨がない)。"
            "ブートストラップ方式への切替は現時点で推奨されない。"
        )
    else:
        verdict = (
            "CI が 0 を跨ぐため優劣は確認できず (正直な報告)。"
            "適用可能率とソフトターゲットの質も併せて判断材料とすること。"
        )
    print(f"  {verdict}")
    print(f"  ソフトターゲットの質 (M_mid中盤AUC) = {auc_mid_quality:.4f}"
          f"  適用可能率 = {coverage_rate:.1%}")
    if coverage_rate < 0.5:
        print("  [注意] 適用可能率が5割未満: 序盤の半分以上の試合が中盤帯まで"
              "到達する前に終わっている (短期決着) か、試合区切り推定の精度"
              "不足の可能性がある。数字を鵜呑みにせず試合区切りの妥当性を"
              "別途確認すべき。")

    # 12. TSV 保存
    print("\n=== 12. TSV 保存 ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {"metric": "n_videos", "value": n_videos_raw},
        {"metric": "n_matches_estimated", "value": n_matches},
        {"metric": "tsumo_q33", "value": q33},
        {"metric": "tsumo_q67", "value": q67},
        {"metric": "n_early_pairs_total", "value": int(early_mask.sum())},
        {"metric": "n_mid_pairs_total", "value": int(mid_mask.sum())},
        {"metric": "auc_construction_a_all_early", "value": auc_a_all_early},
        {"metric": "auc_mid_model_quality", "value": auc_mid_quality},
        {"metric": "coverage_rate_early_to_mid", "value": coverage_rate},
        {"metric": "n_early_pairs_covered", "value": int(coverage_mask.sum())},
        {"metric": "auc_construction_a_covered_subset", "value": auc_a_covered},
        {"metric": "auc_construction_b_covered_subset", "value": auc_b_covered},
        {"metric": "diff_b_minus_a_point", "value": ci.point},
        {"metric": "diff_b_minus_a_ci_low", "value": ci.ci_low},
        {"metric": "diff_b_minus_a_ci_high", "value": ci.ci_high},
        {"metric": "diff_n_resamples", "value": ci.n_resamples},
    ]
    summary_path = OUT_DIR / "auc_summary.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    print(f"  保存: {summary_path}")

    match_diag_path = OUT_DIR / "match_segmentation_diagnostics.tsv"
    match_sizes.rename("n_pairs").reset_index().to_csv(
        match_diag_path, sep="\t", index=False
    )
    print(f"  保存: {match_diag_path}")

    fold_diag_path = OUT_DIR / "video_fold_assignment.tsv"
    pd.DataFrame(
        [{"video_id": v, "fold": k} for v, k in fold_map.items()]
    ).to_csv(fold_diag_path, sep="\t", index=False)
    print(f"  保存: {fold_diag_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
