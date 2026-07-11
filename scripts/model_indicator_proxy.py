"""指標 v2 -- 多変量モデルで優勢proxy の組み合わせ重要度を測る。

## 手法
- GroupKFold (video_id 単位) で out-of-fold 予測 → リーク防止
- HistGradientBoostingRegressor (scikit-learn 1.8.0)
- Permutation Importance (OOF fold 全体で算出)
- 盤面密度系のみ vs 全指標の OOF R² 比較 (火力系の ΔR² を測定)
- 位相別 (序盤/中盤/終盤) モデルで指標重要度を比較

## proxy 定義 (analyze_indicator_proxy.py と同一)
    proxy = 0.7 * z(ojama_net_balance_raw) + 0.3 * z(death_margin_raw_1p - death_margin_raw_2p)

## 使い方
    python -m scripts.model_indicator_proxy --study data/indicators_v2/study
    python -m scripts.model_indicator_proxy --study data/indicators_v2/study --out data/indicators_v2/model_importance.csv

## 注意
- proxy != win (真ターゲットは勝敗。proxy はお邪魔net収支+窒息余裕の代理信号)
- 20本 (各動画のfull + mid の2ファイル) だが video_id は10本
- GroupKFold は video_id でグループ化 (mid/full 混在でも同一動画が train/test に跨がらない)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

# =============================================================================
# 定数
# =============================================================================

# GroupKFold の fold 数 (video_id 数が10なので最大10。5で相互検証)
N_FOLDS: int = 5

# proxy 重み (analyze_indicator_proxy.py と同一)
PROXY_W_OJAMA: float = 0.7
PROXY_W_DEATH: float = 0.3

# ペアリング最大時刻差 (秒)
DEFAULT_MAX_TDIFF: float = 1.0

# 手数三分位境界
TSUMO_EARLY_RATIO: float = 0.33
TSUMO_LATE_RATIO: float = 0.67

# proxy 構成要素 (自明予測を防ぐため除外)
PROXY_COMPONENTS: frozenset[str] = frozenset([
    "ojama_net_balance", "ojama_net_balance_raw",
    "ojama_forecast", "ojama_forecast_raw",
    "death_margin", "death_margin_raw",
    "death_margin_neighbor", "death_margin_neighbor_raw",
])

# absorption_capacity は board_puyo_total と完全重複のため除外
REDUNDANT_COLS: frozenset[str] = frozenset(["absorption_capacity", "absorption_capacity_raw"])

# 非数値・メタ・raw 列 (特徴量対象外)
META_COLS: frozenset[str] = frozenset([
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side",
    "reach_fire_power_source", "chain_duration_source",
    "reach_fire_power_max_chain",
])

# 盤面密度系指標 (「密度系のみ」モデルで使う)
DENSITY_INDICATORS: frozenset[str] = frozenset([
    "board_puyo_total",
    "board_color_puyo_total",
    "board_ojama_count",
    "max_column_height",
    "column_bumpiness",
])

# 火力系指標
FIRE_INDICATORS: frozenset[str] = frozenset([
    "current_max_chain",
    "immediate_fire_power",
    "reach_fire_power",
    "chain_efficiency",
    "min_puyos_to_ignite",
    "second_chain_potential",
])

# HistGBM パラメータ (CPU 軽負荷 = 小規模)
GBM_PARAMS: dict = {
    "max_iter": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "random_state": 42,
    "early_stopping": False,  # OOF 評価に集中
}

# permutation importance の繰り返し回数
PERM_N_REPEATS: int = 20
PERM_RANDOM_STATE: int = 42


# =============================================================================
# データ読み込み (corr_*.csv を除外)
# =============================================================================

def load_study_csvs(study_dir: str) -> pd.DataFrame:
    """study ディレクトリの指標 CSV (corr_*を除く) を結合して返す。"""
    paths = sorted(
        p for p in Path(study_dir).glob("*.csv")
        if not p.name.startswith("corr_")
    )
    if not paths:
        raise FileNotFoundError(f"CSV が見つかりません: {study_dir}")
    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        # video_id が欠損している行 (フッター等) を除去
        df = df.dropna(subset=["video_id", "side"])
        dfs.append(df)
        print(f"  読み込み: {p.name}  {len(df)} 行")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  合計: {combined.shape[0]} 行, {combined.shape[1]} 列")
    return combined


# =============================================================================
# 1P/2P ペアリング (analyze_indicator_proxy.py の pair_sides と同ロジック)
# =============================================================================

def pair_sides(df: pd.DataFrame, max_tdiff: float) -> pd.DataFrame:
    """1P/2P を (video_id, game_idx) 内で時刻最近傍マッチ。"""
    p1 = df[df["side"] == "1P"].reset_index(drop=True)
    p2 = df[df["side"] == "2P"].reset_index(drop=True)
    rows: list[dict] = []
    for (vid, gidx), g1 in p1.groupby(["video_id", "game_idx"]):
        g2 = p2[(p2["video_id"] == vid) & (p2["game_idx"] == gidx)].reset_index(drop=True)
        if len(g2) == 0:
            continue
        t2 = g2["t_sec"].values
        for _, r1 in g1.iterrows():
            diffs = np.abs(t2 - float(r1["t_sec"]))
            idx_min = int(diffs.argmin())
            if diffs[idx_min] <= max_tdiff:
                merged_row: dict = {}
                for col in r1.index:
                    merged_row[f"{col}_1p"] = r1[col]
                for col in g2.columns:
                    merged_row[f"{col}_2p"] = g2.iloc[idx_min][col]
                merged_row["t_diff"] = diffs[idx_min]
                rows.append(merged_row)
    paired = pd.DataFrame(rows)
    total_1p = len(p1)
    pair_rate = len(paired) / total_1p if total_1p > 0 else 0.0
    print(
        f"  ペア成立: {len(paired)} / 1P行 {total_1p}"
        f"  (成立率 {pair_rate:.1%},"
        f" t_diff中央値 {paired['t_diff'].median():.2f}秒)"
    )
    return paired


# =============================================================================
# proxy 算出 (analyze_indicator_proxy.py の compute_proxy と同一)
# =============================================================================

def _zscore(s: pd.Series) -> pd.Series:
    """標準化 (std=0 の場合はゼロ返し)。"""
    std = float(s.std(ddof=1))
    if std < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def compute_proxy(paired: pd.DataFrame) -> pd.Series:
    """優勢 proxy を算出して Series で返す。"""
    ojama_raw = paired["ojama_net_balance_raw_1p"].astype(float)
    dm_diff = (
        paired["death_margin_raw_1p"].astype(float)
        - paired["death_margin_raw_2p"].astype(float)
    )
    return PROXY_W_OJAMA * _zscore(ojama_raw) + PROXY_W_DEATH * _zscore(dm_diff)


# =============================================================================
# 特徴量構築
# =============================================================================

def _get_indicator_cols(paired: pd.DataFrame) -> list[str]:
    """分析対象の 1P 側指標列名 (接尾辞 _1p なし) を返す。"""
    all_exclude = META_COLS | PROXY_COMPONENTS | REDUNDANT_COLS
    result: list[str] = []
    for col in paired.columns:
        if not col.endswith("_1p"):
            continue
        base = col[:-3]  # "_1p" を除去
        if base in all_exclude:
            continue
        if base.endswith("_raw") or base.endswith("_source"):
            continue
        if base == "reach_fire_power_max_chain":
            continue
        if pd.api.types.is_numeric_dtype(paired[col]):
            result.append(base)
    return result


def build_features(paired: pd.DataFrame, indicator_cols: list[str]) -> pd.DataFrame:
    """特徴量 DataFrame を構築する。

    構成:
      - 1P 側の指標 (col_1p)
      - 2P 側の指標 (col_2p)
      - 差分 (col_diff = 1p - 2p)
    """
    feat_rows: dict[str, pd.Series] = {}
    for col in indicator_cols:
        col_1p = f"{col}_1p"
        col_2p = f"{col}_2p"
        if col_1p in paired.columns:
            feat_rows[f"{col}_1p"] = paired[col_1p].astype(float)
        if col_2p in paired.columns:
            feat_rows[f"{col}_2p"] = paired[col_2p].astype(float)
        if col_1p in paired.columns and col_2p in paired.columns:
            feat_rows[f"{col}_diff"] = (
                paired[col_1p].astype(float) - paired[col_2p].astype(float)
            )
    return pd.DataFrame(feat_rows, index=paired.index)


# =============================================================================
# OOF 予測 (GroupKFold)
# =============================================================================

def run_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
) -> tuple[np.ndarray, list[HistGradientBoostingRegressor]]:
    """GroupKFold で OOF 予測を返す。fitted モデルのリストも返す。"""
    oof_preds = np.full(len(y), np.nan)
    models: list[HistGradientBoostingRegressor] = []
    gkf = GroupKFold(n_splits=n_folds)
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]
        model = HistGradientBoostingRegressor(**GBM_PARAMS)
        model.fit(X_tr, y_tr)
        oof_preds[test_idx] = model.predict(X_te)
        models.append(model)
        n_groups_test = len(np.unique(groups[test_idx]))
        print(
            f"    fold {fold_idx + 1}/{n_folds}:"
            f" train={len(train_idx)} test={len(test_idx)}"
            f" (video {n_groups_test}本)"
        )
    return oof_preds, models


# =============================================================================
# Permutation Importance (OOF 全体で算出)
# =============================================================================

def compute_perm_importance(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    n_folds: int,
) -> pd.DataFrame:
    """OOF fold ごとに permutation importance を算出し平均±std を返す。

    fold ごとに test fold のデータで算出し、fold 間の安定性を std で評価する。
    """
    gkf = GroupKFold(n_splits=n_folds)
    importances_per_fold: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        model = HistGradientBoostingRegressor(**GBM_PARAMS)
        model.fit(X_tr, y_tr)
        perm = permutation_importance(
            model, X_te, y_te,
            n_repeats=PERM_N_REPEATS,
            random_state=PERM_RANDOM_STATE,
            scoring="r2",
        )
        # shape: (n_features,) - fold ごとの平均 importance
        importances_per_fold.append(perm.importances_mean)
        print(f"    perm fold {fold_idx + 1}/{n_folds} 完了")

    imp_matrix = np.array(importances_per_fold)  # (n_folds, n_features)
    result = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": imp_matrix.mean(axis=0),
        "importance_std": imp_matrix.std(axis=0, ddof=1),
    })
    result = result.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


# =============================================================================
# 位相別モデル
# =============================================================================

def run_phase_models(
    paired: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    tsumo_q33: float,
    tsumo_q67: float,
) -> dict[str, float]:
    """序盤/中盤/終盤で OOF R² を算出して返す。"""
    tsumo_vals = paired["tsumo_1p"].astype(float).values
    phase_masks = {
        "序盤": tsumo_vals <= tsumo_q33,
        "中盤": (tsumo_vals > tsumo_q33) & (tsumo_vals <= tsumo_q67),
        "終盤": tsumo_vals > tsumo_q67,
    }
    phase_r2: dict[str, float] = {}
    for phase, mask in phase_masks.items():
        X_ph = X[mask]
        y_ph = y[mask]
        groups_ph = groups[mask]
        # グループ数が n_folds 未満の場合は fold を減らす
        n_unique = len(np.unique(groups_ph))
        folds = min(n_folds, max(2, n_unique))
        if len(X_ph) < 20:
            phase_r2[phase] = float("nan")
            continue
        oof_ph, _ = run_oof(X_ph, y_ph, groups_ph, folds)
        valid = ~np.isnan(oof_ph)
        phase_r2[phase] = float(r2_score(y_ph[valid], oof_ph[valid])) if valid.sum() > 0 else float("nan")
        print(f"    {phase}: n={valid.sum()}  OOF R²={phase_r2[phase]:.4f}")
    return phase_r2


# =============================================================================
# 密度系のみ / 全指標 の R² 比較
# =============================================================================

def compare_density_vs_all(
    paired: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    indicator_cols: list[str],
) -> dict[str, float]:
    """盤面密度系のみ / 全指標 / 火力系のみ の OOF R² を返す。"""
    result: dict[str, float] = {}

    def _oof_r2(feature_cols: list[str], label: str) -> float:
        feat_names = [c for c in feature_cols if c in paired.columns]
        if not feat_names:
            print(f"    [{label}] 有効特徴量なし -> nan")
            return float("nan")
        X = paired[feat_names].astype(float).fillna(0.0).values
        oof, _ = run_oof(X, y, groups, n_folds)
        valid = ~np.isnan(oof)
        r2 = float(r2_score(y[valid], oof[valid])) if valid.sum() > 0 else float("nan")
        print(f"    [{label}] n_features={len(feat_names)}  OOF R²={r2:.4f}")
        return r2

    # 盤面密度系: 1p/2p/diff すべて
    density_cols = [
        f"{col}_{suf}"
        for col in DENSITY_INDICATORS
        for suf in ("1p", "2p", "diff")
    ]
    result["density_only"] = _oof_r2(density_cols, "密度系のみ")

    # 火力系: 1p/2p/diff すべて
    fire_cols = [
        f"{col}_{suf}"
        for col in FIRE_INDICATORS
        for suf in ("1p", "2p", "diff")
    ]
    result["fire_only"] = _oof_r2(fire_cols, "火力系のみ")

    # 全指標
    all_feature_cols = [
        f"{col}_{suf}"
        for col in indicator_cols
        for suf in ("1p", "2p", "diff")
        if f"{col}_{suf}" in paired.columns
    ]
    result["all_indicators"] = _oof_r2(all_feature_cols, "全指標")

    # 密度系 + 火力系 (= 交互作用寄与の抽出用)
    density_plus_fire = density_cols + fire_cols
    result["density_plus_fire"] = _oof_r2(density_plus_fire, "密度+火力")

    # 火力の純増分 (ΔR²)
    result["delta_r2_fire"] = result["all_indicators"] - result["density_only"]
    return result


# =============================================================================
# 結果出力
# =============================================================================

def _fmt_r2(v: float) -> str:
    if np.isnan(v):
        return "  n/a  "
    return f"{v:+.4f}"


def print_report(
    oof_r2_all: float,
    r2_comparison: dict[str, float],
    perm_df: pd.DataFrame,
    phase_r2: dict[str, float],
    n_samples: int,
    n_videos: int,
    n_features: int,
    feature_names: list[str],
) -> None:
    """結果をコンソールに出力する。"""
    print()
    print("=" * 80)
    print("  指標 v2 -- 多変量モデル 優勢proxy 重要度分析")
    print("=" * 80)
    print(f"  サンプル数: {n_samples}  動画数: {n_videos}  特徴量数: {n_features}")
    print(f"  モデル: HistGradientBoostingRegressor  GroupKFold(n={N_FOLDS})")
    print()
    print(f"  proxy = {PROXY_W_OJAMA} * z(ojama_net_balance_raw)")
    print(f"        + {PROXY_W_DEATH} * z(death_margin_raw_1p - death_margin_raw_2p)")
    print("  ※ proxy 構成要素・absorption_capacity は除外")
    print()

    # OOF R² サマリ
    print("  ─── OOF R² 比較 (指標セット別) ───")
    print(f"  {'指標セット':<25}  {'OOF R²':>8}")
    print("  " + "-" * 35)
    labels = [
        ("全指標 (1p + 2p + diff)", "all_indicators"),
        ("盤面密度系のみ",           "density_only"),
        ("火力系のみ",                "fire_only"),
        ("密度+火力",                 "density_plus_fire"),
        ("火力の純増分 ΔR²",          "delta_r2_fire"),
    ]
    for label, key in labels:
        print(f"  {label:<25}  {_fmt_r2(r2_comparison.get(key, float('nan'))):>8}")
    print()

    # 位相別 R²
    print("  ─── 位相別 OOF R² ───")
    for phase, r2 in phase_r2.items():
        print(f"  {phase}: {_fmt_r2(r2)}")
    print()

    # permutation importance ランキング (上位 30 件)
    print("  ─── Permutation Importance ランキング (全指標モデル) ───")
    print(
        f"  {'rank':>4}  {'feature':<35}  {'importance':>11}  {'±std':>8}"
    )
    print("  " + "-" * 62)
    for _, row in perm_df.head(30).iterrows():
        sign_mark = "*" if row["importance_mean"] > 0.001 else " "
        print(
            f"  {int(row['rank']):>4}{sign_mark} {row['feature']:<35}"
            f"  {row['importance_mean']:>+11.6f}  ±{row['importance_std']:>7.6f}"
        )

    print()
    print("  ─── 指標カテゴリ別 集計 ───")
    _print_category_summary(perm_df)

    print()
    print("  ─── 結論メモ ───")
    delta = r2_comparison.get("delta_r2_fire", float("nan"))
    density_r2 = r2_comparison.get("density_only", float("nan"))
    all_r2 = r2_comparison.get("all_indicators", float("nan"))
    _print_conclusion(delta, density_r2, all_r2, perm_df)

    print()
    print("  注意事項:")
    print("  - proxy != win: 真ターゲットは勝敗ラベル。proxy は代理信号。")
    print("  - 動画10本 (~18K行)。GroupKFold 5-fold でリーク防止済。")
    print("  - permutation importance は相関特徴量 (多重共線性) で過小評価しうる。")
    print("  - 差分特徴 (diff) を含む → 自己相関を低減し純粋な予測力を測る。")
    print()


def _print_category_summary(perm_df: pd.DataFrame) -> None:
    """特徴量カテゴリ別の合計 importance を出力する。"""
    categories = {
        "火力系 (_1p)":    [f"{c}_1p" for c in FIRE_INDICATORS],
        "火力系 (_2p)":    [f"{c}_2p" for c in FIRE_INDICATORS],
        "火力系 (diff)":   [f"{c}_diff" for c in FIRE_INDICATORS],
        "密度系 (_1p)":    [f"{c}_1p" for c in DENSITY_INDICATORS],
        "密度系 (_2p)":    [f"{c}_2p" for c in DENSITY_INDICATORS],
        "密度系 (diff)":   [f"{c}_diff" for c in DENSITY_INDICATORS],
    }
    imp_map = dict(zip(perm_df["feature"], perm_df["importance_mean"]))
    for cat_name, cols in categories.items():
        total = sum(imp_map.get(c, 0.0) for c in cols)
        print(f"  {cat_name:<20}: sum importance = {total:+.6f}")


def _print_conclusion(
    delta: float,
    density_r2: float,
    all_r2: float,
    perm_df: pd.DataFrame,
) -> None:
    """火力系の寄与に関する自動判定コメントを出力する。"""
    fire_top5 = perm_df[
        perm_df["feature"].str.contains(
            "|".join(FIRE_INDICATORS), regex=True
        )
    ].head(5)
    fire_in_top10 = (perm_df.head(10)["feature"].str.contains(
        "|".join(FIRE_INDICATORS), regex=True
    )).sum()

    print(f"  全指標 OOF R²={all_r2:.4f}  密度系のみ={density_r2:.4f}  火力 ΔR²={delta:.4f}")
    if not np.isnan(delta):
        if delta > 0.02:
            print("  -> 火力系は多変量モデルで有意に追加寄与あり (ΔR² > 0.02)")
        elif delta > 0.005:
            print("  -> 火力系は小幅な追加寄与あり (ΔR² 0.005-0.02)")
        else:
            print("  -> 火力系の追加寄与は軽微 (ΔR² < 0.005)")
    print(f"  上位10指標中で火力系が {fire_in_top10} 件ランクイン")


# =============================================================================
# CSV 保存
# =============================================================================

def save_results(
    perm_df: pd.DataFrame,
    r2_comparison: dict[str, float],
    phase_r2: dict[str, float],
    out_path: str,
) -> None:
    """permutation importance と R² 比較を CSV に保存する。"""
    # 重要度ランキング
    perm_df.to_csv(out_path, index=False)
    print(f"  CSV 保存 (importance): {out_path}")

    # R² サマリ
    r2_path = Path(out_path).with_suffix("").as_posix() + "_r2_summary.csv"
    r2_rows = []
    for key, val in r2_comparison.items():
        r2_rows.append({"metric": key, "value": val})
    for phase, val in phase_r2.items():
        r2_rows.append({"metric": f"phase_r2_{phase}", "value": val})
    pd.DataFrame(r2_rows).to_csv(r2_path, index=False)
    print(f"  CSV 保存 (R² summary): {r2_path}")


# =============================================================================
# メイン
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="指標 v2 多変量モデル proxy 重要度分析")
    parser.add_argument(
        "--study", default="data/indicators_v2/study",
        help="study ディレクトリ (デフォルト: data/indicators_v2/study)",
    )
    parser.add_argument(
        "--max-tdiff", type=float, default=DEFAULT_MAX_TDIFF,
        help=f"ペアリング最大時刻差 秒 (デフォルト: {DEFAULT_MAX_TDIFF})",
    )
    parser.add_argument(
        "--out", default=None,
        help="permutation importance CSV 出力パス",
    )
    parser.add_argument(
        "--no-phase", action="store_true",
        help="位相別モデルをスキップ (高速化)",
    )
    args = parser.parse_args()

    print(f"[model_indicator_proxy] study={args.study} max_tdiff={args.max_tdiff}秒")
    print()

    # 1. データ読み込み
    print("=== 1. データ読み込み ===")
    df = load_study_csvs(args.study)
    n_videos = df["video_id"].nunique()

    # 2. ペアリング
    print("\n=== 2. 1P/2P ペアリング ===")
    paired = pair_sides(df, args.max_tdiff)
    if len(paired) == 0:
        print("[ERROR] ペアが 1 件も成立しませんでした。--max-tdiff を増やしてください。")
        sys.exit(1)

    # 3. proxy 算出
    proxy = compute_proxy(paired)
    y = proxy.values.astype(float)

    # 手数三分位
    tsumo_vals = paired["tsumo_1p"].astype(float)
    tsumo_q33 = float(tsumo_vals.quantile(TSUMO_EARLY_RATIO))
    tsumo_q67 = float(tsumo_vals.quantile(TSUMO_LATE_RATIO))
    print(f"  手数三分位: 序盤 tsumo<={tsumo_q33:.0f}, 中盤 {tsumo_q33:.0f}< tsumo <={tsumo_q67:.0f}, 終盤 tsumo>{tsumo_q67:.0f}")

    # グループ配列 (GroupKFold 用)
    groups = paired["video_id_1p"].values

    # 4. 特徴量構築
    print("\n=== 3. 特徴量構築 ===")
    indicator_cols = _get_indicator_cols(paired)
    print(f"  指標ベース列数: {len(indicator_cols)} -> 特徴量 (1p/2p/diff) = {len(indicator_cols)*3}")
    print(f"  指標一覧: {indicator_cols}")
    feat_df = build_features(paired, indicator_cols)
    # NaN は 0 埋め (HistGBM は NaN 対応しているが permutation で問題が出る場合のため)
    X_all = feat_df.fillna(0.0).values.astype(float)
    feature_names = list(feat_df.columns)

    # 5. 指標セット別 R² 比較
    print("\n=== 4. 指標セット別 OOF R² 比較 ===")
    r2_comparison = compare_density_vs_all(paired, y, groups, N_FOLDS, indicator_cols)

    # 6. Permutation importance (全指標モデル)
    print("\n=== 5. Permutation Importance (全指標モデル) ===")
    perm_df = compute_perm_importance(
        X_all, y, groups, feature_names, N_FOLDS
    )

    # 7. 位相別モデル
    phase_r2: dict[str, float] = {}
    if not args.no_phase:
        print("\n=== 6. 位相別 OOF R² ===")
        phase_r2 = run_phase_models(
            paired, X_all, y, groups, N_FOLDS, tsumo_q33, tsumo_q67
        )

    # 8. レポート出力
    # OOF R² (全指標) を r2_comparison から取得
    oof_r2_all = r2_comparison.get("all_indicators", float("nan"))
    print_report(
        oof_r2_all, r2_comparison, perm_df, phase_r2,
        n_samples=len(paired),
        n_videos=n_videos,
        n_features=len(feature_names),
        feature_names=feature_names,
    )

    # 9. CSV 保存
    if args.out:
        save_results(perm_df, r2_comparison, phase_r2, args.out)


if __name__ == "__main__":
    main()
