"""指標 v2 -- 真ターゲット = 勝敗 (won) で指標重要度を再評価する。

## 背景
model_indicator_proxy.py は「優勢 proxy」(お邪魔net収支+窒息余裕) を目的変数に
しており、火力系が無寄与と判明した。しかし proxy は「今の危険度」であり
火力の将来勝利への貢献を測れない。本スクリプトは真ターゲット=勝敗で再評価する。

## 手法
- 入力: label_win_from_winners.py が生成した labeled_win.csv
  (won 列: 1=このサイドが当該試合に勝つ, 0=負け, NaN=ラベル不能)
- 1P/2P 対応ペアを (video_id, game_idx=試合区間, t_sec 近傍マッチ) で構成
- 目的変数: won_1p (0/1、1Pが勝つなら1)
- モデル:
  A. GradientBoostingClassifier (HistGBC)
  B. LogisticRegression
- GroupKFold (video_id 単位) でリーク防止
- Permutation Importance (OOF fold ごとに算出, fold 平均)
- 比較: proxy での重要度 vs win での重要度 (火力系の変化に注目)

## 使い方
    python -m scripts.model_indicator_win \
        --labeled data/indicators_v2/study/labeled_win.csv \
        --out data/indicators_v2/model_win_importance.csv

## 注意
- proxy 構成要素 (ojama_net_balance/death_margin 等) は今回も特徴量に含める
  (win 予測では除外理由がなく、実際の寄与を確認したい)
- GroupKFold は video_id でグループ化
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.production_config import reorg_removed_indicator_names  # noqa: E402

# =============================================================================
# 定数
# =============================================================================

# GroupKFold の fold 数 (video_id 数10で最大10。5で相互検証)
N_FOLDS: int = 5

# ペアリング最大時刻差 (秒)
DEFAULT_MAX_TDIFF: float = 1.0

# 手数三分位境界
TSUMO_EARLY_RATIO: float = 0.33
TSUMO_LATE_RATIO: float = 0.67

# 非数値・メタ列 (特徴量対象外)
META_COLS: frozenset[str] = frozenset([
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side",
    "reach_fire_power_source", "chain_duration_source",
    "reach_fire_power_max_chain",
    "won",  # 目的変数
])

# absorption_capacity は board_puyo_total と完全重複のため除外。
# saturated_chain_count は current_max_chain と完全一致のため a-1決定
# (2026-08-12) で削除確定済みだが、本タプルへの反映が漏れていた
# (build_labeled_win_from_npz.py にしか反映されていなかった、
# docs/CROSS_CUTTING_AUDIT_2026-08-13.md P2)。2026-08-13 是正: 削除台帳
# `src.production_config.REORG_REMOVED_INDICATORS` を単一情報源とし、
# 台帳の *_raw 版も含めて機械的に合成する (台帳を更新するだけで自動追従する)。
REDUNDANT_COLS: frozenset[str] = frozenset(
    {"absorption_capacity", "absorption_capacity_raw"}
    | reorg_removed_indicator_names()
    | {f"{name}_raw" for name in reorg_removed_indicator_names()}
)

# 火力系指標 (注目対象)
FIRE_INDICATORS: frozenset[str] = frozenset([
    "current_max_chain",
    "immediate_fire_power",
    "reach_fire_power",
    "chain_efficiency",
    "min_puyos_to_ignite",
    "second_chain_potential",
])

# 盤面密度系指標
DENSITY_INDICATORS: frozenset[str] = frozenset([
    "board_puyo_total",
    "board_color_puyo_total",
    "board_ojama_count",
    "max_column_height",
    "column_bumpiness",
])

# 危険度系指標 (proxy 構成要素)
DANGER_INDICATORS: frozenset[str] = frozenset([
    "ojama_net_balance",
    "ojama_forecast",
    "death_margin",
    "death_margin_neighbor",
])

# HistGBC パラメータ
GBC_PARAMS: dict = {
    "max_iter": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "random_state": 42,
    "early_stopping": False,
}

# LogisticRegression パラメータ
LR_PARAMS: dict = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": 42,
    "solver": "lbfgs",
}

# permutation importance
PERM_N_REPEATS: int = 20
PERM_RANDOM_STATE: int = 42

# AUC が有意とみなす閾値
AUC_SIGNIFICANT: float = 0.55


# =============================================================================
# データ読み込み
# =============================================================================

def load_labeled_csv(labeled_path: str) -> pd.DataFrame:
    """labeled_win.csv を読み込んで won 付きの DataFrame を返す。"""
    df = pd.read_csv(labeled_path)
    df = df.dropna(subset=["video_id", "side"])
    n_total = len(df)
    n_labeled = df["won"].notna().sum()
    print(f"  読み込み: {n_total} 行, won ラベル有り: {n_labeled} "
          f"({n_labeled/n_total:.1%})")
    # won 不明行を除外
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)
    print(f"  won ラベル付き行のみ使用: {len(df)} 行")
    return df


# =============================================================================
# ペアリング
# =============================================================================

def pair_sides_for_win(df: pd.DataFrame, max_tdiff: float) -> pd.DataFrame:
    """
    1P/2P を (video_id, t_sec 近傍) でペアリングする。

    重要: study CSV の game_idx は窓内相対インデックスであり、
    full窓 (0-300s) と mid窓 (1200-1560s) の game_idx=0 は別試合を指す。
    そのため game_idx でグループ化せず、t_sec の近傍マッチで同時刻のペアを構成する。

    ペアリング後、won_1p と won_2p が整合していること
    (won_1p + won_2p == 1.0) を確認してフィルタする。

    ペアリング条件:
    - 同一 video_id
    - |t_sec_1p - t_sec_2p| <= max_tdiff
    - won が両者とも非 NaN かつ won_1p + won_2p == 1 (整合チェック)
    """
    p1 = df[df["side"] == "1P"].reset_index(drop=True)
    p2 = df[df["side"] == "2P"].reset_index(drop=True)
    rows: list[dict] = []
    for vid, g1 in p1.groupby("video_id"):
        g2 = p2[p2["video_id"] == vid].reset_index(drop=True)
        if len(g2) == 0:
            continue
        t2 = g2["t_sec"].values
        for _, r1 in g1.iterrows():
            diffs = np.abs(t2 - float(r1["t_sec"]))
            idx_min = int(diffs.argmin())
            if diffs[idx_min] > max_tdiff:
                continue
            r2 = g2.iloc[idx_min]
            # won 整合チェック: 1P won=1 なら 2P won=0、逆も然り
            won1 = r1["won"]
            won2 = r2["won"]
            if pd.isna(won1) or pd.isna(won2):
                continue
            if abs(float(won1) + float(won2) - 1.0) > 0.01:
                # 整合しない (両者が同じ試合を指していない)
                continue
            merged_row: dict = {}
            for col in r1.index:
                merged_row[f"{col}_1p"] = r1[col]
            for col in g2.columns:
                merged_row[f"{col}_2p"] = r2[col]
            merged_row["t_diff"] = diffs[idx_min]
            rows.append(merged_row)
    paired = pd.DataFrame(rows)
    total_1p = len(p1)
    pair_rate = len(paired) / total_1p if total_1p > 0 else 0.0
    print(f"  ペア成立 (won整合チェック後): {len(paired)} / 1P行 {total_1p}"
          f"  (成立率 {pair_rate:.1%},"
          f" t_diff中央値 {paired['t_diff'].median():.2f}秒)")
    return paired


# =============================================================================
# 特徴量構築
# =============================================================================

def _get_indicator_cols(paired: pd.DataFrame) -> list[str]:
    """分析対象の指標列名 (接尾辞なし) を返す。"""
    all_exclude = META_COLS | REDUNDANT_COLS
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
    """1P/2P/diff の特徴量 DataFrame を構築する。"""
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
# OOF 評価 (HistGBC)
# =============================================================================

def run_oof_classifier(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
) -> tuple[np.ndarray, list[HistGradientBoostingClassifier]]:
    """GroupKFold で OOF 確率予測を返す (shape: [n, 2])。"""
    oof_proba = np.full((len(y), 2), np.nan)
    models: list[HistGradientBoostingClassifier] = []
    gkf = GroupKFold(n_splits=n_folds)
    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups)
    ):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]
        model = HistGradientBoostingClassifier(**GBC_PARAMS)
        model.fit(X_tr, y_tr)
        oof_proba[test_idx] = model.predict_proba(X_te)
        models.append(model)
        n_groups_test = len(np.unique(groups[test_idx]))
        print(f"    fold {fold_idx + 1}/{n_folds}:"
              f" train={len(train_idx)} test={len(test_idx)}"
              f" (video {n_groups_test} 本)")
    return oof_proba, models


def run_oof_lr(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
) -> np.ndarray:
    """LogisticRegression の GroupKFold OOF 確率予測 (shape: [n, 2])。"""
    oof_proba = np.full((len(y), 2), np.nan)
    gkf = GroupKFold(n_splits=n_folds)
    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups)
    ):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        lr = LogisticRegression(**LR_PARAMS)
        lr.fit(X_tr_s, y_tr)
        oof_proba[test_idx] = lr.predict_proba(X_te_s)
        n_groups_test = len(np.unique(groups[test_idx]))
        print(f"    fold {fold_idx + 1}/{n_folds}:"
              f" train={len(train_idx)} test={len(test_idx)}"
              f" (video {n_groups_test} 本)")
    return oof_proba


def _eval_oof_proba(
    y: np.ndarray,
    proba: np.ndarray,
    label: str,
) -> tuple[float, float]:
    """AUC と logloss を計算してコンソール出力する。"""
    valid = ~np.isnan(proba[:, 0])
    y_v = y[valid]
    p_v = proba[valid, 1]
    auc = float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else float("nan")
    ll = float(log_loss(y_v, np.column_stack([1 - p_v, p_v])))
    print(f"    [{label}] OOF AUC={auc:.4f}  logloss={ll:.4f}"
          f"  n={valid.sum()}")
    return auc, ll


# =============================================================================
# Permutation Importance
# =============================================================================

def compute_perm_importance_win(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    n_folds: int,
) -> pd.DataFrame:
    """OOF fold ごとに permutation importance (roc_auc) を算出し平均±std を返す。"""
    gkf = GroupKFold(n_splits=n_folds)
    importances_per_fold: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups)
    ):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        model = HistGradientBoostingClassifier(**GBC_PARAMS)
        model.fit(X_tr, y_tr)
        perm = permutation_importance(
            model, X_te, y_te,
            n_repeats=PERM_N_REPEATS,
            random_state=PERM_RANDOM_STATE,
            scoring="roc_auc",
        )
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
# 指標セット別 AUC 比較
# =============================================================================

def _auc_for_cols(
    feat_names: list[str],
    paired: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    label: str,
) -> float:
    """指定列セットで HistGBC OOF AUC を返す。"""
    valid_cols = [c for c in feat_names if c in paired.columns]
    if not valid_cols:
        print(f"    [{label}] 有効特徴量なし -> nan")
        return float("nan")
    X = paired[valid_cols].astype(float).fillna(0.0).values
    oof_proba, _ = run_oof_classifier(X, y, groups, n_folds)
    valid = ~np.isnan(oof_proba[:, 0])
    y_v = y[valid]
    p_v = oof_proba[valid, 1]
    auc = float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else float("nan")
    print(f"    [{label}] n_features={len(valid_cols)}  OOF AUC={auc:.4f}")
    return auc


def compare_indicator_sets(
    paired: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    indicator_cols: list[str],
) -> dict[str, float]:
    """盤面密度系/危険度系/火力系/全指標 の AUC を比較する。"""
    result: dict[str, float] = {}

    def _cols_for_category(cats: frozenset[str]) -> list[str]:
        return [
            f"{col}_{suf}"
            for col in cats
            for suf in ("1p", "2p", "diff")
        ]

    result["density_only"] = _auc_for_cols(
        _cols_for_category(DENSITY_INDICATORS), paired, y, groups, n_folds, "密度系のみ"
    )
    result["danger_only"] = _auc_for_cols(
        _cols_for_category(DANGER_INDICATORS), paired, y, groups, n_folds, "危険度系のみ"
    )
    result["fire_only"] = _auc_for_cols(
        _cols_for_category(FIRE_INDICATORS), paired, y, groups, n_folds, "火力系のみ"
    )

    all_feature_cols = [
        f"{col}_{suf}"
        for col in indicator_cols
        for suf in ("1p", "2p", "diff")
        if f"{col}_{suf}" in paired.columns
    ]
    result["all_indicators"] = _auc_for_cols(
        all_feature_cols, paired, y, groups, n_folds, "全指標"
    )

    # 密度+危険度 (火力なし) ベースライン
    density_danger = _cols_for_category(DENSITY_INDICATORS) + _cols_for_category(DANGER_INDICATORS)
    result["density_plus_danger"] = _auc_for_cols(
        density_danger, paired, y, groups, n_folds, "密度+危険度"
    )

    # 火力の純増分 ΔAUCを計算
    result["delta_auc_fire"] = result["all_indicators"] - result["density_plus_danger"]
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
    """序盤/中盤/終盤で OOF AUC を算出して返す。"""
    tsumo_vals = paired["tsumo_1p"].astype(float).values
    phase_masks = {
        "序盤": tsumo_vals <= tsumo_q33,
        "中盤": (tsumo_vals > tsumo_q33) & (tsumo_vals <= tsumo_q67),
        "終盤": tsumo_vals > tsumo_q67,
    }
    phase_auc: dict[str, float] = {}
    for phase, mask in phase_masks.items():
        X_ph = X[mask]
        y_ph = y[mask]
        groups_ph = groups[mask]
        n_unique = len(np.unique(groups_ph))
        folds = min(n_folds, max(2, n_unique))
        if len(X_ph) < 20 or len(np.unique(y_ph)) < 2:
            phase_auc[phase] = float("nan")
            print(f"    {phase}: データ不足 -> nan")
            continue
        oof_proba, _ = run_oof_classifier(X_ph, y_ph, groups_ph, folds)
        valid = ~np.isnan(oof_proba[:, 0])
        p_v = oof_proba[valid, 1]
        y_v = y_ph[valid]
        auc = float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else float("nan")
        phase_auc[phase] = auc
        print(f"    {phase}: n={valid.sum()}  OOF AUC={auc:.4f}")
    return phase_auc


# =============================================================================
# レポート出力
# =============================================================================

def _fmt(v: float) -> str:
    return "  n/a  " if np.isnan(v) else f"{v:.4f}"


def print_report(
    auc_gbc: float,
    ll_gbc: float,
    auc_lr: float,
    ll_lr: float,
    auc_comparison: dict[str, float],
    perm_df: pd.DataFrame,
    phase_auc: dict[str, float],
    n_samples: int,
    n_videos: int,
    n_features: int,
) -> None:
    """全結果をコンソール出力する。"""
    print()
    print("=" * 80)
    print("  指標 v2 -- win予測モデル 重要度分析 (真ターゲット = 勝敗)")
    print("=" * 80)
    print(f"  サンプル数: {n_samples}  動画数: {n_videos}  特徴量数: {n_features}")
    print(f"  目的変数: won_1p (0/1, 1P がそのゲームに勝つ=1)")
    print(f"  GroupKFold({N_FOLDS} fold) でリーク防止")
    print()

    print("  ─── OOF 評価 ───")
    print(f"  {'モデル':<30}  {'AUC':>6}  {'logloss':>8}")
    print("  " + "-" * 48)
    print(f"  {'HistGradientBoostingClassifier':<30}  {_fmt(auc_gbc):>6}  {_fmt(ll_gbc):>8}")
    print(f"  {'LogisticRegression':<30}  {_fmt(auc_lr):>6}  {_fmt(ll_lr):>8}")
    print()

    print("  ─── 指標セット別 OOF AUC 比較 ───")
    print(f"  {'指標セット':<25}  {'OOF AUC':>8}")
    print("  " + "-" * 35)
    labels = [
        ("全指標",              "all_indicators"),
        ("密度+危険度 (火力なし)", "density_plus_danger"),
        ("危険度系のみ",         "danger_only"),
        ("盤面密度系のみ",       "density_only"),
        ("火力系のみ",           "fire_only"),
        ("火力の純増分 ΔAUC",    "delta_auc_fire"),
    ]
    for lbl, key in labels:
        print(f"  {lbl:<25}  {_fmt(auc_comparison.get(key, float('nan'))):>8}")
    print()

    print("  ─── 位相別 OOF AUC ───")
    for phase, auc in phase_auc.items():
        print(f"  {phase}: {_fmt(auc)}")
    print()

    print("  ─── Permutation Importance ランキング (HistGBC, 上位 30 件) ───")
    print(f"  {'rank':>4}  {'feature':<40}  {'importance':>11}  {'±std':>8}")
    print("  " + "-" * 68)
    for _, row in perm_df.head(30).iterrows():
        sign_mark = "*" if row["importance_mean"] > 0.001 else " "
        print(f"  {int(row['rank']):>4}{sign_mark} {row['feature']:<40}"
              f"  {row['importance_mean']:>+11.6f}  ±{row['importance_std']:>7.6f}")
    print()

    print("  ─── 指標カテゴリ別 集計 ───")
    _print_category_summary(perm_df)
    print()

    print("  ─── 結論 ───")
    _print_conclusion(auc_comparison, perm_df)
    print()

    print("  注意事項:")
    print("  - GroupKFold(video_id) でリーク防止済。")
    print("  - permutation importance は相関特徴量で過小評価しうる。")
    print("  - 動画10本 x 窓内ゲームのみ。カバレッジは label_win_from_winners.py 参照。")
    print("  - AUC=0.5 はランダム予測レベル (= 指標に予測力なし)。")
    print()


def _print_category_summary(perm_df: pd.DataFrame) -> None:
    """指標カテゴリ別の合計 importance を出力する。"""
    categories = {
        "火力系 (1p)":    [f"{c}_1p" for c in FIRE_INDICATORS],
        "火力系 (2p)":    [f"{c}_2p" for c in FIRE_INDICATORS],
        "火力系 (diff)":  [f"{c}_diff" for c in FIRE_INDICATORS],
        "危険度系 (1p)":  [f"{c}_1p" for c in DANGER_INDICATORS],
        "危険度系 (2p)":  [f"{c}_2p" for c in DANGER_INDICATORS],
        "危険度系 (diff)":[f"{c}_diff" for c in DANGER_INDICATORS],
        "密度系 (1p)":    [f"{c}_1p" for c in DENSITY_INDICATORS],
        "密度系 (2p)":    [f"{c}_2p" for c in DENSITY_INDICATORS],
        "密度系 (diff)":  [f"{c}_diff" for c in DENSITY_INDICATORS],
    }
    imp_map = dict(zip(perm_df["feature"], perm_df["importance_mean"]))
    for cat_name, cols in categories.items():
        total = sum(imp_map.get(c, 0.0) for c in cols)
        print(f"  {cat_name:<22}: sum importance = {total:+.6f}")


def _print_conclusion(
    auc_comparison: dict[str, float],
    perm_df: pd.DataFrame,
) -> None:
    """fire 系の win 予測への寄与に関する自動判定コメントを出力する。"""
    all_auc = auc_comparison.get("all_indicators", float("nan"))
    danger_auc = auc_comparison.get("danger_only", float("nan"))
    fire_only_auc = auc_comparison.get("fire_only", float("nan"))
    delta = auc_comparison.get("delta_auc_fire", float("nan"))

    fire_in_top10 = (
        perm_df.head(10)["feature"]
        .str.contains("|".join(FIRE_INDICATORS), regex=True)
    ).sum()

    print(f"  全指標 AUC={_fmt(all_auc)}  危険度系のみ={_fmt(danger_auc)}"
          f"  火力系のみ={_fmt(fire_only_auc)}")
    print(f"  火力の純増分 ΔAUC={_fmt(delta)}")

    if not np.isnan(delta):
        if delta > 0.02:
            print("  -> 火力系は win 予測で有意に追加寄与あり (ΔAUC > 0.02)")
        elif delta > 0.005:
            print("  -> 火力系は win 予測で小幅な追加寄与あり (ΔAUC 0.005-0.02)")
        else:
            print("  -> 火力系の win 予測への純増分は軽微 (ΔAUC < 0.005)")

    if not np.isnan(fire_only_auc):
        if fire_only_auc >= AUC_SIGNIFICANT:
            print(f"  -> 火力系単独 AUC={_fmt(fire_only_auc)} >= {AUC_SIGNIFICANT}: 単独でも予測力あり")
        else:
            print(f"  -> 火力系単独 AUC={_fmt(fire_only_auc)} < {AUC_SIGNIFICANT}: 単独では予測力弱い")

    print(f"  上位10指標中で火力系が {fire_in_top10} 件ランクイン")


# =============================================================================
# CSV 保存
# =============================================================================

def save_results(
    perm_df: pd.DataFrame,
    auc_comparison: dict[str, float],
    phase_auc: dict[str, float],
    out_path: str,
) -> None:
    """permutation importance と AUC 比較を CSV に保存する。"""
    perm_df.to_csv(out_path, index=False)
    print(f"  CSV 保存 (importance): {out_path}")

    auc_path = Path(out_path).with_suffix("").as_posix() + "_auc_summary.csv"
    auc_rows = []
    for key, val in auc_comparison.items():
        auc_rows.append({"metric": key, "value": val})
    for phase, val in phase_auc.items():
        auc_rows.append({"metric": f"phase_auc_{phase}", "value": val})
    pd.DataFrame(auc_rows).to_csv(auc_path, index=False)
    print(f"  CSV 保存 (AUC summary): {auc_path}")


# =============================================================================
# メイン
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="指標 v2 win予測モデル 重要度分析"
    )
    parser.add_argument(
        "--labeled", default="data/indicators_v2/study/labeled_win.csv",
        help="labeled_win.csv パス",
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
    parser.add_argument(
        "--fixed-q33", type=float, default=None,
        help="序盤/中盤境界を手数固定値で上書き (未指定時は従来通り分位点自動算出)",
    )
    parser.add_argument(
        "--fixed-q67", type=float, default=None,
        help="中盤/終盤境界を手数固定値で上書き (未指定時は従来通り分位点自動算出)",
    )
    args = parser.parse_args()

    print(f"[model_indicator_win] labeled={args.labeled}")
    print()

    # 1. データ読み込み
    print("=== 1. データ読み込み ===")
    df = load_labeled_csv(args.labeled)
    n_videos = df["video_id"].nunique()

    # 2. ペアリング
    print("\n=== 2. 1P/2P ペアリング ===")
    paired = pair_sides_for_win(df, args.max_tdiff)
    if len(paired) == 0:
        print("[ERROR] ペアが成立しなかった。--max-tdiff を増やしてください。")
        sys.exit(1)

    # won_1p を目的変数に (1P が勝つ=1)
    y = paired["won_1p"].astype(int).values
    print(f"  won=1 (1P勝ち): {(y==1).sum()}  won=0 (2P勝ち): {(y==0).sum()}")

    # 手数三分位 (--fixed-q33/--fixed-q67 指定時は固定境界で上書き=新旧データ同条件比較用)
    tsumo_vals = paired["tsumo_1p"].astype(float)
    if args.fixed_q33 is not None and args.fixed_q67 is not None:
        tsumo_q33 = float(args.fixed_q33)
        tsumo_q67 = float(args.fixed_q67)
        print(f"  手数境界(固定指定): 序盤<={tsumo_q33:.0f}, 中盤 {tsumo_q33:.0f}-{tsumo_q67:.0f}, "
              f"終盤>{tsumo_q67:.0f}")
    else:
        tsumo_q33 = float(tsumo_vals.quantile(TSUMO_EARLY_RATIO))
        tsumo_q67 = float(tsumo_vals.quantile(TSUMO_LATE_RATIO))
        print(f"  手数三分位(自動算出): 序盤<={tsumo_q33:.0f}, 中盤 {tsumo_q33:.0f}-{tsumo_q67:.0f}, "
              f"終盤>{tsumo_q67:.0f}")

    # グループ配列
    groups = paired["video_id_1p"].values

    # 3. 特徴量構築
    print("\n=== 3. 特徴量構築 ===")
    indicator_cols = _get_indicator_cols(paired)
    print(f"  指標ベース列数: {len(indicator_cols)} -> 特徴量(1p/2p/diff)={len(indicator_cols)*3}")
    feat_df = build_features(paired, indicator_cols)
    X_all = feat_df.fillna(0.0).values.astype(float)
    feature_names = list(feat_df.columns)

    # 4. 指標セット別 AUC 比較
    print("\n=== 4. 指標セット別 OOF AUC 比較 ===")
    auc_comparison = compare_indicator_sets(
        paired, y, groups, N_FOLDS, indicator_cols
    )

    # 5. HistGBC 全指標 OOF 評価
    print("\n=== 5. OOF 評価 (HistGBC, 全指標) ===")
    oof_proba_gbc, _ = run_oof_classifier(X_all, y, groups, N_FOLDS)
    auc_gbc, ll_gbc = _eval_oof_proba(y, oof_proba_gbc, "HistGBC")

    # 6. LogisticRegression OOF 評価
    print("\n=== 6. OOF 評価 (LogisticRegression, 全指標) ===")
    oof_proba_lr = run_oof_lr(X_all, y, groups, N_FOLDS)
    auc_lr, ll_lr = _eval_oof_proba(y, oof_proba_lr, "LR")

    # 7. Permutation importance
    print("\n=== 7. Permutation Importance (HistGBC) ===")
    perm_df = compute_perm_importance_win(
        X_all, y, groups, feature_names, N_FOLDS
    )

    # 8. 位相別モデル
    phase_auc: dict[str, float] = {}
    if not args.no_phase:
        print("\n=== 8. 位相別 OOF AUC ===")
        phase_auc = run_phase_models(
            paired, X_all, y, groups, N_FOLDS, tsumo_q33, tsumo_q67
        )

    # 9. レポート
    print_report(
        auc_gbc, ll_gbc, auc_lr, ll_lr,
        auc_comparison, perm_df, phase_auc,
        n_samples=len(paired),
        n_videos=n_videos,
        n_features=len(feature_names),
    )

    # 10. CSV 保存
    if args.out:
        save_results(perm_df, auc_comparison, phase_auc, args.out)


if __name__ == "__main__":
    main()
