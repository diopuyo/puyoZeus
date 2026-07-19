"""打ち合いラベル予測性能の検証スクリプト (A-2 火力統制版)。

exchange_labels.csv を読み込み、盤面特徴 → 各ターゲットの予測性能を測る。

A-2 追加機能:
  - m1(火力系のみ) vs m2(全特徴) を比較し「残差リフト = AUC(m2)-AUC(m1)」を算出。
  - opp_buried など不均衡ターゲットは PR-AUC(average_precision)も併記。
  - m2 で残差リフト最大のターゲットについて permutation importance Top5 を表示。
  - ターゲット: returned_competitive, returned, opp_buried, net_ojama_sign, won。

モデル: sklearn HistGradientBoostingClassifier / Regressor
holdout: video_id 単位の GroupKFold (5-fold)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import warnings

# スレッド制限
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

# ============================
# 定数定義
# ============================
INPUT_PATH = PROJ_ROOT / "data" / "indicators_v2" / "exchange_labels.csv"
N_SPLITS: int = 5
RANDOM_STATE: int = 42

# A-2: 火力系特徴のプレフィックス+名称リスト(発火側ぶんのみ)
FIRE_POWER_FEAT_NAMES: tuple[str, ...] = (
    "fire_immediate_fire_power",
    "fire_current_max_chain",
    "fire_potential_fire_power",
    "fire_honsen_output",
)

# PR-AUC を追加表示するターゲット(不均衡ターゲット)
PR_AUC_TARGETS: frozenset[str] = frozenset({"opp_buried", "returned_competitive", "returned"})

# permutation importance: 残差リフト最大のターゲット 1件で出す Top N
PERM_IMP_TOP_N: int = 5
PERM_IMP_N_REPEATS: int = 5


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """特徴列名を返す。fire_/opp_/diff_ プレフィックスを持つ列。"""
    meta_cols = {
        "video_id", "game_idx", "t_sec", "fire_side", "phase",
        "won", "net_ojama", "returned", "returned_competitive",
        "opp_buried", "return_window_sec", "approx_fire_chains",
    }
    return [c for c in df.columns if c not in meta_cols]


def _get_fire_power_cols(feat_cols: list[str]) -> list[str]:
    """m1 用: 火力系特徴列名を返す。FIRE_POWER_FEAT_NAMES に一致する列のみ。"""
    return [c for c in feat_cols if c in FIRE_POWER_FEAT_NAMES]


def _auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """クラス単一の場合に nan を返す安全な ROC-AUC 計算。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _prauc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """クラス単一の場合に nan を返す安全な PR-AUC(average_precision)計算。"""
    if len(np.unique(y_true)) < 2 or y_true.sum() == 0:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def _build_model(is_regressor: bool) -> object:
    """HistGBM モデルを共通パラメータで生成する(分類 or 回帰)。"""
    kwargs = dict(max_iter=200, max_depth=4, random_state=RANDOM_STATE, n_iter_no_change=20)
    if is_regressor:
        return HistGradientBoostingRegressor(**kwargs)
    return HistGradientBoostingClassifier(**kwargs)


def _cv_predict_proba(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    is_regressor: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """GroupKFold で OOF 予測スコア・対応ラベルを返す。

    Returns: (oof_scores, oof_labels) — val fold 分のみ結合。
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_scores: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = _build_model(is_regressor)
        model.fit(X[train_idx], y[train_idx])
        if is_regressor:
            pred = model.predict(X[val_idx])
        else:
            pred = model.predict_proba(X[val_idx])[:, 1]
        oof_scores.append(pred)
        oof_labels.append(y[val_idx])
    if not oof_scores:
        return np.array([]), np.array([])
    return np.concatenate(oof_scores), np.concatenate(oof_labels)


def _run_one_target(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    is_regressor: bool,
    want_prauc: bool,
) -> dict[str, float]:
    """1ターゲット・1特徴セットについて AUC / PR-AUC / n を返す辞書。"""
    y_bin = (y > 0).astype(np.int8) if is_regressor else y.astype(np.int8)
    oof_scores, oof_labels = _cv_predict_proba(X, y, groups, n_splits, is_regressor)
    if len(oof_scores) == 0:
        return {"auc": float("nan"), "prauc": float("nan"), "n": 0}
    auc = _auc_safe(oof_labels if is_regressor else oof_labels, oof_scores)
    prauc = _prauc_safe(oof_labels, oof_scores) if want_prauc else float("nan")
    return {"auc": auc, "prauc": prauc, "n": len(oof_scores)}


def _prepare_groups(df_sub: pd.DataFrame) -> tuple[np.ndarray, int]:
    """video_id を整数エンコードし (groups, n_splits_actual) を返す。"""
    le = LabelEncoder()
    groups = le.fit_transform(df_sub["video_id"].values)
    n_actual = min(N_SPLITS, len(np.unique(groups)))
    return groups, n_actual


def _evaluate_subset_m1m2(
    df_sub: pd.DataFrame,
    feat_cols: list[str],
    fire_cols: list[str],
    targets_cfg: list[dict],
) -> dict[str, dict[str, float]]:
    """サブセットに対して全ターゲット×{m1,m2}を評価する。

    targets_cfg: [{"name": str, "y": np.ndarray, "is_reg": bool, "prauc": bool}, ...]
    Returns: {target_name: {"m1_auc", "m2_auc", "lift", "m2_prauc", "base_rate", "n"}}
    """
    if len(df_sub) < 30:
        return {}
    groups, n_splits = _prepare_groups(df_sub)
    if n_splits < 2:
        return {}
    X_all = df_sub[feat_cols].values.astype(np.float32)
    X_fire = df_sub[fire_cols].values.astype(np.float32)

    results: dict[str, dict[str, float]] = {}
    for cfg in targets_cfg:
        name: str = cfg["name"]
        y: np.ndarray = cfg["y"]
        is_reg: bool = cfg["is_reg"]
        want_pr: bool = cfg["prauc"]
        y_bin = (y > 0).astype(np.int8) if is_reg else y.astype(np.int8)
        base_rate = float(y_bin.mean()) if len(y_bin) > 0 else float("nan")

        r_m1 = _run_one_target(X_fire, y, groups, n_splits, is_reg, want_pr)
        r_m2 = _run_one_target(X_all, y, groups, n_splits, is_reg, want_pr)

        lift = (r_m2["auc"] - r_m1["auc"]
                if not (np.isnan(r_m2["auc"]) or np.isnan(r_m1["auc"]))
                else float("nan"))
        results[name] = {
            "m1_auc": r_m1["auc"],
            "m2_auc": r_m2["auc"],
            "lift": lift,
            "m2_prauc": r_m2["prauc"],
            "base_rate": base_rate,
            "n": r_m2["n"],
        }
    return results


def _make_targets_cfg(df_sub: pd.DataFrame, target_names: list[str]) -> list[dict]:
    """ターゲット名リストから targets_cfg を生成する。"""
    cfg_list: list[dict] = []
    for name in target_names:
        if name == "net_ojama_sign":
            y = df_sub["net_ojama"].values.astype(np.float32)
            cfg_list.append({"name": name, "y": y, "is_reg": True, "prauc": False})
        elif name in df_sub.columns:
            y = df_sub[name].values.astype(np.int8)
            cfg_list.append({"name": name, "y": y, "is_reg": False,
                             "prauc": name in PR_AUC_TARGETS})
    return cfg_list


def _print_results_table(
    results_by_phase: dict[str, dict[str, dict[str, float]]],
    targets: list[str],
) -> None:
    """位相別×ターゲット×{m1 AUC, m2 AUC, lift, PR-AUC, base_rate} の表を出力。"""
    phases = ["全体", "序", "中", "終"]
    print(f"\n{'ターゲット':<22} {'位相':<4} {'m1 AUC':>8} {'m2 AUC':>8} {'lift':>7} {'m2 PR-AUC':>10} {'base_rate':>10} {'n':>6}")
    print("-" * 85)
    for tgt in targets:
        for ph in phases:
            res = results_by_phase.get(ph, {}).get(tgt)
            if res is None:
                continue
            m1 = f"{res['m1_auc']:.3f}" if not np.isnan(res["m1_auc"]) else "  N/A"
            m2 = f"{res['m2_auc']:.3f}" if not np.isnan(res["m2_auc"]) else "  N/A"
            lift = f"{res['lift']:+.3f}" if not np.isnan(res["lift"]) else "   N/A"
            prauc = f"{res['m2_prauc']:.3f}" if not np.isnan(res["m2_prauc"]) else "     N/A"
            br = f"{res['base_rate']:.3f}" if not np.isnan(res["base_rate"]) else "      N/A"
            n = int(res["n"]) if res["n"] > 0 else 0
            print(f"{tgt:<22} {ph:<4} {m1:>8} {m2:>8} {lift:>7} {prauc:>10} {br:>10} {n:>6}")
        print()


def _compute_perm_importance(
    df_sub: pd.DataFrame,
    feat_cols: list[str],
    target_name: str,
    is_reg: bool,
) -> list[tuple[str, float]]:
    """permutation importance を最後の fold モデルで近似計算し Top N を返す。

    完全 CV は重いため、全データを train/val=8:2 で一回評価。
    """
    y = df_sub[target_name].values.astype(np.float32 if is_reg else np.int8)
    X = df_sub[feat_cols].values.astype(np.float32)
    n = len(X)
    split = int(n * 0.8)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]
    if len(np.unique(y_tr)) < 2:
        return []
    model = _build_model(is_reg)
    model.fit(X_tr, y_tr)
    scoring = "r2" if is_reg else "roc_auc"
    try:
        result = permutation_importance(
            model, X_va, y_va, n_repeats=PERM_IMP_N_REPEATS,
            random_state=RANDOM_STATE, scoring=scoring,
        )
    except Exception:
        return []
    mean_imp = result.importances_mean
    ranked = sorted(zip(feat_cols, mean_imp), key=lambda x: -x[1])
    return ranked[:PERM_IMP_TOP_N]


def _find_max_lift_target(
    results_full: dict[str, dict[str, float]],
    targets: list[str],
) -> str | None:
    """全体位相で残差リフト最大のターゲット名を返す。"""
    best_tgt: str | None = None
    best_lift = float("-inf")
    for tgt in targets:
        r = results_full.get(tgt)
        if r is None or np.isnan(r.get("lift", float("nan"))):
            continue
        if r["lift"] > best_lift:
            best_lift = r["lift"]
            best_tgt = tgt
    return best_tgt


def main() -> None:
    """メイン処理。"""
    warnings.filterwarnings("ignore")
    if not INPUT_PATH.exists():
        print(
            f"[ERROR] {INPUT_PATH} が見つかりません。label_exchange_outcome.py を先に実行してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] exchange_labels.csv 読み込み: {len(df)} 行")
    print(f"  位相別: {df['phase'].value_counts().to_dict()}")
    # returned_competitive が存在しない旧 CSV の互換チェック
    if "returned_competitive" not in df.columns:
        print("[WARN] returned_competitive 列がありません。label_exchange_outcome.py を再実行してください。")
        df["returned_competitive"] = 0
    print(f"  returned={df['returned'].mean():.3f}  returned_competitive={df['returned_competitive'].mean():.3f}")
    print(f"  opp_buried={df['opp_buried'].mean():.3f}  won={df['won'].mean():.3f}")
    print(f"  net_ojama: mean={df['net_ojama'].mean():.1f}")
    print()

    feat_cols = _get_feature_cols(df)
    fire_cols = _get_fire_power_cols(feat_cols)
    print(f"[INFO] 全特徴: {len(feat_cols)} 列  火力系特徴(m1): {len(fire_cols)} 列")
    print(f"  m1 列: {fire_cols}")
    print()

    target_names: list[str] = [
        "returned_competitive", "returned", "opp_buried", "net_ojama_sign", "won",
    ]

    subsets = {
        "全体": df,
        "序": df[df["phase"] == "序"].copy(),
        "中": df[df["phase"] == "中"].copy(),
        "終": df[df["phase"] == "終"].copy(),
    }

    results_by_phase: dict[str, dict[str, dict[str, float]]] = {}
    for phase_name, df_sub in subsets.items():
        print(f"[INFO] {phase_name} 位相 (n={len(df_sub)}) 評価中 ...")
        targets_cfg = _make_targets_cfg(df_sub, target_names)
        results_by_phase[phase_name] = _evaluate_subset_m1m2(
            df_sub, feat_cols, fire_cols, targets_cfg,
        )

    print()
    print("=" * 85)
    print("  A-2 火力統制: m1(火力系のみ) vs m2(全特徴) AUC 比較表")
    print("  残差リフト = m2-m1: 大きいほど火力以外の盤面情報が効いている")
    print("=" * 85)
    _print_results_table(results_by_phase, target_names)

    # 全体位相での残差リフト最大ターゲットを特定
    results_full = results_by_phase.get("全体", {})
    max_lift_tgt = _find_max_lift_target(results_full, target_names)

    print("=" * 85)
    print("  判定: 中盤で火力統制後に残る戦略シグナル")
    print("=" * 85)
    midgame = results_by_phase.get("中", {})
    for tgt in target_names:
        r = midgame.get(tgt)
        if r is None:
            print(f"  中盤 {tgt}: N/A")
            continue
        lift_mark = ""
        if not np.isnan(r["lift"]):
            if r["lift"] > 0.05:
                lift_mark = " ← 火力以外の情報が強く効いている"
            elif r["lift"] > 0.02:
                lift_mark = " ← 火力以外の情報がやや効いている"
            else:
                lift_mark = " ← ほぼ火力readinessで説明可能"
        prauc_str = f"  PR-AUC={r['m2_prauc']:.3f}" if not np.isnan(r["m2_prauc"]) else ""
        print(
            f"  中盤 {tgt:<22}: m1={r['m1_auc']:.3f} m2={r['m2_auc']:.3f}"
            f" lift={r['lift']:+.3f}  base={r['base_rate']:.3f}{prauc_str}{lift_mark}"
        )

    # permutation importance (全体位相、残差リフト最大ターゲット)
    if max_lift_tgt is not None:
        print()
        print(f"[INFO] permutation importance (全体、残差リフト最大ターゲット={max_lift_tgt})")
        is_reg = max_lift_tgt == "net_ojama_sign"
        col_name = "net_ojama" if is_reg else max_lift_tgt
        if col_name in df.columns:
            ranked = _compute_perm_importance(df, feat_cols, col_name, is_reg)
            print(f"  火力以外で効いた特徴 Top{PERM_IMP_TOP_N} (近似):")
            for rank, (feat, imp) in enumerate(ranked, 1):
                is_fire = feat in FIRE_POWER_FEAT_NAMES
                tag = "[火力]" if is_fire else "[その他]"
                print(f"    {rank}. {feat:<35} imp={imp:+.4f}  {tag}")


if __name__ == "__main__":
    main()
