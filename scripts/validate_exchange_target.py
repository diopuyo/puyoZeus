"""打ち合いラベル予測性能の検証スクリプト。

exchange_labels.csv を読み込み、盤面特徴 → 各ターゲットの予測性能を測る。
最終勝者(won)との AUC 比較を行い、「近い地平ターゲットが当たるか」を検証する。

比較ターゲット:
  (i)   returned  : 2値, AUC
  (ii)  opp_buried: 2値, AUC
  (iii) net_ojama : 回帰→符号AUC (net_ojama>0 を正, AUC で測定)
  (iv)  won       : 2値, AUC (比較用)

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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ============================
# 定数定義
# ============================
INPUT_PATH = PROJ_ROOT / "data" / "indicators_v2" / "exchange_labels.csv"
N_SPLITS: int = 5
RANDOM_STATE: int = 42


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """特徴列名を返す。fire_/opp_/diff_ プレフィックスを持つ列。"""
    exclude = {"video_id", "game_idx", "t_sec", "fire_side", "phase",
               "won", "net_ojama", "returned", "opp_buried"}
    return [c for c in df.columns if c not in exclude]


def _auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """クラス単一の場合に 0.5 を返す安全な AUC 計算。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _run_classification(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_SPLITS,
) -> tuple[float, int]:
    """GroupKFold で 2値分類の AUC と有効サンプル数を返す。"""
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,
        random_state=RANDOM_STATE,
        n_iter_no_change=20,
    )
    gkf = GroupKFold(n_splits=n_splits)
    scores_list: list[float] = []
    n_total = 0
    for train_idx, val_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
            continue
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[val_idx])[:, 1]
        auc = _auc_safe(y[val_idx], proba)
        if not np.isnan(auc):
            scores_list.append(auc)
            n_total += len(val_idx)
    if not scores_list:
        return float("nan"), 0
    return float(np.mean(scores_list)), n_total


def _run_regression_sign_auc(
    X: np.ndarray,
    y_cont: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_SPLITS,
) -> tuple[float, int]:
    """net_ojama の符号 AUC: y_cont>0 を正ラベルとして AUC を計算。"""
    y_sign = (y_cont > 0).astype(np.int8)
    # 偏りが強い場合は符号が全0/全1でAUC出ない → 安全チェック
    if len(np.unique(y_sign)) < 2:
        return float("nan"), 0
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=4,
        random_state=RANDOM_STATE,
        n_iter_no_change=20,
    )
    gkf = GroupKFold(n_splits=n_splits)
    scores_list: list[float] = []
    n_total = 0
    for train_idx, val_idx in gkf.split(X, y_cont, groups):
        if len(np.unique(y_sign[train_idx])) < 2:
            continue
        model.fit(X[train_idx], y_cont[train_idx])
        pred = model.predict(X[val_idx])
        auc = _auc_safe(y_sign[val_idx], pred)
        if not np.isnan(auc):
            scores_list.append(auc)
            n_total += len(val_idx)
    if not scores_list:
        return float("nan"), 0
    return float(np.mean(scores_list)), n_total


def _evaluate_subset(
    df_sub: pd.DataFrame,
    feat_cols: list[str],
    n_splits: int,
) -> dict[str, tuple[float, int]]:
    """サブセット df に対して全ターゲットを評価し、{target: (auc, n)} を返す。"""
    if len(df_sub) < 20:
        return {}
    X = df_sub[feat_cols].values.astype(np.float32)

    # groups: video_id を整数エンコード
    le = LabelEncoder()
    groups = le.fit_transform(df_sub["video_id"].values)
    n_unique_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_unique_groups)
    if actual_splits < 2:
        return {}

    results: dict[str, tuple[float, int]] = {}

    # (i) returned
    y_ret = df_sub["returned"].values.astype(np.int8)
    results["returned"] = _run_classification(X, y_ret, groups, actual_splits)

    # (ii) opp_buried
    y_bur = df_sub["opp_buried"].values.astype(np.int8)
    results["opp_buried"] = _run_classification(X, y_bur, groups, actual_splits)

    # (iii) net_ojama 符号AUC
    y_net = df_sub["net_ojama"].values.astype(np.float32)
    results["net_ojama_sign"] = _run_regression_sign_auc(X, y_net, groups, actual_splits)

    # (iv) won
    y_won = df_sub["won"].values.astype(np.int8)
    results["won"] = _run_classification(X, y_won, groups, actual_splits)

    return results


def _print_results_table(
    results_by_phase: dict[str, dict[str, tuple[float, int]]],
    targets: Sequence[str],
) -> None:
    """位相別×ターゲットの AUC 表を標準出力に表示する。"""
    phases = ["全体", "序", "中", "終"]
    col_width = 16

    # ヘッダ
    header = f"{'ターゲット':<20}" + "".join(f"{p:>{col_width}}" for p in phases)
    print(header)
    print("-" * len(header))

    for tgt in targets:
        row_str = f"{tgt:<20}"
        for ph in phases:
            res = results_by_phase.get(ph, {})
            val = res.get(tgt)
            if val is None or np.isnan(val[0]):
                cell = "   N/A"
            else:
                auc, n = val
                cell = f"{auc:.3f}(n={n})"
            row_str += f"{cell:>{col_width}}"
        print(row_str)


def main() -> None:
    """メイン処理。"""
    warnings.filterwarnings("ignore")
    if not INPUT_PATH.exists():
        print(f"[ERROR] {INPUT_PATH} が見つかりません。label_exchange_outcome.py を先に実行してください。", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] exchange_labels.csv 読み込み: {len(df)} 行")
    print(f"  位相別: {df['phase'].value_counts().to_dict()}")
    print(f"  returned 分布: {df['returned'].value_counts().to_dict()}")
    print(f"  opp_buried 分布: {df['opp_buried'].value_counts().to_dict()}")
    print(f"  net_ojama: mean={df['net_ojama'].mean():.2f} std={df['net_ojama'].std():.2f}")
    print(f"  won 分布: {df['won'].value_counts().to_dict()}")
    print()

    feat_cols = _get_feature_cols(df)
    print(f"[INFO] 特徴量: {len(feat_cols)} 列")
    print()

    # 位相別サブセットを準備
    subsets = {
        "全体": df,
        "序": df[df["phase"] == "序"],
        "中": df[df["phase"] == "中"],
        "終": df[df["phase"] == "終"],
    }

    results_by_phase: dict[str, dict[str, tuple[float, int]]] = {}
    for phase_name, df_sub in subsets.items():
        print(f"[INFO] {phase_name} 位相 (n={len(df_sub)}) 評価中 ...")
        results_by_phase[phase_name] = _evaluate_subset(df_sub, feat_cols, N_SPLITS)

    targets = ["returned", "opp_buried", "net_ojama_sign", "won"]

    print()
    print("=" * 80)
    print("  打ち合いラベル予測 vs 最終勝者 AUC 比較表 (holdout: video_id GroupKFold)")
    print("=" * 80)
    _print_results_table(results_by_phase, targets)
    print()

    # 「近い地平ターゲットは最終勝者より当たるか」の判定
    print("=" * 80)
    print("  判定: 中盤における「近い地平ターゲット」vs「最終勝者(won)」")
    print("=" * 80)
    midgame = results_by_phase.get("中", {})
    won_mid = midgame.get("won", (float("nan"), 0))
    near_targets = {k: midgame.get(k, (float("nan"), 0)) for k in ["returned", "opp_buried", "net_ojama_sign"]}

    print(f"  中盤 won-AUC: {won_mid[0]:.3f}  (n={won_mid[1]})")
    better_count = 0
    for name, (auc, n) in near_targets.items():
        diff = auc - won_mid[0] if not (np.isnan(auc) or np.isnan(won_mid[0])) else float("nan")
        mark = ""
        if not np.isnan(diff):
            if diff > 0.01:
                mark = " ← 近い地平が勝る"
                better_count += 1
            elif diff < -0.01:
                mark = " ← won の方が高い"
            else:
                mark = " ← ほぼ同等"
        print(f"  中盤 {name}-AUC: {auc:.3f}  (n={n})  Δ={diff:+.3f}{mark}")

    print()
    if better_count >= 2:
        print("[結論] 近い地平ターゲットは中盤で won より予測しやすい可能性あり → 採用候補")
    elif better_count == 1:
        print("[結論] 一部の近い地平ターゲットが won を上回る → 部分的に有望")
    else:
        print("[結論] 近い地平ターゲットが中盤 won を明確に上回るものはなし → 現時点では won と大差なし")


if __name__ == "__main__":
    main()
