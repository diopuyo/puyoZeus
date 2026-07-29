"""校正 (calibration) 実測 -- 「モデルが0.8と言ったとき本当に80%勝つか」を数値で確認する。

## 背景
user 問い: 「AUC 0.70で有利80%と出るなら、30%の確率で80%ではないということ?」
これは AUC(順序の正しさ) と 校正(数値の目盛りの正しさ) の混同である。
AUC はスコアの単調変換で不変 (全予測を0.5方向に半分寄せても AUC は同一) なので、
現状 model_indicator_win.py の AUC/logloss 報告だけでは「80%が本当に80%か」は
測っていない。本スクリプトは GroupKFold OOF 予測確率を用いて
校正曲線・ECE (期待校正誤差)・Brier score を実測する。

## 手法
- scripts/model_indicator_win.py の既存関数を import してそのまま再利用 (無改変)。
  load_labeled_csv -> pair_sides_for_win -> build_features -> run_oof_classifier
  (HistGBC, GroupKFold video_id 単位、既存 Step2 学習と同一条件)
- 校正ビンは幅 0.1 固定 (0.0-0.1, 0.1-0.2, ..., 0.9-1.0) の 10 ビン
- 位相別 (序盤/中盤/終盤) は既存の --fixed-q33/--fixed-q67 (手数境界) をそのまま流用し、
  run_phase_models と同じ「位相ごとに別モデルを OOF 学習」方式で確率を得る
  (全体一本の OOF 確率を位相で単純フィルタするのではなく、
   既存レポートの位相別 AUC 0.59/0.68/0.74 と対応させるため)

## 制約
- src/indicators_v2.py, src/chain_bitboard.py には一切触れない・import もしない
  (#24 Step2 で別エージェントが同時編集中のため)
- scripts/model_indicator_win.py は無改変、既存関数を import するのみ

## 使い方
    nice -n 19 python -m scripts._calibration_check_2026-07-29 \
        --labeled data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv \
        --fixed-q33 18 --fixed-q67 40 \
        --out-dir data/verify/win_eval_combined66_2026-07-29/calibration_2026-07-29
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts.model_indicator_win import (
    DEFAULT_MAX_TDIFF,
    N_FOLDS,
    _get_indicator_cols,
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
    run_oof_classifier,
)

# =============================================================================
# 定数
# =============================================================================

# 校正ビン幅 (0.1刻み = 10ビン、 全指標特徴量の学習と同条件)
N_CALIBRATION_BINS: int = 10

# ECE集計・報告で「信頼できる」とみなす最小ビン件数 (これ未満は注記のみ)
MIN_RELIABLE_BIN_COUNT: int = 100

# ユーザーの問いに直接答えるための代表予測値 (0.1幅ビンの下端に対応)
ANSWER_PROBS: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)

# 分布要約で報告する percentile 一覧
DIST_PERCENTILES: tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)


# =============================================================================
# 校正テーブル計算
# =============================================================================

def compute_calibration_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """予測確率を0.1幅ビンに区切り、ビンごとの平均予測確率・実勝率・件数を返す。"""
    bin_edges = np.linspace(0.0, 1.0, N_CALIBRATION_BINS + 1)
    # p=1.0 も最終ビンに含める (right境界含む)
    bin_idx = np.clip(
        np.digitize(p, bin_edges[1:-1], right=False), 0, N_CALIBRATION_BINS - 1
    )
    rows: list[dict] = []
    for b in range(N_CALIBRATION_BINS):
        mask = bin_idx == b
        n = int(mask.sum())
        row = {
            "bin": f"{bin_edges[b]:.1f}-{bin_edges[b + 1]:.1f}",
            "n": n,
            "mean_pred": float(p[mask].mean()) if n > 0 else float("nan"),
            "actual_rate": float(y[mask].mean()) if n > 0 else float("nan"),
            "reliable": n >= MIN_RELIABLE_BIN_COUNT,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compute_ece(calib_table: pd.DataFrame, n_total: int) -> float:
    """期待校正誤差 (Expected Calibration Error) = Σ (n_bin/n) * |実勝率 - 平均予測|。"""
    ece = 0.0
    for _, row in calib_table.iterrows():
        if row["n"] == 0:
            continue
        weight = row["n"] / n_total
        ece += weight * abs(row["actual_rate"] - row["mean_pred"])
    return float(ece)


def summarize_distribution(p: np.ndarray) -> dict:
    """予測確率分布の要約統計 (percentile・両端割合) を返す。"""
    pct = {f"p{q}": float(np.percentile(p, q)) for q in DIST_PERCENTILES}
    frac_mid = float(((p >= 0.4) & (p <= 0.6)).mean())
    frac_extreme = float(((p <= 0.2) | (p >= 0.8)).mean())
    return {
        "mean": float(p.mean()),
        "std": float(p.std()),
        "min": float(p.min()),
        "max": float(p.max()),
        **pct,
        "frac_in_0.4_0.6": frac_mid,
        "frac_in_extreme_0.2_0.8": frac_extreme,
    }


# =============================================================================
# 位相別 OOF 確率取得 (run_phase_models と同一方式: 位相ごとに別モデル)
# =============================================================================

def get_phase_oof(
    paired: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    tsumo_q33: float,
    tsumo_q67: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """序盤/中盤/終盤ごとに GroupKFold OOF 予測確率を計算し (y, p) を返す。"""
    tsumo_vals = paired["tsumo_1p"].astype(float).values
    phase_masks = {
        "序盤": tsumo_vals <= tsumo_q33,
        "中盤": (tsumo_vals > tsumo_q33) & (tsumo_vals <= tsumo_q67),
        "終盤": tsumo_vals > tsumo_q67,
    }
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for phase, mask in phase_masks.items():
        X_ph, y_ph, groups_ph = X[mask], y[mask], groups[mask]
        n_unique = len(np.unique(groups_ph))
        folds = min(n_folds, max(2, n_unique))
        if len(X_ph) < 20 or len(np.unique(y_ph)) < 2:
            print(f"    {phase}: データ不足 -> skip")
            continue
        oof_proba, _ = run_oof_classifier(X_ph, y_ph, groups_ph, folds)
        valid = ~np.isnan(oof_proba[:, 0])
        result[phase] = (y_ph[valid], oof_proba[valid, 1])
        print(f"    {phase}: n={valid.sum()}")
    return result


# =============================================================================
# レポート出力
# =============================================================================

def print_calibration_report(
    label: str,
    y: np.ndarray,
    p: np.ndarray,
) -> pd.DataFrame:
    """校正テーブル・ECE・Brier・logloss・分布要約を出力し、テーブルを返す。"""
    calib = compute_calibration_table(y, p)
    ece = compute_ece(calib, len(y))
    brier = float(brier_score_loss(y, p))
    ll = float(log_loss(y, np.column_stack([1 - p, p])))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

    print(f"\n  ─── 校正曲線: {label} (n={len(y)}, AUC={auc:.4f}) ───")
    print(f"  {'bin':<12}  {'n':>7}  {'平均予測':>8}  {'実勝率':>8}  {'差':>7}  信頼性")
    print("  " + "-" * 62)
    for _, row in calib.iterrows():
        if row["n"] == 0:
            print(f"  {row['bin']:<12}  {0:>7}  {'n/a':>8}  {'n/a':>8}")
            continue
        diff = row["actual_rate"] - row["mean_pred"]
        rel = "" if row["reliable"] else f"(件数少 n<{MIN_RELIABLE_BIN_COUNT}, 信頼度低)"
        print(f"  {row['bin']:<12}  {int(row['n']):>7}  {row['mean_pred']:>8.4f}"
              f"  {row['actual_rate']:>8.4f}  {diff:>+7.4f}  {rel}")
    print(f"  ECE (期待校正誤差) = {ece:.4f}")
    print(f"  Brier score        = {brier:.4f}")
    print(f"  logloss            = {ll:.4f}")

    dist = summarize_distribution(p)
    print(f"  予測確率分布: mean={dist['mean']:.3f} std={dist['std']:.3f}"
          f" min={dist['min']:.3f} max={dist['max']:.3f}")
    print(f"    percentile: " + " ".join(
        f"p{q}={dist[f'p{q}']:.3f}" for q in DIST_PERCENTILES
    ))
    print(f"    0.4-0.6中央域割合={dist['frac_in_0.4_0.6']:.1%}"
          f"  0.2以下or0.8以上割合={dist['frac_in_extreme_0.2_0.8']:.1%}")

    calib.attrs["ece"] = ece
    calib.attrs["brier"] = brier
    calib.attrs["logloss"] = ll
    calib.attrs["auc"] = auc
    return calib


def print_direct_answers(calib: pd.DataFrame, label: str) -> None:
    """ユーザーの問い「予測0.8/0.9/0.7/0.6の局面群の実際の勝率は?」に直接回答する。"""
    print(f"\n  ─── 直接回答: {label} ───")
    for target in ANSWER_PROBS:
        bin_label = f"{target:.1f}-{target + 0.1:.1f}"
        row = calib[calib["bin"] == bin_label]
        if row.empty or int(row.iloc[0]["n"]) == 0:
            print(f"  予測{target:.1f}台 (bin {bin_label}): データなし")
            continue
        r = row.iloc[0]
        rel_note = "" if r["reliable"] else f" [件数少 n={int(r['n'])}, 信頼度低]"
        print(f"  予測{target:.1f}台 (bin {bin_label}, n={int(r['n'])}):"
              f" 平均予測={r['mean_pred']:.3f} 実際の勝率={r['actual_rate']:.3f}{rel_note}")


# =============================================================================
# CSV 保存
# =============================================================================

def save_calibration_csv(calib: pd.DataFrame, out_path: Path, label: str) -> None:
    """校正テーブルを CSV 保存する。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calib.to_csv(out_path, index=False)
    print(f"  CSV 保存 ({label}): {out_path}")


# =============================================================================
# メイン
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="校正 (calibration) 実測")
    parser.add_argument(
        "--labeled",
        default="data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv",
    )
    parser.add_argument("--max-tdiff", type=float, default=DEFAULT_MAX_TDIFF)
    parser.add_argument("--fixed-q33", type=float, default=18.0)
    parser.add_argument("--fixed-q67", type=float, default=40.0)
    parser.add_argument(
        "--out-dir",
        default="data/verify/win_eval_combined66_2026-07-29/calibration_2026-07-29",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    print(f"[_calibration_check] labeled={args.labeled}")

    print("\n=== 1. データ読み込み ===")
    df = load_labeled_csv(args.labeled)

    print("\n=== 2. 1P/2P ペアリング ===")
    paired = pair_sides_for_win(df, args.max_tdiff)
    y_all = paired["won_1p"].astype(int).values
    groups_all = paired["video_id_1p"].values

    print("\n=== 3. 特徴量構築 ===")
    indicator_cols = _get_indicator_cols(paired)
    feat_df = build_features(paired, indicator_cols)
    X_all = feat_df.fillna(0.0).values.astype(float)
    print(f"  特徴量数: {X_all.shape[1]}  サンプル数: {len(y_all)}")

    print("\n=== 4. 全体 OOF 予測確率 (HistGBC, 全指標) ===")
    oof_proba, _ = run_oof_classifier(X_all, y_all, groups_all, N_FOLDS)
    valid = ~np.isnan(oof_proba[:, 0])
    y_v, p_v = y_all[valid], oof_proba[valid, 1]

    calib_all = print_calibration_report("全体", y_v, p_v)
    print_direct_answers(calib_all, "全体")
    save_calibration_csv(calib_all, out_dir / "calibration_overall.csv", "全体")

    print("\n=== 5. 位相別 OOF 予測確率 (序盤/中盤/終盤 別モデル) ===")
    phase_oof = get_phase_oof(
        paired, X_all, y_all, groups_all, N_FOLDS, args.fixed_q33, args.fixed_q67
    )
    for phase, (y_ph, p_ph) in phase_oof.items():
        calib_ph = print_calibration_report(f"位相={phase}", y_ph, p_ph)
        print_direct_answers(calib_ph, f"位相={phase}")
        save_calibration_csv(calib_ph, out_dir / f"calibration_{phase}.csv", phase)

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
