"""勝率予測モデルの系統的自信過剰を補正する Platt scaling (全位相共通) 校正器を
学習し、data/indicators_v2/platt_calibration.json に保存する (本番用)。

## 経緯
scripts/_calibration_fit_2026-07-29.py の実測 (入れ子 GroupKFold で校正手法の
汎化を検証) で Platt(全位相共通) が ECE全体 0.0264→0.0189、終盤
0.0559→0.0354 に改善すると確認され、user承認済み。本スクリプトは
「本番運用でそのまま使う1個の最終校正器」を作る。

## リーク防止設計 (入れ子でなく単一 OOF にする理由、user承認済み)
_calibration_fit_2026-07-29.py の入れ子 (外側+内側 GroupKFold) 構成は
「校正手法自体が未知動画にどこまで一般化するか」を検証するための実験構成。
本番用の最終校正器はそれと目的が異なり、以下の理由で単一 GroupKFold OOF
予測1本から学習してよい:
  - 校正器の適用対象は「これから処理する未知の新しい動画」であり、学習時に
    存在した66動画は校正器の学習に全て使い切ってよい(検証実験ではないため
    outer-test を隔離する必要が無い)。
  - 単一 OOF 予測は GroupKFold で「モデル自身が学習に使っていない fold」に
    対する予測なので、少なくとも「モデルの訓練誤差をそのまま校正器が見て
    しまう」リーク(校正器が過学習したモデルの自信過剰をそのまま追認する)は
    避けられている。これは新規動画への汎化"率"を測定する検証ではなく、
    「学習済みモデルの既知の歪みパターンを補正する関数を1つ確定する」工程
    だと整理する。

## 使い方
    nice -n 19 python -m scripts.fit_platt_calibration \\
        --labeled data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv \\
        --out data/indicators_v2/platt_calibration.json
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probability_calibration import (  # noqa: E402
    PlattCalibrationParams, apply_platt_calibration, save_platt_calibration,
)
from scripts.model_indicator_win import (  # noqa: E402
    DEFAULT_MAX_TDIFF, GBC_PARAMS, N_FOLDS, _get_indicator_cols,
    build_features, load_labeled_csv, pair_sides_for_win, run_oof_classifier,
)

# 校正ビン幅 (0.1刻み = 10ビン。_calibration_fit_2026-07-29.py と同一定義)
N_CALIBRATION_BINS: int = 10

# logit変換の発散防止クリップ幅 (src/probability_calibration.py と同一値)
LOGIT_EPS: float = 1e-6

# 校正器の乱数シード (GBC_PARAMS の random_state と揃える)
RANDOM_STATE: int = 42

DEFAULT_LABELED = (
    "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"
)
DEFAULT_OUT = "data/indicators_v2/platt_calibration.json"


def _compute_ece(y: np.ndarray, p: np.ndarray) -> float:
    """期待校正誤差 (ECE) = Σ (n_bin/n) * |実勝率 - 平均予測| (10ビン)。"""
    bin_edges = np.linspace(0.0, 1.0, N_CALIBRATION_BINS + 1)
    bin_idx = np.clip(
        np.digitize(p, bin_edges[1:-1], right=False), 0, N_CALIBRATION_BINS - 1
    )
    ece = 0.0
    for b in range(N_CALIBRATION_BINS):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / len(y)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def _fit_platt_coefficients(raw_p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """logit(raw_p) を1特徴量とする1次元ロジスティック回帰で a, b を学習する。"""
    p_clip = np.clip(raw_p, LOGIT_EPS, 1.0 - LOGIT_EPS)
    logit_p = np.log(p_clip / (1.0 - p_clip))
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(logit_p.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def _build_oof(
    labeled_path: str, max_tdiff: float, n_folds: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """データ読込〜単一 GroupKFold OOF 予測までをまとめる (50行制約対応の分割)。"""
    df = load_labeled_csv(labeled_path)
    paired = pair_sides_for_win(df, max_tdiff)
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    indicator_cols = _get_indicator_cols(paired)
    feat_df = build_features(paired, indicator_cols)
    X = feat_df.fillna(0.0).values.astype(float)
    print(f"  特徴量数: {X.shape[1]}  サンプル数: {len(y)}  指標数: {len(indicator_cols)}")
    oof, _ = run_oof_classifier(X, y, groups, n_folds)
    valid = ~np.isnan(oof[:, 0])
    return oof[valid, 1], y[valid], int(valid.sum())


def _report_and_build_meta(
    raw_p: np.ndarray, y: np.ndarray, a: float, b: float,
    labeled_path: str, n_folds: int,
) -> dict:
    """校正前後の ECE/Brier/AUC を計測・表示し、保存用メタ情報を組み立てる。"""
    calibrated_p = np.array([apply_platt_calibration(float(p), PlattCalibrationParams(a, b))
                              for p in raw_p])
    ece_before, ece_after = _compute_ece(y, raw_p), _compute_ece(y, calibrated_p)
    brier_before = float(brier_score_loss(y, raw_p))
    brier_after = float(brier_score_loss(y, calibrated_p))
    auc = float(roc_auc_score(y, raw_p)) if len(np.unique(y)) > 1 else float("nan")
    print(f"\n=== Platt(全位相共通) 本番校正器 ===")
    print(f"  a={a:.4f}  b={b:.4f}")
    print(f"  ECE  校正前={ece_before:.4f} -> 校正後={ece_after:.4f}")
    print(f"  Brier 校正前={brier_before:.4f} -> 校正後={brier_after:.4f}")
    print(f"  AUC(校正で不変)={auc:.4f}")
    return {
        "source_csv": labeled_path,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(y)),
        "n_folds": n_folds,
        "gbc_params": GBC_PARAMS,
        "ece_before": ece_before, "ece_after": ece_after,
        "brier_before": brier_before, "brier_after": brier_after,
        "auc": auc,
        "note": (
            "全位相共通Platt。model_indicator_win.py の全指標特徴量+"
            "combined66データ、単一GroupKFold OOF(nestedでない)から学習。"
            "適用対象がこのモデルと異なる場合(例: visualize_advantage_overlay.py の"
            "tier1軽量モデル)は近似適用となる点に注意 (詳細は呼出元のdocstring参照)。"
        ),
    }


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する。"""
    parser = argparse.ArgumentParser(description="Platt scaling 本番校正器の学習")
    parser.add_argument("--labeled", default=DEFAULT_LABELED)
    parser.add_argument("--max-tdiff", type=float, default=DEFAULT_MAX_TDIFF)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[fit_platt_calibration] labeled={args.labeled}")
    raw_p, y, n = _build_oof(args.labeled, args.max_tdiff, args.n_folds)
    print(f"  OOF有効サンプル数: {n}")
    a, b = _fit_platt_coefficients(raw_p, y)
    meta = _report_and_build_meta(raw_p, y, a, b, args.labeled, args.n_folds)
    params = PlattCalibrationParams(a=a, b=b, meta=meta)
    out_path = Path(args.out)
    save_platt_calibration(params, out_path)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
