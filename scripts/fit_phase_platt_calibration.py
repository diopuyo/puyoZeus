"""表示勝率較正の位相別 (match_progress) Platt scaling 本番校正器を学習する。

## 重要な注意 (2026-08-11、実測に基づく非推奨表明)
scripts/_calibration_phase_fit_2026-08-11.py の入れ子 GroupKFold 実測
(66動画・73,416ペア、B-1対称化修正+B-2進行度列 後の tier1 モデル) で、
**現状データでは校正なし (raw) が既に最良** と判明した:

    手法                              ECE全体   ECE序盤   ECE中盤   ECE終盤
    校正なし (raw)                    0.0125    0.0184    0.0102    0.0104
    現行Platt(本番approx適用)          0.0138    0.0173    0.0119    0.0172
    Platt(全位相共通、tier1で再学習)    0.0207    0.0340    0.0149    0.0149
    Isotonic(全位相共通)               0.0146    0.0225    0.0154    0.0181
    Platt(位相別)                      0.0236    0.0374    0.0165    0.0197
    Isotonic(位相別)                   0.0132    0.0063    0.0167    0.0225

B-1 (対称化ミラー標本の符号バグ修正) が系統的自信過剰の主因 (side非対称な
学習データ) を直接解消したため、旧モデル(非対称)向けに設計された後段校正が
もはや逆効果になっている可能性が高い (詳細CSV:
data/verify/calibration_phase_2026-08-11/calibration_phase_comparison.csv)。

本スクリプトは Phase1-2 ロードマップの成果物 (位相別Plattの学習・保存・
scripts/visualize_advantage_overlay.py --phase-calibration への配線) を完成
させるために実装するが、**現時点で --phase-calibration / --platt-calibration
を有効化することは推奨しない** (generate() の既定 False を維持すること)。
将来 tier1 モデルが変わり再び過信が生じた場合の切替先として保守する。

## 設計 (fit_platt_calibration.py と同じ単一 OOF 構成、位相ごとに分割)
「これから処理する未知の新しい動画」に適用する本番用の最終校正器なので、
学習時に存在した66動画は校正器の学習に全て使い切ってよい
(fit_platt_calibration.py のリーク防止設計と同じ理由、単一 GroupKFold OOF
で十分)。tier1 モデル (対称化ミラー標本込み) の単一 OOF 予測を match_progress
均等3分割でスライスし、位相ごとに独立な Platt scaling を学習する。

## 使い方
    python -m scripts.fit_phase_platt_calibration \
        --labeled data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv \
        --out data/indicators_v2/phase_platt_calibration.json
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probability_calibration import (  # noqa: E402
    PHASE_BOUND_EARLY, PHASE_BOUND_LATE, PHASE_NAMES, PhaseCalibrationParams,
    PlattCalibrationParams, apply_platt_calibration, phase_label_for_progress,
    save_phase_platt_calibration,
)
from scripts.fit_platt_calibration import _compute_ece, _fit_platt_coefficients  # noqa: E402
from scripts.model_indicator_win import (  # noqa: E402
    GBC_PARAMS, N_FOLDS, build_features, load_labeled_csv, pair_sides_for_win,
)
import scripts.visualize_advantage_overlay as vao  # noqa: E402

DEFAULT_LABELED = (
    "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"
)
DEFAULT_OUT = "data/indicators_v2/phase_platt_calibration.json"
# ペアリング最大時刻差 (_train_model と同一値)
MAX_TDIFF: float = 1.0
# 校正器を学習可能とみなす最小件数 (これ未満なら恒等変換 a=1,b=0 にフォールバック)
MIN_CALIB_FIT_N: int = 200


def _build_tier1_oof(
    labeled_path: str, n_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """データ読込〜tier1モデル(対称化込み)の単一GroupKFold OOF予測までをまとめる。"""
    df = load_labeled_csv(labeled_path)
    feat_cols = vao._resolve_features(df)
    paired = pair_sides_for_win(df, MAX_TDIFF)
    feat = build_features(paired, feat_cols)
    feat, cols = vao._add_interaction_columns(feat, feat_cols, paired)
    X = feat[cols].fillna(0.0).values.astype(float)
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    progress = np.asarray(vao._match_progress_from_totals(
        paired["board_puyo_total_1p"], paired["board_puyo_total_2p"]))
    mirror_sign = vao._mirror_sign(cols)
    print(f"  tier1特徴量数: {len(cols)}  サンプル数: {len(y)}  動画数: {len(np.unique(groups))}")

    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in GroupKFold(n_splits=n_folds).split(X, y, groups=groups):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_sym = np.vstack([X_tr, X_tr * mirror_sign])
        y_sym = np.concatenate([y_tr, 1 - y_tr])
        model = HistGradientBoostingClassifier(**GBC_PARAMS)
        model.fit(X_sym, y_sym)
        oof[te_idx] = model.predict_proba(X[te_idx])[:, 1]
    return oof, y, progress


def _fit_one_phase(raw_p: np.ndarray, y: np.ndarray) -> PlattCalibrationParams:
    """1位相分のPlatt係数を学習する(データ不足時は恒等変換にフォールバック)。"""
    if len(raw_p) < MIN_CALIB_FIT_N or len(np.unique(y)) < 2:
        print(f"    データ不足 (n={len(raw_p)}) -> 恒等変換 (a=1, b=0)")
        return PlattCalibrationParams(a=1.0, b=0.0, meta={"n_samples": int(len(raw_p)),
                                                            "fallback_identity": True})
    a, b = _fit_platt_coefficients(raw_p, y)
    return PlattCalibrationParams(a=a, b=b, meta={"n_samples": int(len(raw_p))})


def _report_phase(y: np.ndarray, raw_p: np.ndarray, calibrated_p: np.ndarray, phase: str) -> dict:
    """1位相分の校正前後 ECE/Brier/AUC を表示し、メタ情報を返す。"""
    ece_before, ece_after = _compute_ece(y, raw_p), _compute_ece(y, calibrated_p)
    auc = float(roc_auc_score(y, raw_p)) if len(np.unique(y)) > 1 else float("nan")
    print(f"    [{phase}] n={len(y)}  ECE {ece_before:.4f}->{ece_after:.4f}"
          f"  Brier {brier_score_loss(y, raw_p):.4f}->{brier_score_loss(y, calibrated_p):.4f}"
          f"  AUC(不変){auc:.4f}")
    return {"n": int(len(y)), "ece_before": ece_before, "ece_after": ece_after, "auc": auc}


def _fit_all_phases(
    oof: np.ndarray, y: np.ndarray, progress: np.ndarray,
) -> tuple[dict[str, PlattCalibrationParams], dict[str, dict]]:
    """位相ごとに Platt を学習し、係数とレポート用メタを返す (main の50行制約対応)。"""
    labels = np.array([phase_label_for_progress(float(p), PHASE_BOUND_EARLY, PHASE_BOUND_LATE)
                        for p in progress], dtype=object)
    phases: dict[str, PlattCalibrationParams] = {}
    reports: dict[str, dict] = {}
    for ph in PHASE_NAMES:
        mask = labels == ph
        params = _fit_one_phase(oof[mask], y[mask])
        calibrated = np.array([apply_platt_calibration(float(p), params) for p in oof[mask]])
        reports[ph] = _report_phase(y[mask], oof[mask], calibrated, ph)
        phases[ph] = params
    return phases, reports


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する (main を50行以内に保つための分割)。"""
    parser = argparse.ArgumentParser(description="位相別 Platt scaling 本番校正器の学習")
    parser.add_argument("--labeled", default=DEFAULT_LABELED)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[fit_phase_platt_calibration] labeled={args.labeled}")
    print("  !!! 注意: 2026-08-11実測ではraw(校正なし)が最良。既定は無効のまま推奨 !!!")
    oof, y, progress = _build_tier1_oof(args.labeled, args.n_folds)
    print("\n位相ごとの Platt 学習:")
    phases, reports = _fit_all_phases(oof, y, progress)
    meta = {
        "source_csv": args.labeled,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_folds": args.n_folds,
        "gbc_params": GBC_PARAMS,
        "phase_reports": reports,
        "note": (
            "tier1モデル(scripts.visualize_advantage_overlay._train_model と同一構成"
            "・対称化ミラー標本込み)の単一GroupKFold OOFから位相別Plattを学習。"
            "2026-08-11実測ではraw(校正なし)がECE全体で最良のため、本番での既定は"
            "無効(enable_phase_calibration=False)を推奨。"
            "詳細: data/verify/calibration_phase_2026-08-11/"
        ),
    }
    params = PhaseCalibrationParams(phases=phases, meta=meta)
    out_path = Path(args.out)
    save_phase_platt_calibration(params, out_path)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
