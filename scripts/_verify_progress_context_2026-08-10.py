"""有利不利モデルへの「進行度の文脈列」追加 (Phase1-1 B-2) の効果検証。

アーキ設計 (2026-08-10、user承認済み)・コーダ実装: `match_progress`
(両者の盤面ぷよ総数平均、0=序盤〜1=終盤) と `color_puyo_x_earliness`
(色ぷよ差×(1-進行度)、「序盤ほど色ぷよ差が効く」の交互作用) を
scripts/visualize_advantage_overlay.py の学習/推論パイプラインへ追加した。

確定事実 (data/verify/j1_color_lead_clean_noinflight_2026-08-10.txt):
発火±12s除外のクリーン盤面で色ぷよ+8〜15リード側の実勝率は
序盤79.7% / 中盤65.2% (中立48.1%)。

本スクリプトは **追加前 (--no-progress) / 追加後 (既定)** の両方で
GroupKFold (video_id 単位) OOF 評価を行い:
  1. 全体 AUC (無悪化ゲート: 現状 0.6605 を下回らないこと)
  2. おじゃまフラット局面 AUC (現状0.5554、_verify_mirror_symmetrize_fix_
     2026-08-10.py の層別と同じ閾値)
  3. 位相別 (序/中/終、tsumo_count_rate の3分位) OOF AUC
  4. match_progress_diff / color_puyo_x_earliness_diff の permutation importance
を比較レポートする。

使い方 (本測定・重い。66動画5fold、前例で数十分):
    python -m scripts._verify_progress_context_2026-08-10 --no-progress
    python -m scripts._verify_progress_context_2026-08-10

スモーク (高速・小サンプルで動作確認のみ):
    python -m scripts._verify_progress_context_2026-08-10 --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.visualize_advantage_overlay import (  # noqa: E402
    COLOR_EARLINESS_INTERACTION_COL,
    COLOR_OJAMA_INTERACTION_COL,
    MATCH_PROGRESS_COL,
    OJAMA_FLAT_COL,
    TRAIN_CSV_PATH,
    _add_interaction_columns,
    _mirror_sign,
    _resolve_features,
)
from scripts.model_indicator_win import (  # noqa: E402
    GBC_PARAMS,
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
)

# _verify_mirror_symmetrize_fix_2026-08-10.py と同じ層別閾値 (踏襲)。
FLAT_HI: float = 0.8
FLAT_LO: float = 0.37

# 手数三分位境界 (scripts/model_indicator_win.py の既定値を踏襲)
TSUMO_EARLY_RATIO: float = 0.33
TSUMO_LATE_RATIO: float = 0.67

N_FOLDS: int = 5
PERM_N_REPEATS: int = 20
PERM_RANDOM_STATE: int = 42

# 無悪化ゲート基準 (現状値、タスク仕様書より)
BASELINE_OVERALL_AUC: float = 0.6605
BASELINE_FLAT_AUC: float = 0.5554


def _prepare(smoke: bool, with_progress: bool) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """paired 特徴量・使用列・tsumo三分位計算用の paired 生データを用意する。"""
    df = load_labeled_csv(TRAIN_CSV_PATH)
    if smoke:
        vids = sorted(df["video_id"].unique())[:3]
        df = df[df["video_id"].isin(vids)].reset_index(drop=True)
    feat_cols = _resolve_features(df)
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    feat = build_features(paired, feat_cols)
    feat, cols = _add_interaction_columns(
        feat, feat_cols, paired if with_progress else None,
    )
    feat["video_id_1p"] = paired["video_id_1p"].values
    feat["won_1p"] = paired["won_1p"].astype(int).values
    feat["tsumo_1p"] = paired["tsumo_1p"].astype(float).values
    return feat, cols, paired


def _flat_score_from_diff(feat: pd.DataFrame) -> np.ndarray:
    """層別用: ojama_flat_score_diff 列 (無ければ全 NaN 扱いで層別スキップ)。"""
    col = f"{OJAMA_FLAT_COL}_diff"
    if col not in feat.columns:
        return np.full(len(feat), np.nan)
    return feat[col].fillna(0.0).values


def _auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def _phase_masks(tsumo: np.ndarray) -> dict[str, np.ndarray]:
    q_low = float(np.quantile(tsumo, TSUMO_EARLY_RATIO))
    q_high = float(np.quantile(tsumo, TSUMO_LATE_RATIO))
    return {
        "序盤": tsumo <= q_low,
        "中盤": (tsumo > q_low) & (tsumo <= q_high),
        "終盤": tsumo > q_high,
    }


def run(with_progress: bool, smoke: bool, gbc_params: dict) -> None:
    label = "追加後(match_progress有効)" if with_progress else "追加前(--no-progress)"
    print(f"\n{'='*70}\n{label}  (smoke={smoke})\n{'='*70}")
    feat, cols, _paired = _prepare(smoke, with_progress)
    X = feat[cols].fillna(0.0).values
    y = feat["won_1p"].values
    groups = feat["video_id_1p"].values
    tsumo = feat["tsumo_1p"].values
    flat = _flat_score_from_diff(feat)

    progress_col = f"{MATCH_PROGRESS_COL}_diff"
    earliness_col = f"{COLOR_EARLINESS_INTERACTION_COL}_diff"
    perm_targets = [c for c in (progress_col, earliness_col) if c in cols]

    n_splits = 2 if smoke else N_FOLDS
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    perm_importances: dict[str, list[float]] = {name: [] for name in perm_targets}

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        sign = _mirror_sign(cols)
        X_tr_sym = np.vstack([X_tr, X_tr * sign])
        y_tr_sym = np.concatenate([y_tr, 1 - y_tr])
        model = HistGradientBoostingClassifier(**gbc_params)
        model.fit(X_tr_sym, y_tr_sym)
        oof[te_idx] = model.predict_proba(X[te_idx])[:, 1]
        for name in perm_targets:
            idx = cols.index(name)
            pi = permutation_importance(
                model, X[te_idx], y[te_idx], n_repeats=PERM_N_REPEATS,
                random_state=PERM_RANDOM_STATE, scoring="roc_auc",
            )
            perm_importances[name].append(float(pi.importances_mean[idx]))
        print(f"  fold {fold+1}/{n_splits} 完了 (train={len(tr_idx)} test={len(te_idx)})")

    valid = ~np.isnan(oof)
    overall_auc = _auc(y[valid], oof[valid])
    print(f"\n全体 OOF AUC: {overall_auc:.4f}  (無悪化ゲート基準: {BASELINE_OVERALL_AUC})")

    if not np.all(np.isnan(flat)):
        for name, mask in (
            (f"フラット (>= {FLAT_HI})", flat >= FLAT_HI),
            ("中間", (flat > FLAT_LO) & (flat < FLAT_HI)),
            (f"差が大きい (<= {FLAT_LO})", flat <= FLAT_LO),
        ):
            m = mask & valid
            n = int(m.sum())
            if n < 50:
                print(f"  層別[{name}] n={n} (サンプル不足)")
                continue
            print(f"  層別[{name}] n={n:6d}  AUC={_auc(y[m], oof[m]):.4f}"
                  f"  (無悪化ゲート基準: {BASELINE_FLAT_AUC})")
    else:
        print("  (ojama_flat_score_diff 列が無いため層別スキップ)")

    print("\n  位相別 (tsumo_count 3分位) OOF AUC:")
    for phase, mask in _phase_masks(tsumo).items():
        m = mask & valid
        n = int(m.sum())
        if n < 50 or len(np.unique(y[m])) < 2:
            print(f"    {phase}: n={n} (データ不足 -> nan)")
            continue
        print(f"    {phase}: n={n:6d}  AUC={_auc(y[m], oof[m]):.4f}")

    for name in perm_targets:
        vals = perm_importances[name]
        print(f"  permutation importance[{name}]: 平均 {np.mean(vals):+.5f}"
              f" (fold別 {['%.5f' % v for v in vals]})")
    if not perm_targets:
        print("  (進行度列は未使用 — --no-progress or 列欠如)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-progress", action="store_true",
                     help="追加前の挙動 (match_progress/color_puyo_x_earliness 無し) を再現")
    ap.add_argument("--smoke", action="store_true",
                     help="動画3本・2fold・軽量パラメータで構造検証のみ行う")
    a = ap.parse_args()
    gbc_params = dict(GBC_PARAMS)
    if a.smoke:
        gbc_params["max_iter"] = 30
    run(not a.no_progress, a.smoke, gbc_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
