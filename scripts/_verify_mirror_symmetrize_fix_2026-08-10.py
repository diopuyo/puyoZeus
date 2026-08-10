"""_train_model 対称化バグ修正 (side反転で符号可変/不変列を区別) の効果検証。

アーキ設計 (案B-1)・コーダ実装 (2026-08-10) の検証スクリプト。
修正内容: scripts/visualize_advantage_overlay.py の `_train_model` は
side 入れ替え対称性のため全列を無条件反転したミラー標本を作っていたが、
`ojama_flat_score_diff` (np.abs ベースの side非依存な絶対量) まで反転すると
あり得ない値 (負のフラット度) が混入していた。 `_mirror_sign()` で
列ごとに符号可変/不変を区別するよう修正済み。

本スクリプトは **修正前 (--old-buggy) / 修正後 (既定)** の両方で
GroupKFold (video_id 単位) OOF 評価を行い:
  1. 全体 AUC (無悪化ゲート: 現状 0.6615 を下回らないこと)
  2. おじゃまフラット局面 / 非フラット局面の層別 AUC
     (_verify_color_ojama_interaction_2026-08-09.py と同じ層別閾値)
  3. permutation importance (color_puyo_x_ojama_flat_diff / ojama_flat_score_diff)
を比較レポートする。

使い方 (本測定・重い。 CPU負荷が高い間は避けること):
    python -m scripts._verify_mirror_symmetrize_fix_2026-08-10 --old-buggy
    python -m scripts._verify_mirror_symmetrize_fix_2026-08-10

スモーク (高速・小サンプルで動作確認のみ):
    python -m scripts._verify_mirror_symmetrize_fix_2026-08-10 --smoke
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
    COLOR_OJAMA_INTERACTION_COL,
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

# _verify_color_ojama_interaction_2026-08-09.py と同じ層別閾値 (踏襲)。
FLAT_HI: float = 0.8
FLAT_LO: float = 0.37

N_FOLDS: int = 5
PERM_N_REPEATS: int = 20
PERM_RANDOM_STATE: int = 42


def _mirror_sign_buggy(cols: list[str]) -> np.ndarray:
    """修正前の挙動を再現: 全列一律 -1 (無条件反転、バグ)。比較専用。"""
    return -np.ones(len(cols), dtype=float)


def _prepare(smoke: bool) -> tuple[pd.DataFrame, list[str], list[str]]:
    """paired 特徴量・使用列・video group 列を用意する。"""
    df = load_labeled_csv(TRAIN_CSV_PATH)
    if smoke:
        # スモーク: 動画3本だけに絞り高速化 (構造検証のみが目的)。
        vids = sorted(df["video_id"].unique())[:3]
        df = df[df["video_id"].isin(vids)].reset_index(drop=True)
    feat_cols = _resolve_features(df)
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    feat = build_features(paired, feat_cols)
    feat, cols = _add_interaction_columns(feat, feat_cols)
    feat["video_id_1p"] = paired["video_id_1p"].values
    feat["won_1p"] = paired["won_1p"].astype(int).values
    return feat, cols, feat_cols


def _flat_score_from_diff(feat: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """層別用: ojama_flat_score_diff 列 (無ければ全 NaN 扱いで層別スキップ)。"""
    col = f"{OJAMA_FLAT_COL}_diff"
    if col not in feat.columns:
        return np.full(len(feat), np.nan)
    return feat[col].fillna(0.0).values


def _auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def run(use_old_buggy: bool, smoke: bool, gbc_params: dict) -> None:
    label = "修正前(バグ再現)" if use_old_buggy else "修正後"
    print(f"\n{'='*70}\n{label}  (smoke={smoke})\n{'='*70}")
    feat, cols, feat_cols = _prepare(smoke)
    X = feat[cols].fillna(0.0).values
    y = feat["won_1p"].values
    groups = feat["video_id_1p"].values
    flat = _flat_score_from_diff(feat, cols)
    sign_fn = _mirror_sign_buggy if use_old_buggy else _mirror_sign

    n_splits = 2 if smoke else N_FOLDS
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    perm_importances: dict[str, list[float]] = {
        f"{COLOR_OJAMA_INTERACTION_COL}_diff": [],
        f"{OJAMA_FLAT_COL}_diff": [],
    }
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        sign = sign_fn(cols)
        X_tr_sym = np.vstack([X_tr, X_tr * sign])
        y_tr_sym = np.concatenate([y_tr, 1 - y_tr])
        model = HistGradientBoostingClassifier(**gbc_params)
        model.fit(X_tr_sym, y_tr_sym)
        oof[te_idx] = model.predict_proba(X[te_idx])[:, 1]
        for name in perm_importances:
            if name not in cols:
                continue
            idx = cols.index(name)
            pi = permutation_importance(
                model, X[te_idx], y[te_idx], n_repeats=PERM_N_REPEATS,
                random_state=PERM_RANDOM_STATE, scoring="roc_auc",
            )
            perm_importances[name].append(float(pi.importances_mean[idx]))
        print(f"  fold {fold+1}/{n_splits} 完了 (train={len(tr_idx)} test={len(te_idx)})")

    valid = ~np.isnan(oof)
    overall_auc = _auc(y[valid], oof[valid])
    print(f"\n全体 OOF AUC: {overall_auc:.4f}  (無悪化ゲート基準: 0.6615)")

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
            print(f"  層別[{name}] n={n:6d}  AUC={_auc(y[m], oof[m]):.4f}")
    else:
        print("  (ojama_flat_score_diff 列が無いため層別スキップ)")

    for name, vals in perm_importances.items():
        if not vals:
            print(f"  permutation importance[{name}]: 列なし (交互作用未使用)")
            continue
        print(f"  permutation importance[{name}]: 平均 {np.mean(vals):+.5f}"
              f" (fold別 {['%.5f' % v for v in vals]})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-buggy", action="store_true",
                     help="修正前の挙動 (全列無条件反転) を再現して評価する")
    ap.add_argument("--smoke", action="store_true",
                     help="動画3本・2fold・軽量パラメータで構造検証のみ行う")
    a = ap.parse_args()
    gbc_params = dict(GBC_PARAMS)
    if a.smoke:
        gbc_params["max_iter"] = 30
    run(a.old_buggy, a.smoke, gbc_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
