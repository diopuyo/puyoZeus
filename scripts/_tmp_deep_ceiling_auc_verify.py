"""深化天井 (build_ceiling_chain bitboardバッチ深化版) の win-AUC検証。

コーディネータ指示 (2026-07-22): 深化した天井が current_max_chain を
「勝敗予測で」本当に超えるかを、既存 A' 検証の枠組み
(GroupKFold(video_id,5fold) OOF AUC、フェーズ分割=序盤/中盤/終盤 by tsumo三分位、
scripts/model_indicator_win.py / scripts/_verify_ukeyasusa_subchain_adoption_2026-07.py
の既存手法を流用) で検証する。

4構成を横並び比較:
    1) current_max_chain 単体 (diff特徴のみ)
    2) 深化天井(deep_ceiling_raw) 単体
    3) 伸びしろ (deep_ceiling_raw - current_max_chain_raw) 単体
    4) 両方入り (current_max_chain + deep_ceiling_raw)

クロスチェック: HistGBC(GroupKFold OOF) と 単純diff値そのものの AUC
(モデル無し、point-biserial相当) の2手法で符号が安定するかも確認する。

前提: scripts/_tmp_deep_ceiling_gen.py の出力
(data/indicators_v2/study/deep_ceiling_video_result.csv) が存在すること。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_deep_ceiling_auc_verify
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_indicator_win import (  # noqa: E402
    N_FOLDS, TSUMO_EARLY_RATIO, TSUMO_LATE_RATIO,
    pair_sides_for_win, build_features, run_oof_classifier,
)

DEEP_CEILING_CSV = Path("data/indicators_v2/study/deep_ceiling_video_result.csv")
MAX_TDIFF = 1.0

CONFIGS: dict[str, tuple[str, ...]] = {
    "current_max_chain": ("current_max_chain_raw",),
    "deep_ceiling": ("deep_ceiling_raw",),
    "growth_gap(伸びしろ)": ("deep_ceiling_margin",),
    "both": ("current_max_chain_raw", "deep_ceiling_raw"),
}


def _phase_masks(paired: pd.DataFrame) -> dict[str, np.ndarray]:
    tsumo = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(tsumo, TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(tsumo, TSUMO_LATE_RATIO))
    return {
        "序盤": tsumo <= q33,
        "中盤": (tsumo > q33) & (tsumo <= q67),
        "終盤": tsumo > q67,
    }


def _auc_histgbc(
    paired: pd.DataFrame, feat_cols: tuple[str, ...], y: np.ndarray,
    groups: np.ndarray, mask: "np.ndarray | None" = None,
) -> "tuple[float, int]":
    """diff特徴 + GroupKFold OOF HistGBC の AUC (ukeyasusa検証と同一手法)。"""
    feat = build_features(paired, list(feat_cols))
    cols = [f"{c}_diff" for c in feat_cols]
    X = feat[cols].fillna(0.0).values
    yy, gg = y, groups
    if mask is not None:
        X, yy, gg = X[mask], y[mask], groups[mask]
    n_unique = len(np.unique(gg))
    folds = min(N_FOLDS, max(2, n_unique))
    if len(X) < 20 or len(np.unique(yy)) < 2:
        return float("nan"), len(X)
    oof, _ = run_oof_classifier(X, yy, gg, folds)
    valid = ~np.isnan(oof[:, 0])
    if len(np.unique(yy[valid])) < 2:
        return float("nan"), int(valid.sum())
    auc = float(roc_auc_score(yy[valid], oof[valid, 1]))
    return auc, int(valid.sum())


def _auc_raw_diff_only(
    paired: pd.DataFrame, feat_col: str, y: np.ndarray, mask: "np.ndarray | None" = None,
) -> "tuple[float, int]":
    """モデル無し・単純diff値そのものの AUC (point-biserial相当のクロスチェック手法)。

    1特徴のみ対応 (伸びしろ・単体指標の符号安定性確認用)。
    """
    diff_col = f"{feat_col}_diff"
    feat = build_features(paired, [feat_col])
    score = feat[diff_col].fillna(0.0).values
    yy = y
    if mask is not None:
        score, yy = score[mask], y[mask]
    if len(score) < 20 or len(np.unique(yy)) < 2:
        return float("nan"), len(score)
    auc = float(roc_auc_score(yy, score))
    auc = max(auc, 1.0 - auc)  # 符号不定 (正/負どちらの方向でも「効いているか」を見る)
    return auc, len(score)


def main() -> None:
    print("=== データ読み込み ===")
    df = pd.read_csv(DEEP_CEILING_CSV)
    df = df.dropna(subset=["video_id", "side", "won"]).copy()
    df["won"] = df["won"].astype(int)
    print(f"  読み込み行数: {len(df)}")

    paired = pair_sides_for_win(df, MAX_TDIFF)
    print(f"  ペアリング後: {len(paired)} 行")
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    masks = _phase_masks(paired)

    print()
    print("=" * 86)
    print("  深化天井 win-AUC 検証 (HistGBC, GroupKFold OOF, diff特徴)")
    print("=" * 86)
    header = f"  {'構成':<22}  {'全体':>7}  " + "  ".join(f"{p:>7}" for p in masks)
    print(header)
    print("  " + "-" * (len(header) - 2))

    results: dict[str, dict[str, float]] = {}
    ns: dict[str, dict[str, int]] = {}
    for name, cols in CONFIGS.items():
        row: dict[str, float] = {}
        n_row: dict[str, int] = {}
        auc_all, n_all = _auc_histgbc(paired, cols, y, groups)
        row["全体"] = auc_all
        n_row["全体"] = n_all
        line = f"  {name:<22}  {auc_all:>7.4f}  "
        for phase, mask in masks.items():
            auc_p, n_p = _auc_histgbc(paired, cols, y, groups, mask)
            row[phase] = auc_p
            n_row[phase] = n_p
            line += f"{auc_p:>7.4f}  "
        print(line)
        results[name] = row
        ns[name] = n_row

    print()
    print("  --- n (サンプル数、phase別) ---")
    for name in CONFIGS:
        print(f"  {name:<22}  " + "  ".join(f"{p}:{ns[name][p]}" for p in ["全体", *masks]))

    print()
    print("  --- current_max_chain 比の差分 (deep_ceiling / growth_gap / both) ---")
    base = results["current_max_chain"]
    for name in ("deep_ceiling", "growth_gap(伸びしろ)", "both"):
        deltas = " ".join(
            f"{p}:{results[name][p] - base[p]:+.4f}" for p in ["全体", *masks]
        )
        print(f"  {name:<22}  {deltas}")

    print()
    print("=" * 86)
    print("  クロスチェック: 単純diff値そのもののAUC (モデル無し、符号安定性確認)")
    print("=" * 86)
    for name, col in (
        ("current_max_chain", "current_max_chain_raw"),
        ("deep_ceiling", "deep_ceiling_raw"),
        ("growth_gap(伸びしろ)", "deep_ceiling_margin"),
    ):
        auc_all, n_all = _auc_raw_diff_only(paired, col, y)
        line = f"  {name:<22}  全体={auc_all:.4f}(n={n_all})  "
        for phase, mask in masks.items():
            auc_p, n_p = _auc_raw_diff_only(paired, col, y, mask)
            line += f"{phase}={auc_p:.4f}(n={n_p})  "
        print(line)

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
