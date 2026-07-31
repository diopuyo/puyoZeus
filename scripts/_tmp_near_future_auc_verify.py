"""近未来最大火力 (K=1..5) の win-AUC 検証。

コーディネータ指示 (2026-07-22): current_max_chain (K=0=今すぐ) と
near_future_firepower K=1..5 を、フェーズ別 (序盤/中盤/終盤) で横並び
単純diff-AUC (point-biserial相当、モデル無し) 比較する。どのKがどのフェーズで
current_max_chain を上回るかを表にする。4動画・薄さのノイズ幅と95%CIを併記。

scripts/_tmp_ama_ceiling_auc_verify.py と同じ突合・ペアリング手順を踏襲する
(scripts/model_indicator_win.py の pair_sides_for_win/build_features/
フェーズ三分位定数を再利用)。

得点はスケールが大きい (数万点) ため、コーディネータ指示に従い各K列を
min-max正規化 (0-1) してから diff-AUC を計算する (単一特徴の diff-AUC は
アフィン変換で数値的に不変なので raw と同じ値になるが、指示通り明示適用する)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_near_future_auc_verify
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_indicator_win import (  # noqa: E402
    TSUMO_EARLY_RATIO, TSUMO_LATE_RATIO, pair_sides_for_win, build_features,
)

NEAR_FUTURE_CSV = Path("data/indicators_v2/study/near_future_firepower_video_result.csv")
MAX_TDIFF: float = 1.0
K_LEVELS: "tuple[int, ...]" = (1, 2, 3, 4, 5)

N_BOOTSTRAP: int = 500
BOOTSTRAP_SEED: int = 20260722
MIN_BOOTSTRAP_ROWS: int = 20
MIN_BOOTSTRAP_VALID: int = 20


def _phase_masks(paired: pd.DataFrame) -> "dict[str, np.ndarray]":
    tsumo = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(tsumo, TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(tsumo, TSUMO_LATE_RATIO))
    return {
        "序盤": tsumo <= q33,
        "中盤": (tsumo > q33) & (tsumo <= q67),
        "終盤": tsumo > q67,
    }


def _minmax_normalize(paired: pd.DataFrame, col_1p: str, col_2p: str) -> pd.DataFrame:
    """1P/2P込みでmin-max正規化する (0-1)。単一特徴のdiff-AUCには数値的に無関係。"""
    both = pd.concat([paired[col_1p], paired[col_2p]])
    lo, hi = float(both.min()), float(both.max())
    span = max(hi - lo, 1e-9)
    out = paired.copy()
    out[col_1p] = (paired[col_1p] - lo) / span
    out[col_2p] = (paired[col_2p] - lo) / span
    return out


def _diff_auc(
    paired: pd.DataFrame, feat_col: str, y: np.ndarray, mask: "np.ndarray | None" = None,
) -> "tuple[float, int]":
    feat = build_features(paired, [feat_col])
    diff_col = f"{feat_col}_diff"
    score = feat[diff_col].fillna(0.0).values
    yy = y
    if mask is not None:
        score, yy = score[mask], y[mask]
    if len(score) < 20 or len(np.unique(yy)) < 2:
        return float("nan"), len(score)
    auc = float(roc_auc_score(yy, score))
    auc = max(auc, 1.0 - auc)
    return auc, len(score)


def _bootstrap_ci(
    paired: pd.DataFrame, feat_col: str, y: np.ndarray, mask: "np.ndarray | None",
) -> "tuple[float, float]":
    """video_id単位のクラスタブートストラップで AUC の 95%CI を推定する (参考値)。"""
    if mask is not None:
        sub = paired[mask].reset_index(drop=True)
        yy = y[mask]
    else:
        sub = paired.reset_index(drop=True)
        yy = y
    videos = sub["video_id_1p"].unique()
    if len(videos) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    aucs: "list[float]" = []
    video_arr = sub["video_id_1p"].values
    for _ in range(N_BOOTSTRAP):
        sampled_videos = rng.choice(videos, size=len(videos), replace=True)
        idx = np.concatenate([np.where(video_arr == v)[0] for v in sampled_videos])
        if len(idx) < MIN_BOOTSTRAP_ROWS:
            continue
        sub_boot = sub.iloc[idx]
        y_boot = yy[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        auc, _ = _diff_auc(sub_boot, feat_col, y_boot)
        if not np.isnan(auc):
            aucs.append(auc)
    if len(aucs) < MIN_BOOTSTRAP_VALID:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, 2.5))
    hi = float(np.percentile(aucs, 97.5))
    return lo, hi


def main() -> None:
    print("=== データ読み込み ===")
    df = pd.read_csv(NEAR_FUTURE_CSV)
    print(f"  読み込み行数: {len(df)}")
    df = df.dropna(subset=["video_id", "side", "won"]).copy()
    df["won"] = df["won"].astype(int)

    paired = pair_sides_for_win(df, MAX_TDIFF)
    n_videos = paired["video_id_1p"].nunique()
    print(f"  ペアリング後: {len(paired)} 行 (video数: {n_videos})")
    y = paired["won_1p"].astype(int).values
    masks = _phase_masks(paired)

    configs: "list[tuple[str, pd.DataFrame, str]]" = [
        ("current_max_chain(K=0=今すぐ)", paired, "current_max_chain_raw"),
    ]
    for k in K_LEVELS:
        col = f"near_future_firepower_k{k}_raw"
        col_1p, col_2p = f"{col}_1p", f"{col}_2p"
        # ここで build_features 呼び出し前に 1P/2P 列が既に paired に含まれている
        # ことを利用し、min-max 正規化する (コーディネータ指示)。
        normed = _minmax_normalize(paired, col_1p, col_2p)
        configs.append((f"near_future K={k} (正規化)", normed, col))

    print()
    print("=" * 100)
    print("  主指標: 単純diff値AUC (point-biserial相当、モデル無し)")
    print("=" * 100)
    header = f"  {'指標':<30}  {'全体':>7}  " + "  ".join(f"{p:>7}" for p in masks)
    print(header)
    print("  " + "-" * (len(header) - 2))

    results: "dict[str, dict[str, tuple[float, int]]]" = {}
    for name, data, col in configs:
        row: "dict[str, tuple[float, int]]" = {}
        auc_all, n_all = _diff_auc(data, col, y)
        row["全体"] = (auc_all, n_all)
        line = f"  {name:<30}  {auc_all:>7.4f}  "
        for phase, mask in masks.items():
            auc_p, n_p = _diff_auc(data, col, y, mask)
            row[phase] = (auc_p, n_p)
            line += f"{auc_p:>7.4f}  "
        print(line)
        results[name] = row

    print()
    print("  --- n (サンプル数、phase別、全指標共通) ---")
    first_name = configs[0][0]
    print(
        f"  共通  " + "  ".join(f"{p}:{results[first_name][p][1]}" for p in ["全体", *masks]),
    )

    print()
    print("  --- current_max_chain(K=0) 比の差分 ---")
    base = results["current_max_chain(K=0=今すぐ)"]
    for k in K_LEVELS:
        name = f"near_future K={k} (正規化)"
        deltas = " ".join(
            f"{p}:{results[name][p][0] - base[p][0]:+.4f}" for p in ["全体", *masks]
        )
        print(f"  {name:<30}  {deltas}")

    print()
    print("=" * 100)
    print(
        f"  ノイズ幅 (video単位クラスタブートストラップ 95%CI, n_boot={N_BOOTSTRAP}, "
        f"対象video数={n_videos} のため参考値・幅広め)",
    )
    print("=" * 100)
    for name, data, col in configs:
        line = f"  {name:<30}\n"
        for phase, mask in masks.items():
            lo, hi = _bootstrap_ci(data, col, y, mask)
            point = results[name][phase][0]
            line += f"    {phase}: 点推定={point:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]\n"
        print(line)

    print()
    print("=" * 100)
    print("  どのKがどのフェーズで current_max_chain を上回るか (◯=上回る, ×=下回る/同値)")
    print("=" * 100)
    header2 = f"  {'':<12}  " + "  ".join(f"{p:>6}" for p in ["全体", *masks])
    print(header2)
    for k in K_LEVELS:
        name = f"near_future K={k} (正規化)"
        marks = []
        for p in ["全体", *masks]:
            beats = results[name][p][0] > base[p][0]
            marks.append(f"{'○' if beats else '×':>6}")
        print(f"  {'K=' + str(k):<12}  " + "  ".join(marks))

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
