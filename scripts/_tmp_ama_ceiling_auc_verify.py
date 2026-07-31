"""得点ベース飽和火力 (ama_ceiling_score) の win-AUC 検証。

コーディネータ指示 (2026-07-22): 得点ベースの天井が勝敗を予測するかを最優先で
検証する (難所割りより価値判定が先)。主指標は単純diff値のAUC
(point-biserial相当、モデル無し)。前回 HistGBC は1-2特徴で異常値が出た前例が
あるため主指標にしない。

現在の3指標をフェーズ別(序盤/中盤/終盤 by tsumo三分位)で横並び比較する:
    1) current_max_chain_raw (今の火力、5色無制限)
    2) deep_ceiling_raw (前回の連鎖数ベース深化天井、5色無制限、chain_bitboard高速化版)
    3) ama_ceiling_score (今回の得点ベース飽和火力、試合別4色限定、ama構成ループ単線greedy)

ama_ceiling_video_result.csv (scripts/_tmp_ama_ceiling_gen.py の出力) と
deep_ceiling_video_result.csv (前回資産) を (video_id, side, game_idx, t_sec) で
内部結合し、完全に同一のサンプル集合上で3指標を比較する (突合が同一である
ことを確認済み: 両者とも matched=1913 行、キー完全一致)。

得点 (ama_ceiling_score) は数万点スケールのため、コーディネータ指示に従い
min-max 正規化 (0-1) してから diff-AUC を計算する。単一特徴の diff-AUC は
アフィン変換 (min-max) で数値的に不変 (AUCは順位のみに依存するため) だが、
指示通り明示的に適用する。

サンプルの薄さ (対象4動画: video_29/35/36/37) 由来のノイズ幅を、video単位の
ブートストラップ95%CIで併記する (誇張しない)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_ama_ceiling_auc_verify
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

AMA_CEILING_CSV = Path("data/indicators_v2/study/ama_ceiling_video_result.csv")
DEEP_CEILING_CSV = Path("data/indicators_v2/study/deep_ceiling_video_result.csv")
MAX_TDIFF: float = 1.0
MERGE_KEY: "list[str]" = ["video_id", "side", "game_idx", "t_sec"]

# video単位ブートストラップ (ノイズ幅推定、サンプルの薄さを明示するため)。
N_BOOTSTRAP: int = 500
BOOTSTRAP_SEED: int = 20260722
MIN_BOOTSTRAP_ROWS: int = 20
MIN_BOOTSTRAP_VALID: int = 20


def _load_merged() -> pd.DataFrame:
    """ama_ceiling と deep_ceiling を同一サンプル集合上で内部結合する。"""
    ama = pd.read_csv(AMA_CEILING_CSV)
    deep = pd.read_csv(DEEP_CEILING_CSV)
    deep_cols = MERGE_KEY + ["deep_ceiling_raw", "deep_ceiling_margin"]
    merged = ama.merge(deep[deep_cols], on=MERGE_KEY, how="inner")
    return merged


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
    """得点系列を1P/2P込みでmin-max正規化する (0-1)。

    注記: 単一特徴の diff-AUC はアフィン変換 (min-max) で数値的に不変
    (AUC は順位のみに依存する)。コーディネータ指示に従い明示的に適用するが、
    結果 (AUC値) を変えるものではないことをここに明記する。
    """
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
    """モデル無し・単純diff値そのものの AUC (point-biserial相当)。"""
    feat = build_features(paired, [feat_col])
    diff_col = f"{feat_col}_diff"
    score = feat[diff_col].fillna(0.0).values
    yy = y
    if mask is not None:
        score, yy = score[mask], y[mask]
    if len(score) < 20 or len(np.unique(yy)) < 2:
        return float("nan"), len(score)
    auc = float(roc_auc_score(yy, score))
    auc = max(auc, 1.0 - auc)  # 符号不定 (正/負どちらの方向でも「効いているか」を見る)
    return auc, len(score)


def _bootstrap_ci(
    paired: pd.DataFrame, feat_col: str, y: np.ndarray, mask: "np.ndarray | None",
) -> "tuple[float, float]":
    """video_id単位のクラスタブートストラップで AUC の 95%CI を推定する。

    対象4動画 (video_29/35/36/37) のみなので、CI は参考値として幅広くなる
    (n_video=4 由来のノイズ幅を誇張せずそのまま報告する)。
    """
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
    print("=== データ読み込み・突合 ===")
    merged = _load_merged()
    print(f"  ama_ceiling + deep_ceiling 突合行数: {len(merged)}")
    merged = merged.dropna(subset=["video_id", "side", "won"]).copy()
    merged["won"] = merged["won"].astype(int)

    paired = pair_sides_for_win(merged, MAX_TDIFF)
    n_videos = paired["video_id_1p"].nunique()
    print(f"  ペアリング後: {len(paired)} 行 (video数: {n_videos})")
    y = paired["won_1p"].astype(int).values
    masks = _phase_masks(paired)

    # 得点のmin-max正規化 (コーディネータ指示、数値的にはAUC不変)。
    paired_norm = _minmax_normalize(paired, "ama_ceiling_score_1p", "ama_ceiling_score_2p")

    configs: "list[tuple[str, pd.DataFrame, str]]" = [
        ("current_max_chain(今の火力)", paired, "current_max_chain_raw"),
        ("deep_ceiling(連鎖数版天井)", paired, "deep_ceiling_raw"),
        ("ama_ceiling(得点版天井,正規化)", paired_norm, "ama_ceiling_score"),
    ]

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
    print("  --- n (サンプル数、phase別) ---")
    for name, row in results.items():
        print(f"  {name:<30}  " + "  ".join(f"{p}:{row[p][1]}" for p in ["全体", *masks]))

    print()
    print("  --- current_max_chain 比の差分 (deep_ceiling / ama_ceiling) ---")
    base = results["current_max_chain(今の火力)"]
    for name in ("deep_ceiling(連鎖数版天井)", "ama_ceiling(得点版天井,正規化)"):
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

    print("=== 完了 ===")


if __name__ == "__main__":
    main()
