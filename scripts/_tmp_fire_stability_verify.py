"""火力の受けの多さ (fire_stability, K=2,4,6) の win-AUC 検証。

user提案#30: 検証済み中盤本命「受けやすさ(ukeyasusa)」の火力版。
near_future_fire_power と同じ突合・単純diff-AUC(point-biserial)手法で、
current_max_chain・near_future_fire・fire_stability を横並び比較する。
「火力安定性が独立の寄与を持つか (near_futureと相関しつつ別軸か)」を
Pearson相関で確認する。

scripts/_tmp_near_future_prod_verify.py と同じ突合手順 (video_29-38,
_compute_active_colors_by_game による試合単位4色) を踏襲する。

閾値比較 (0.8 vs 0.9) も併記する (コーディネータ指示)。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_fire_stability_verify
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._tmp_validate_build_ceiling_subset import (  # noqa: E402
    _load_npz_for_video, _match_grid, _grid_to_board, TARGET_VIDEO_IDS, BOARDS_DIR,
)
from scripts._tmp_ama_builder import _compute_active_colors_by_game  # noqa: E402
from scripts.model_indicator_win import (  # noqa: E402
    TSUMO_EARLY_RATIO, TSUMO_LATE_RATIO, pair_sides_for_win, build_features,
)
import src.indicators_v2 as iv  # noqa: E402

LABELED_WIN_CSV = Path("data/indicators_v2/study/labeled_win.csv")
OUT_CSV = Path("data/indicators_v2/study/fire_stability_verify_result.csv")
MAX_TDIFF: float = 1.0

# 検証用の閾値比較 (コーディネータ指示: 0.8/0.9 を試す)。
THRESHOLD_CANDIDATES: "tuple[float, ...]" = (0.8, 0.9)

N_BOOTSTRAP: int = 500
BOOTSTRAP_SEED: int = 20260722
MIN_BOOTSTRAP_ROWS: int = 20
MIN_BOOTSTRAP_VALID: int = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _generate() -> pd.DataFrame:
    """近未来最大火力 (K5参考) + 火力安定性 (閾値0.8/0.9 x K2,4,6) を計算する。"""
    df = pd.read_csv(LABELED_WIN_CSV)
    df = df[df["video_id"].isin(TARGET_VIDEO_IDS)]
    df = df[df["won"].notna()].reset_index(drop=True)
    logger.info("対象行数 (won付き): %d", len(df))

    npz_cache = {vid: _load_npz_for_video(vid) for vid in TARGET_VIDEO_IDS}
    active_colors_cache: "dict[tuple[str, int], tuple[int, ...]]" = {}
    for vid in TARGET_VIDEO_IDS:
        stem = vid.replace("video_", "v")
        active_colors_cache.update(_compute_active_colors_by_game(BOARDS_DIR / f"{stem}.npz"))
    logger.info("active_colors (試合単位) 計算完了: %d 組", len(active_colors_cache))

    rows_out: "list[dict]" = []
    n_matched = 0
    n_missed = 0
    n_missed_colors = 0
    t_start = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        vid = str(row["video_id"])
        side = str(row["side"])
        game_idx = int(row["game_idx"])
        t_sec = float(row["t_sec"])
        grid = _match_grid(npz_cache[vid], side, game_idx, t_sec)
        if grid is None:
            n_missed += 1
            continue
        colors = active_colors_cache.get((vid, game_idx))
        if colors is None:
            n_missed_colors += 1
            continue
        board = _grid_to_board(grid)
        t0 = time.perf_counter()

        nf = iv.near_future_fire_power(board, elapsed_sec=0.0, active_colors=colors)
        row_dict = row.to_dict()
        for k in iv.NEAR_FUTURE_K_LEVELS:
            row_dict[f"near_future_fire_k{k}_raw"] = nf.values[k].raw

        for thr in THRESHOLD_CANDIDATES:
            fs = iv.fire_stability(board, active_colors=colors, threshold_ratio=thr)
            suffix = "" if thr == iv.FIRE_STABILITY_THRESHOLD_RATIO else f"_thr{int(thr*100)}"
            for k in iv.FIRE_STABILITY_K_LEVELS:
                row_dict[f"fire_stability_k{k}{suffix}_raw"] = fs.values[k].raw

        cost = time.perf_counter() - t0
        row_dict["cost_sec"] = cost
        rows_out.append(row_dict)
        n_matched += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            logger.info(
                "進捗: %d/%d matched=%d missed=%d missed_colors=%d elapsed=%.0fs",
                i + 1, len(df), n_matched, n_missed, n_missed_colors, elapsed,
            )

    logger.info("完了: matched=%d missed=%d missed_colors=%d", n_matched, n_missed, n_missed_colors)
    out_df = pd.DataFrame(rows_out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    logger.info("結果 CSV 保存: %s", OUT_CSV)
    cost_ms = out_df["cost_sec"] * 1000.0
    logger.info(
        "cost: mean=%.1fms p95=%.1fms max=%.1fms", cost_ms.mean(), cost_ms.quantile(0.95),
        cost_ms.max(),
    )
    return out_df


def _phase_masks(paired: pd.DataFrame) -> "dict[str, np.ndarray]":
    tsumo = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(tsumo, TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(tsumo, TSUMO_LATE_RATIO))
    return {
        "序盤": tsumo <= q33,
        "中盤": (tsumo > q33) & (tsumo <= q67),
        "終盤": tsumo > q67,
    }


def _diff_auc(
    paired: pd.DataFrame, feat_col: str, y: np.ndarray, mask: "np.ndarray | None" = None,
) -> "tuple[float, int]":
    feat = build_features(paired, [feat_col])
    score = feat[f"{feat_col}_diff"].fillna(0.0).values
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
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _verify(df: pd.DataFrame) -> None:
    df = df.dropna(subset=["video_id", "side", "won"]).copy()
    df["won"] = df["won"].astype(int)
    paired = pair_sides_for_win(df, MAX_TDIFF)
    n_videos = paired["video_id_1p"].nunique()
    y = paired["won_1p"].astype(int).values
    masks = _phase_masks(paired)
    logger.info("ペアリング後: %d 行 (video数: %d)", len(paired), n_videos)

    configs = [
        ("current_max_chain", "current_max_chain_raw"),
        ("near_future_fire_k5", "near_future_fire_k5_raw"),
        ("fire_stability_k2(thr0.8)", "fire_stability_k2_raw"),
        ("fire_stability_k4(thr0.8)", "fire_stability_k4_raw"),
        ("fire_stability_k6(thr0.8)", "fire_stability_k6_raw"),
        ("fire_stability_k6(thr0.9)", "fire_stability_k6_thr90_raw"),
    ]

    print()
    print("=" * 100)
    print("  主指標: 単純diff値AUC (point-biserial相当、モデル無し)")
    print("=" * 100)
    header = f"  {'指標':<30}  {'全体':>7}  " + "  ".join(f"{p:>7}" for p in masks)
    print(header)
    print("  " + "-" * (len(header) - 2))
    results: "dict[str, dict[str, tuple[float, int]]]" = {}
    for name, col in configs:
        row: "dict[str, tuple[float, int]]" = {}
        auc_all, n_all = _diff_auc(paired, col, y)
        row["全体"] = (auc_all, n_all)
        line = f"  {name:<30}  {auc_all:>7.4f}  "
        for phase, mask in masks.items():
            auc_p, n_p = _diff_auc(paired, col, y, mask)
            row[phase] = (auc_p, n_p)
            line += f"{auc_p:>7.4f}  "
        print(line)
        results[name] = row

    print()
    print("  --- current_max_chain 比の差分 ---")
    base = results["current_max_chain"]
    for name, _col in configs[1:]:
        deltas = " ".join(
            f"{p}:{results[name][p][0] - base[p][0]:+.4f}" for p in ["全体", *masks]
        )
        print(f"  {name:<30}  {deltas}")

    print()
    print("=" * 100)
    print(f"  ノイズ幅 (video単位クラスタブートストラップ 95%CI, n_boot={N_BOOTSTRAP}, 対象video数={n_videos})")
    print("=" * 100)
    for name, col in configs:
        line = f"  {name:<30}\n"
        for phase, mask in masks.items():
            lo, hi = _bootstrap_ci(paired, col, y, mask)
            point = results[name][phase][0]
            line += f"    {phase}: 点推定={point:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]\n"
        print(line)

    print()
    print("=" * 100)
    print("  独立性チェック: fire_stability と near_future_fire_k5 の相関 (diff値同士、Pearson r)")
    print("=" * 100)
    nf_diff = build_features(paired, ["near_future_fire_k5"])["near_future_fire_k5_diff"].fillna(0.0).values
    for k in (2, 4, 6):
        fs_diff = build_features(paired, [f"fire_stability_k{k}"])[f"fire_stability_k{k}_diff"].fillna(0.0).values
        r = float(np.corrcoef(nf_diff, fs_diff)[0, 1])
        print(f"  fire_stability_k{k} vs near_future_fire_k5: r={r:.4f}")

    print("\n=== 完了 ===")


def main() -> None:
    df = _generate()
    _verify(df)


if __name__ == "__main__":
    main()
