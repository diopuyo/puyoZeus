"""平均ツモ期待火力 (expected_fire_power, K=1..4) の win-AUC 検証。

user新指標: near_future_fire_power (理想ツモ=best case) の逆、K=1,2は全ツモ
色パターン厳密列挙、K=3,4はモンテカルロ近似による火力の期待値 (expected case)。
near_future_fire_power/fire_stability と同じ突合・単純diff-AUC手法で、
current_max_chain・near_future_fire_k5・旧fire_stability_k6・
新expected_fire_k1〜k4 を横並び比較する。

⚠️ コスト注記 (正直な記録): expected_fire_power は重い (実測1-3秒/盤面、
scripts/_tmp_bench_expected_fire.py 参照)。全1913行を計算すると数十分〜1時間
規模になるため、既定でランダムサブサンプル (SAMPLE_SIZE 行) して実行する
(先行検証 _tmp_validate_saturation_subset.py 等と同じ前例に準拠)。

scripts/_tmp_fire_stability_verify.py と同じ突合手順
(video_29-38, _compute_active_colors_by_game) を踏襲する。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_expected_fire_verify
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
OUT_CSV = Path("data/indicators_v2/study/expected_fire_verify_result.csv")
MAX_TDIFF: float = 1.0

N_BOOTSTRAP: int = 500
BOOTSTRAP_SEED: int = 20260722
MIN_BOOTSTRAP_ROWS: int = 20
MIN_BOOTSTRAP_VALID: int = 20

# expected_fire_power のコストが重い (1-3秒/盤面) ため、行数を間引く
# (正直な記録: 全1913行を回すと数十分〜1時間規模になるため)。
SAMPLE_SIZE: int = 700
SAMPLE_SEED: int = 20260722

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _match_all_rows(df: pd.DataFrame) -> "list[dict]":
    """全行に対し盤面照合のみ行う (安価、指標計算はまだしない)。

    matched 行だけをサブサンプルしてから重い expected_fire_power を計算する
    ため、まず照合を分離する (盤面照合自体は npz 配列アクセスのみで安価)。
    """
    npz_cache = {vid: _load_npz_for_video(vid) for vid in TARGET_VIDEO_IDS}
    active_colors_cache: "dict[tuple[str, int], tuple[int, ...]]" = {}
    for vid in TARGET_VIDEO_IDS:
        stem = vid.replace("video_", "v")
        active_colors_cache.update(_compute_active_colors_by_game(BOARDS_DIR / f"{stem}.npz"))
    logger.info("active_colors (試合単位) 計算完了: %d 組", len(active_colors_cache))

    matched: "list[dict]" = []
    n_missed = 0
    n_missed_colors = 0
    for _, row in df.iterrows():
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
        matched.append({"row": row, "grid": grid, "colors": colors})
    logger.info(
        "照合完了: matched=%d missed=%d missed_colors=%d (total=%d)",
        len(matched), n_missed, n_missed_colors, len(df),
    )
    return matched


def _generate() -> pd.DataFrame:
    """current_max_chain(既存列流用)・near_future_fire_k5・fire_stability_k6・

    expected_fire_k1..k4 を matched 行 (コスト対策でサブサンプル) に対して計算する。
    """
    df = pd.read_csv(LABELED_WIN_CSV)
    df = df[df["video_id"].isin(TARGET_VIDEO_IDS)]
    df = df[df["won"].notna()].reset_index(drop=True)
    logger.info("対象行数 (won付き): %d", len(df))

    matched_all = _match_all_rows(df)
    if len(matched_all) > SAMPLE_SIZE:
        rng = np.random.default_rng(SAMPLE_SEED)
        idx = rng.choice(len(matched_all), size=SAMPLE_SIZE, replace=False)
        matched = [matched_all[i] for i in idx]
        logger.info(
            "expected_fire_power のコストが重いため matched 行を乱数サブサンプル: "
            "%d -> %d 行 (正直な記録)",
            len(matched_all), len(matched),
        )
    else:
        matched = matched_all

    rows_out: "list[dict]" = []
    n_matched = 0
    t_start = time.time()
    total = len(matched)
    for i, entry in enumerate(matched):
        row, grid, colors = entry["row"], entry["grid"], entry["colors"]
        board = _grid_to_board(grid)
        t0 = time.perf_counter()

        nf = iv.near_future_fire_power(board, elapsed_sec=0.0, active_colors=colors)
        fs = iv.fire_stability(board, active_colors=colors)
        ef = iv.expected_fire_power(board, elapsed_sec=0.0, active_colors=colors)
        cost = time.perf_counter() - t0

        row_dict = row.to_dict()
        row_dict["near_future_fire_k5_raw"] = nf.values[5].raw
        row_dict["fire_stability_k6_raw"] = fs.values[6].raw
        for k in iv.EXPECTED_FIRE_K_LEVELS:
            row_dict[f"expected_fire_k{k}_raw"] = ef.values[k].raw
        row_dict["cost_sec"] = cost
        rows_out.append(row_dict)
        n_matched += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            logger.info(
                "進捗: %d/%d 計算済み elapsed=%.0fs eta=%.0fs", i + 1, total, elapsed, eta,
            )

    logger.info("完了: %d 行の指標計算を終えた (サブサンプル後)", n_matched)
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
        ("fire_stability_k6", "fire_stability_k6_raw"),
        ("expected_fire_k1", "expected_fire_k1_raw"),
        ("expected_fire_k2", "expected_fire_k2_raw"),
        ("expected_fire_k3", "expected_fire_k3_raw"),
        ("expected_fire_k4", "expected_fire_k4_raw"),
    ]

    print()
    print("=" * 100)
    print("  主指標: 単純diff値AUC (point-biserial相当、モデル無し)")
    print("=" * 100)
    header = f"  {'指標':<24}  {'全体':>7}  " + "  ".join(f"{p:>7}" for p in masks)
    print(header)
    print("  " + "-" * (len(header) - 2))
    results: "dict[str, dict[str, tuple[float, int]]]" = {}
    for name, col in configs:
        row: "dict[str, tuple[float, int]]" = {}
        auc_all, n_all = _diff_auc(paired, col, y)
        row["全体"] = (auc_all, n_all)
        line = f"  {name:<24}  {auc_all:>7.4f}  "
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
        print(f"  {name:<24}  {deltas}")

    print()
    print("=" * 100)
    print(f"  ノイズ幅 (video単位クラスタブートストラップ 95%CI, n_boot={N_BOOTSTRAP}, 対象video数={n_videos})")
    print("=" * 100)
    for name, col in configs:
        line = f"  {name:<24}\n"
        for phase, mask in masks.items():
            lo, hi = _bootstrap_ci(paired, col, y, mask)
            point = results[name][phase][0]
            line += f"    {phase}: 点推定={point:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]\n"
        print(line)

    print()
    print("=" * 100)
    print("  独立性チェック: expected_fire と near_future_fire_k5/fire_stability_k6 の相関 (Pearson r)")
    print("=" * 100)
    nf_diff = build_features(paired, ["near_future_fire_k5_raw"])["near_future_fire_k5_raw_diff"].fillna(0.0).values
    fs_diff = build_features(paired, ["fire_stability_k6_raw"])["fire_stability_k6_raw_diff"].fillna(0.0).values
    cmc_diff = build_features(paired, ["current_max_chain_raw"])["current_max_chain_raw_diff"].fillna(0.0).values
    for k in (1, 2, 3, 4):
        col = f"expected_fire_k{k}_raw"
        ef_diff = build_features(paired, [col])[f"{col}_diff"].fillna(0.0).values
        r_nf = float(np.corrcoef(nf_diff, ef_diff)[0, 1])
        r_fs = float(np.corrcoef(fs_diff, ef_diff)[0, 1])
        r_cmc = float(np.corrcoef(cmc_diff, ef_diff)[0, 1])
        print(
            f"  expected_fire_k{k}: vs near_future_fire_k5 r={r_nf:.4f}  "
            f"vs fire_stability_k6 r={r_fs:.4f}  vs current_max_chain r={r_cmc:.4f}",
        )

    print("\n=== 完了 ===")


def main() -> None:
    df = _generate()
    _verify(df)


if __name__ == "__main__":
    main()
