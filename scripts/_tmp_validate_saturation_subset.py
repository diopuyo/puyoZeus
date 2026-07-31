"""saturation_chain (忠実な飽和連鎖量、非発火構築ビーム) の軽量サブセット検証。

全再収集を避け、既存 data/indicators_v2/boards/v29-v38.npz (10動画) と
data/indicators_v2/study/labeled_win.csv (won ラベル) を突合して検証する。
scripts/_tmp_validate_build_ceiling_subset.py と同じ突合ロジックを再利用する。

検証内容:
    (a) saturation_chain(fill_ratio=0.93) が current_max_chain より
        実際に大きい連鎖を組めているか (平均差・分布)。
    (b) 1盤面あたりの計算コスト実測。
    (c) saturation_chain / 飽和余地 (saturation-current_max) の中盤 win-AUC
        (current_max_chain との比較)。
    (d) fill_ratio 0.88/0.93/0.98 の感度 (サブサンプルで実測)。

熱対策: 単プロセス・スレッド制限。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_validate_saturation_subset
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._tmp_validate_build_ceiling_subset import (  # noqa: E402
    _load_npz_for_video, _match_grid, TARGET_VIDEO_IDS,
)

# ============================
# 定数
# ============================

LABELED_WIN_CSV: Path = Path("data/indicators_v2/study/labeled_win.csv")

TSUMO_MID_LOW: float = 0.33
TSUMO_MID_HIGH: float = 0.67

SATURATION_BEAM_WIDTH: int = 6  # 既定値 (micro-bench で 40-75ms/盤面と確認済)
FILL_RATIO_SWEEP: tuple[float, ...] = (0.88, 0.93, 0.98)
FILL_RATIO_SWEEP_SAMPLE_N: int = 400  # スイープはサブサンプルでコスト抑制

MIN_VALID_ROWS: int = 10

OUT_CSV: Path = Path("data/indicators_v2/saturation_subset_result.csv")
OUT_SWEEP_CSV: Path = Path("data/indicators_v2/saturation_fillratio_sweep.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================
# セクション1: 主指標計算 (fill_ratio=0.93既定)
# ============================


def _grid_to_board(grid: np.ndarray) -> "Any":
    from src.board import Board
    return Board.from_list(grid.tolist())


def build_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """labeled_win.csv 抽出行に saturation_chain を計算して列追加する。"""
    from src.chain import ChainSimulator
    import src.indicators_v2 as iv

    sim = ChainSimulator()
    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    for vid in TARGET_VIDEO_IDS:
        npz_cache[vid] = _load_npz_for_video(vid)

    rows_out: list[dict] = []
    n_matched = 0
    n_missed = 0
    total_cost_sec = 0.0
    total = len(df)
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
        board = _grid_to_board(grid)

        csv_current_max = float(row["current_max_chain_raw"])

        t0 = time.perf_counter()
        sat = iv.saturation_chain(board, beam_width=SATURATION_BEAM_WIDTH, simulator=sim)
        cost = time.perf_counter() - t0
        total_cost_sec += cost

        row_dict = row.to_dict()
        row_dict["saturation_raw"] = sat.raw
        row_dict["saturation_score"] = sat.score
        row_dict["saturation_margin"] = sat.raw - csv_current_max
        row_dict["saturation_cost_sec"] = cost
        rows_out.append(row_dict)
        n_matched += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            logger.info(
                "進捗: %d/%d matched=%d missed=%d 平均コスト=%.1fms elapsed=%.0fs eta=%.0fs",
                i + 1, total, n_matched, n_missed,
                1000.0 * total_cost_sec / max(1, n_matched), elapsed, eta,
            )

    logger.info(
        "完了: matched=%d missed=%d (total=%d) 平均コスト=%.1fms/盤面",
        n_matched, n_missed, total, 1000.0 * total_cost_sec / max(1, n_matched),
    )
    return pd.DataFrame(rows_out)


# ============================
# セクション2: fill_ratio スイープ (サブサンプル)
# ============================


def run_fill_ratio_sweep(feat_df: pd.DataFrame) -> pd.DataFrame:
    """サブサンプルで fill_ratio 0.88/0.93/0.98 の感度を測定する。"""
    from src.chain import ChainSimulator
    import src.indicators_v2 as iv

    sim = ChainSimulator()
    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    for vid in TARGET_VIDEO_IDS:
        npz_cache[vid] = _load_npz_for_video(vid)

    sample_df = feat_df.dropna(subset=["saturation_raw"]).sample(
        n=min(FILL_RATIO_SWEEP_SAMPLE_N, len(feat_df)), random_state=0,
    )
    rows_out: list[dict] = []
    for _, row in sample_df.iterrows():
        vid = str(row["video_id"])
        side = str(row["side"])
        game_idx = int(row["game_idx"])
        t_sec = float(row["t_sec"])
        grid = _match_grid(npz_cache[vid], side, game_idx, t_sec)
        if grid is None:
            continue
        board = _grid_to_board(grid)
        rec = {
            "video_id": vid, "won": row.get("won"),
            "tsumo_count_rate": row.get("tsumo_count_rate"),
            "current_max_chain_raw": row.get("current_max_chain_raw"),
        }
        for fr in FILL_RATIO_SWEEP:
            v = iv.saturation_chain(board, fill_ratio=fr, simulator=sim)
            rec[f"saturation_raw_fr{fr}"] = v.raw
        rows_out.append(rec)
    return pd.DataFrame(rows_out)


# ============================
# セクション3: 分析
# ============================


def _safe_auc(y: np.ndarray, score: np.ndarray) -> tuple[float, int]:
    from sklearn.metrics import roc_auc_score
    mask = ~np.isnan(score) & ~np.isnan(y)
    n = int(mask.sum())
    if n < MIN_VALID_ROWS or len(np.unique(y[mask])) < 2:
        return float("nan"), n
    try:
        auc = float(roc_auc_score(y[mask], score[mask]))
        return max(auc, 1.0 - auc), n
    except Exception:
        return float("nan"), n


def analyze(feat_df: pd.DataFrame, sweep_df: pd.DataFrame) -> None:
    valid = feat_df.dropna(subset=["saturation_raw", "current_max_chain_raw"])

    logger.info("=== (a) saturation_raw - current_max_chain_raw の分布 (build余地) ===")
    margin = valid["saturation_margin"]
    logger.info(
        "mean=%.3f std=%.3f max=%.1f 正の割合=%.1f%% (n=%d)",
        margin.mean(), margin.std(), margin.max(),
        100.0 * float((margin > 0).mean()), len(margin),
    )
    logger.info("分位点: %s", margin.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict())
    logger.info(
        "saturation_raw 分布: mean=%.2f 分位点=%s",
        valid["saturation_raw"].mean(),
        valid["saturation_raw"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict(),
    )

    logger.info("=== (b) 1盤面あたりコスト実測 ===")
    cost_ms = feat_df["saturation_cost_sec"].dropna() * 1000.0
    logger.info(
        "mean=%.1fms median=%.1fms p95=%.1fms max=%.1fms (n=%d)",
        cost_ms.mean(), cost_ms.median(), cost_ms.quantile(0.95), cost_ms.max(), len(cost_ms),
    )

    logger.info("=== (c) 中盤 win-AUC 比較 ===")
    won_df = feat_df[feat_df["won"].notna()].copy()
    won_df["won"] = won_df["won"].astype(int)
    tcr = won_df["tsumo_count_rate"].astype(float)
    mid_mask = (tcr > TSUMO_MID_LOW) & (tcr <= TSUMO_MID_HIGH)
    mid_df = won_df[mid_mask]
    logger.info("中盤行数: %d / 全体 %d", len(mid_df), len(won_df))

    for label, df_subset in (("全体", won_df), ("中盤", mid_df)):
        y = df_subset["won"].values.astype(float)
        for feat in ("current_max_chain_raw", "saturation_raw", "saturation_margin"):
            score = df_subset[feat].values.astype(float)
            auc, n = _safe_auc(y, score)
            logger.info("[%s] won ~ %s AUC=%.4f (n=%d)", label, feat, auc, n)

    logger.info("=== (d) fill_ratio 感度 (サブサンプル n=%d) ===", len(sweep_df))
    if len(sweep_df):
        for fr in FILL_RATIO_SWEEP:
            col = f"saturation_raw_fr{fr}"
            logger.info(
                "fill_ratio=%.2f: mean=%.2f max=%.0f", fr, sweep_df[col].mean(), sweep_df[col].max(),
            )
        sweep_won = sweep_df[sweep_df["won"].notna()].copy()
        sweep_won["won"] = sweep_won["won"].astype(int)
        tcr_s = sweep_won["tsumo_count_rate"].astype(float)
        mid_mask_s = (tcr_s > TSUMO_MID_LOW) & (tcr_s <= TSUMO_MID_HIGH)
        mid_sweep = sweep_won[mid_mask_s]
        for fr in FILL_RATIO_SWEEP:
            col = f"saturation_raw_fr{fr}"
            y = mid_sweep["won"].values.astype(float)
            score = mid_sweep[col].values.astype(float)
            auc, n = _safe_auc(y, score)
            logger.info("[中盤サブサンプル] won ~ %s AUC=%.4f (n=%d)", col, auc, n)


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")

    logger.info("=== saturation_chain 軽量サブセット検証 開始 ===")
    logger.info("対象動画: %s", TARGET_VIDEO_IDS)

    df = pd.read_csv(LABELED_WIN_CSV)
    df = df[df["video_id"].isin(TARGET_VIDEO_IDS)]
    df = df[df["won"].notna()].reset_index(drop=True)
    logger.info("labeled_win.csv 抽出行数 (won付き): %d", len(df))

    feat_df = build_feature_rows(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(OUT_CSV, index=False)
    logger.info("結果 CSV 保存: %s", OUT_CSV)

    sweep_df = run_fill_ratio_sweep(feat_df)
    OUT_SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(OUT_SWEEP_CSV, index=False)
    logger.info("fill_ratio スイープ結果 CSV 保存: %s", OUT_SWEEP_CSV)

    analyze(feat_df, sweep_df)

    logger.info("=== saturation_chain 軽量サブセット検証 完了 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
