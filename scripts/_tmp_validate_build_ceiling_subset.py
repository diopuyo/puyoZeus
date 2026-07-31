"""build_ceiling_chain (XII-1b 本来の飽和・ビームサーチ近似) の軽量サブセット検証。

全再収集を避け、既存 data/indicators_v2/boards/{v29,v33}.npz (盤面 grid) と
data/indicators_v2/study/labeled_win.csv (won ラベル・tsumo_count_rate 等) を
突合して2動画分のみで感触を見る。

検証内容:
    (a) build_ceiling_chain(depth=2) と current_max_chain (既存) の Pearson r。
    (b) build余地 (build_ceiling - current_max_chain) の分布。
    (c) 中盤 (tsumo_count_rate 0.33-0.67) の単変量 win-AUC 比較
        (build_ceiling vs current_max_chain の上積み有無)。

熱対策: 単プロセス・スレッド制限・2動画限定 (軽量サブセット方針)。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_validate_build_ceiling_subset
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

# ============================
# 定数
# ============================

# 検証対象の軽量サブセット。
# 指示は「v29,v33等2動画で十分」だが、boards/*.npz が labeled_win.csv に対し
# 疎サンプル (v29: 3495行中478行のみ突合成立=約14%) であるため、
# 2動画だと中盤n=127と統計的検出力が弱い。既に data/indicators_v2/boards/ に
# 存在する v29-v38 (10動画、追加収集ゼロ・追加DLなし) に拡張してn を稼ぐ。
TARGET_VIDEO_IDS: tuple[str, ...] = tuple(f"video_{i}" for i in range(29, 39))
BOARDS_DIR: Path = Path("data/indicators_v2/boards")
LABELED_WIN_CSV: Path = Path("data/indicators_v2/study/labeled_win.csv")

# 盤面照合の最大時刻差 (秒)。同一収集由来なので厳しめでよい。
MAX_T_DIFF_SEC: float = 1.0

# 中盤の定義 (tsumo_count_rate ベース、既存 scripts/prescreen_candidates.py 準拠)。
TSUMO_MID_LOW: float = 0.33
TSUMO_MID_HIGH: float = 0.67

# build_ceiling_chain のビームサーチ設定 (既定値をそのまま使用)。
BUILD_CEILING_DEPTH: int = 2
BUILD_CEILING_BEAM_WIDTH: int = 8

# AUC 計算に必要な最小サンプル数。
MIN_VALID_ROWS: int = 10

OUT_CSV: Path = Path("data/indicators_v2/build_ceiling_subset_result.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================
# セクション1: npz ロード + 突合
# ============================


def _load_npz_for_video(video_id: str) -> dict[str, np.ndarray]:
    """video_id (例 'video_29') に対応する v*.npz をロードする。"""
    stem = video_id.replace("video_", "v")
    npz_path = BOARDS_DIR / f"{stem}.npz"
    data = np.load(str(npz_path), allow_pickle=True)
    return {
        "side": data["side"],
        "game_idx": data["game_idx"],
        "t_sec": data["t_sec"].astype(float),
        "grids": data["grids"],
    }


def _match_grid(
    npz_data: dict[str, np.ndarray], side: str, game_idx: int, t_sec: float,
) -> np.ndarray | None:
    """(side, game_idx, t_sec) に最も近い grid を返す (許容誤差外なら None)。"""
    mask = (npz_data["side"] == side) & (npz_data["game_idx"] == game_idx)
    if not mask.any():
        return None
    t_arr = npz_data["t_sec"][mask]
    grids = npz_data["grids"][mask]
    diffs = np.abs(t_arr - t_sec)
    idx = int(np.argmin(diffs))
    if float(diffs[idx]) > MAX_T_DIFF_SEC:
        return None
    return grids[idx]


# ============================
# セクション2: 指標計算
# ============================


def _grid_to_board(grid: np.ndarray) -> "Any":
    """shape (13, 6) int8 ndarray を Board オブジェクトに変換する。"""
    from src.board import Board
    return Board.from_list(grid.tolist())


def build_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """labeled_win.csv 抽出行に build_ceiling_chain を計算して列追加する。"""
    from src.chain import ChainSimulator
    import src.indicators_v2 as iv

    sim = ChainSimulator()
    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    for vid in TARGET_VIDEO_IDS:
        npz_cache[vid] = _load_npz_for_video(vid)

    rows_out: list[dict] = []
    n_matched = 0
    n_missed = 0
    n_mismatch_sanity = 0
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

        # サニティ: 再計算した current_max_chain (=saturated_chain_count) が
        # labeled_win.csv の既存値と一致するか (盤面照合が正しいかの確認)。
        sanity = iv.current_max_chain(board, sim)
        csv_current_max = float(row["current_max_chain_raw"])
        if abs(sanity.raw - csv_current_max) > 1e-6:
            n_mismatch_sanity += 1

        ceiling = iv.build_ceiling_chain(
            board, depth=BUILD_CEILING_DEPTH, beam_width=BUILD_CEILING_BEAM_WIDTH,
            simulator=sim,
        )

        row_dict = row.to_dict()
        row_dict["build_ceiling_raw"] = ceiling.raw
        row_dict["build_ceiling_score"] = ceiling.score
        row_dict["build_margin"] = ceiling.raw - csv_current_max
        row_dict["sanity_current_max_recomputed"] = sanity.raw
        rows_out.append(row_dict)
        n_matched += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            logger.info(
                "進捗: %d/%d matched=%d missed=%d sanity_mismatch=%d elapsed=%.0fs eta=%.0fs",
                i + 1, total, n_matched, n_missed, n_mismatch_sanity, elapsed, eta,
            )

    logger.info(
        "完了: matched=%d missed=%d sanity_mismatch=%d (total=%d)",
        n_matched, n_missed, n_mismatch_sanity, total,
    )
    return pd.DataFrame(rows_out)


# ============================
# セクション3: 分析
# ============================


def _safe_auc(y: np.ndarray, score: np.ndarray) -> tuple[float, int]:
    """NaN を除外して単変量 AUC を返す (計算不能なら nan)。"""
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


def analyze(feat_df: pd.DataFrame) -> None:
    """相関・build余地分布・中盤AUC比較をログ出力する。"""
    valid = feat_df.dropna(subset=["build_ceiling_raw", "current_max_chain_raw"])
    logger.info("=== (a) build_ceiling_raw vs current_max_chain_raw の相関 ===")
    r = float(np.corrcoef(valid["build_ceiling_raw"], valid["current_max_chain_raw"])[0, 1])
    logger.info("Pearson r = %.4f (n=%d)", r, len(valid))

    logger.info("=== (b) build余地 (build_ceiling - current_max_chain) の分布 ===")
    margin = valid["build_margin"]
    logger.info(
        "mean=%.3f std=%.3f max=%.1f 非ゼロ率=%.1f%% (n=%d)",
        margin.mean(), margin.std(), margin.max(),
        100.0 * float((margin > 0).mean()), len(margin),
    )
    logger.info("分位点: %s", margin.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict())

    logger.info("=== (c) 中盤 win-AUC 比較 ===")
    won_df = feat_df[feat_df["won"].notna()].copy()
    won_df["won"] = won_df["won"].astype(int)
    tcr = won_df["tsumo_count_rate"].astype(float)
    mid_mask = (tcr > TSUMO_MID_LOW) & (tcr <= TSUMO_MID_HIGH)
    mid_df = won_df[mid_mask]
    logger.info("中盤行数: %d / 全体 %d", len(mid_df), len(won_df))

    for label, df_subset in (("全体", won_df), ("中盤", mid_df)):
        y = df_subset["won"].values.astype(float)
        for feat in ("current_max_chain_raw", "build_ceiling_raw", "build_margin"):
            score = df_subset[feat].values.astype(float)
            auc, n = _safe_auc(y, score)
            logger.info("[%s] won ~ %s AUC=%.4f (n=%d)", label, feat, auc, n)


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")

    logger.info("=== build_ceiling_chain 軽量サブセット検証 開始 ===")
    logger.info("対象動画: %s", TARGET_VIDEO_IDS)

    df = pd.read_csv(LABELED_WIN_CSV)
    df = df[df["video_id"].isin(TARGET_VIDEO_IDS)]
    # won ラベルが付いている行のみ (AUC計算に必須、かつ計算コスト削減)。
    df = df[df["won"].notna()].reset_index(drop=True)
    logger.info("labeled_win.csv 抽出行数 (won付き): %d", len(df))

    feat_df = build_feature_rows(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(OUT_CSV, index=False)
    logger.info("結果 CSV 保存: %s", OUT_CSV)

    analyze(feat_df)

    logger.info("=== build_ceiling_chain 軽量サブセット検証 完了 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
