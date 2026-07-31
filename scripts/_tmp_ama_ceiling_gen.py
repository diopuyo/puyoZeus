"""得点ベース飽和火力 (ama-loop, scripts/_tmp_ama_builder.py, 単線greedy・4色限定)

を labeled_win.csv 対象10動画のキャッシュ済み npz 盤面に対して計算し、CSV出力する。

コーディネータ指示 (2026-07-22): 得点ベースの天井が勝敗を予測するかの
AUC検証が最優先 (難所割りより価値判定が先)。measurement only、src本体は
変更しない (chain_bitboard.py/chain.py 不変更)。

前回 deep_ceiling AUC 検証 (scripts/_tmp_deep_ceiling_gen.py) と同じ突合手順を
踏襲する (scripts/_tmp_validate_build_ceiling_subset.py の
_load_npz_for_video/_match_grid/_grid_to_board/TARGET_VIDEO_IDS を再利用)。
npz キャッシュの疎さで4動画程度しか突合しない既知問題はそのまま (桁を掴む
段階、コーディネータ指示通り)。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_ama_ceiling_gen
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._tmp_validate_build_ceiling_subset import (  # noqa: E402
    _load_npz_for_video, _match_grid, _grid_to_board, TARGET_VIDEO_IDS, BOARDS_DIR,
)
from scripts._tmp_ama_builder import ama_build, _compute_active_colors_by_game  # noqa: E402

LABELED_WIN_CSV = Path("data/indicators_v2/study/labeled_win.csv")
OUT_CSV = Path("data/indicators_v2/study/ama_ceiling_video_result.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_active_colors_cache() -> "dict[tuple[str, int], tuple[int, ...]]":
    """対象10動画それぞれの (video_id, game_idx) -> active_colors (4色) を計算する。

    scripts/_tmp_ama_builder.py の _compute_active_colors_by_game を再利用する
    (出現頻度上位4色、user伝授のドメイン修正 2026-07-22)。
    """
    cache: "dict[tuple[str, int], tuple[int, ...]]" = {}
    for vid in TARGET_VIDEO_IDS:
        stem = vid.replace("video_", "v")
        npz_path = BOARDS_DIR / f"{stem}.npz"
        cache.update(_compute_active_colors_by_game(npz_path))
    return cache


def main() -> int:
    logger.info("=== ama-loop 得点ベース飽和火力 生成開始 (単線greedy, guard有効) ===")
    df = pd.read_csv(LABELED_WIN_CSV)
    df = df[df["video_id"].isin(TARGET_VIDEO_IDS)]
    df = df[df["won"].notna()].reset_index(drop=True)
    logger.info("対象行数 (won付き): %d", len(df))

    npz_cache: "dict[str, dict[str, np.ndarray]]" = {}
    for vid in TARGET_VIDEO_IDS:
        npz_cache[vid] = _load_npz_for_video(vid)
    active_colors_cache = _load_active_colors_cache()
    logger.info("active_colors 計算完了: %d (video, game_idx) 組", len(active_colors_cache))

    rows_out: "list[dict]" = []
    n_matched = 0
    n_missed = 0
    n_missed_colors = 0
    n_chain_ref_below_csv = 0
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
        colors = active_colors_cache.get((vid, game_idx))
        if colors is None:
            n_missed_colors += 1
            continue
        board = _grid_to_board(grid)
        csv_current_max = float(row["current_max_chain_raw"])

        t0 = time.perf_counter()
        result = ama_build(board, active_colors=colors, guard_enabled=True)
        cost_sec = time.perf_counter() - t0

        # 注記 (正直な注記、バグではない): csv の current_max_chain_raw は
        # 5色無制限スキャンで計算されているのに対し、ama_build は試合別
        # active_colors (4色) に限定している。そのため final_chain_ref が
        # csv 値をわずかに下回るのは想定内 (5色→4色制限による差)。
        # deep_ceiling_gen.py のような「実装バグ疑い」の sanity 違反とは性質が
        # 異なるため、件数のみ記録し「違反」とは呼ばない。
        if result.final_chain_ref < csv_current_max - 1e-6:
            n_chain_ref_below_csv += 1

        row_dict = row.to_dict()
        row_dict["ama_ceiling_score"] = result.final_score
        row_dict["ama_before_score"] = result.before_score
        row_dict["ama_ceiling_chain_ref"] = result.final_chain_ref
        row_dict["ama_before_chain_ref"] = result.before_chain_ref
        row_dict["ama_ceiling_margin_score"] = result.final_score - result.before_score
        row_dict["ama_ceiling_cost_sec"] = cost_sec
        row_dict["ama_ceiling_iterations"] = result.iterations
        rows_out.append(row_dict)
        n_matched += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            logger.info(
                "進捗: %d/%d matched=%d missed=%d missed_colors=%d "
                "chain_ref<csv件数=%d 平均コスト=%.1fms elapsed=%.0fs eta=%.0fs",
                i + 1, total, n_matched, n_missed, n_missed_colors, n_chain_ref_below_csv,
                1000.0 * sum(r["ama_ceiling_cost_sec"] for r in rows_out) / max(1, n_matched),
                elapsed, eta,
            )

    logger.info(
        "完了: matched=%d missed=%d missed_colors=%d (total=%d) chain_ref<csv件数=%d",
        n_matched, n_missed, n_missed_colors, total, n_chain_ref_below_csv,
    )

    out_df = pd.DataFrame(rows_out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    logger.info("結果 CSV 保存: %s", OUT_CSV)

    margin = out_df["ama_ceiling_margin_score"]
    cost_ms = out_df["ama_ceiling_cost_sec"] * 1000.0
    logger.info(
        "得点margin: mean=%.1f 正の割合=%.1f%% | cost: mean=%.1fms p95=%.1fms max=%.1fms",
        margin.mean(), 100.0 * float((margin > 0).mean()),
        cost_ms.mean(), cost_ms.quantile(0.95), cost_ms.max(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
