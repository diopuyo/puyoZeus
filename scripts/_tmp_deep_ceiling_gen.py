"""深化天井 (build_ceiling_chain bitboardバッチ深化版、video相当 depth=20/beam=20) を

labeled_win.csv 対象10動画のキャッシュ済みnpz盤面に対して計算し、CSV出力する。
collect_indicators_v2 の ~1fps パイプラインには入れない別バッチ処理
(コーディネータ指示、chain_bitboard高速化により現実的時間で完了する)。

`scripts/_tmp_validate_build_ceiling_subset.py` と同じ突合ロジックを再利用する。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_deep_ceiling_gen
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
    _load_npz_for_video, _match_grid, _grid_to_board, TARGET_VIDEO_IDS,
)
from scripts._tmp_build_ceiling_bitboard import build_ceiling_chain_deep  # noqa: E402

LABELED_WIN_CSV = Path("data/indicators_v2/study/labeled_win.csv")
OUT_CSV = Path("data/indicators_v2/study/deep_ceiling_video_result.csv")

# video相当設定 (前回報告済み: mean=6.22, time_mean=181ms, time_max=283ms, n=23サンプルで確認済み)
DEEP_DEPTH: int = 20
DEEP_BEAM: int = 20

# 全15229行 (won付き) を計算すると170ms/行として約43分かかるため、
# セッション時間内に収まるようランダムサブサンプルする
# (先行検証 _tmp_validate_saturation_subset.py も同様にサブセットで実施した前例に準拠)。
SAMPLE_SIZE: int = 4000
SAMPLE_SEED: int = 20260722

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("=== 深化天井 生成開始 (depth=%d beam=%d) ===", DEEP_DEPTH, DEEP_BEAM)
    df = pd.read_csv(LABELED_WIN_CSV)
    df = df[df["video_id"].isin(TARGET_VIDEO_IDS)]
    df = df[df["won"].notna()].reset_index(drop=True)
    logger.info("対象行数 (won付き): %d", len(df))
    # 実測: 突合成功率は約12% (先行知見の~14%と整合)。事前サブサンプルすると
    # 有効行が少なすぎる (4000サブサンプル→matched=478) と判明したため、
    # 全15229行に対して実行する (突合失敗行は計算コストがほぼゼロなので
    # 全体の所要時間は「突合成功行数×約0.19秒」で決まり、実質 ~6-8分で収まる)。

    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    for vid in TARGET_VIDEO_IDS:
        npz_cache[vid] = _load_npz_for_video(vid)

    rows_out: list[dict] = []
    n_matched = 0
    n_missed = 0
    n_sanity_mismatch = 0
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

        deep_raw, cost_sec = build_ceiling_chain_deep(
            board, depth=DEEP_DEPTH, beam_width=DEEP_BEAM,
        )
        if deep_raw < csv_current_max - 1e-6:
            n_sanity_mismatch += 1  # running max 保証違反 (実装バグ疑い、正直に記録)

        row_dict = row.to_dict()
        row_dict["deep_ceiling_raw"] = deep_raw
        row_dict["deep_ceiling_margin"] = deep_raw - csv_current_max
        row_dict["deep_ceiling_cost_sec"] = cost_sec
        rows_out.append(row_dict)
        n_matched += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            logger.info(
                "進捗: %d/%d matched=%d missed=%d sanity違反=%d "
                "平均コスト=%.1fms elapsed=%.0fs eta=%.0fs",
                i + 1, total, n_matched, n_missed, n_sanity_mismatch,
                1000.0 * sum(r["deep_ceiling_cost_sec"] for r in rows_out) / max(1, n_matched),
                elapsed, eta,
            )

    logger.info(
        "完了: matched=%d missed=%d (total=%d) sanity違反=%d",
        n_matched, n_missed, total, n_sanity_mismatch,
    )

    out_df = pd.DataFrame(rows_out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    logger.info("結果 CSV 保存: %s", OUT_CSV)

    margin = out_df["deep_ceiling_margin"]
    cost_ms = out_df["deep_ceiling_cost_sec"] * 1000.0
    logger.info(
        "margin: mean=%.3f 正の割合=%.1f%% | cost: mean=%.1fms p95=%.1fms max=%.1fms",
        margin.mean(), 100.0 * float((margin > 0).mean()),
        cost_ms.mean(), cost_ms.quantile(0.95), cost_ms.max(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
