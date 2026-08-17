"""scan_judgment_anomalies の1レコードあたり実処理時間を測る (2026-08-11、使い捨て)。

3動画フル実走がCPU競合下で1動画25分超と判明したため、実際に走らせなくても
1レコードあたりコストから全体所要時間を見積もれるよう単独ベンチマークする。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import threadpoolctl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import Board  # noqa: E402
from scripts.scan_judgment_anomalies import make_score_fn  # noqa: E402
from scripts.visualize_advantage_overlay import _train_model  # noqa: E402

NPZ_PATH = Path("data/indicators_v2/boards_lean_phase_l_2026-08-07/29.npz")
N_SAMPLE_PAIRS = 300


def main() -> int:
    with threadpoolctl.threadpool_limits(limits=2):
        t0 = time.time()
        model = _train_model(None)
        print(f"[bench] train: {time.time() - t0:.1f}s", flush=True)
        score_fn = make_score_fn(model)

        d = np.load(NPZ_PATH, allow_pickle=True)
        grids = d["grids"]
        n = len(grids)
        print(f"[bench] {NPZ_PATH.name}: n_frames={n}", flush=True)

        n_boards = min(N_SAMPLE_PAIRS * 2, n)
        boards = [Board.from_list(grids[i].tolist()) for i in range(n_boards)]

        t1 = time.time()
        cnt = 0
        for i in range(0, len(boards) - 1, 2):
            score_fn(boards[i], boards[i + 1])
            cnt += 1
        dt = time.time() - t1
        print(
            f"[bench] scored {cnt} pairs in {dt:.2f}s -> "
            f"{dt / cnt * 1000:.2f} ms/pair",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
