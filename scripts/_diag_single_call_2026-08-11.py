"""_score_advantage() 1回あたりコストを厳密に測る (2026-08-11、使い捨て)。

学習済モデルが expected_fire_k1/k2 (モンテカルロ)・near_future_fire_k1-5・
fire_stability_k2/4/6 (ビームサーチ) を特徴量に含むため (CSV実測)、
1回の呼び出しが本質的に重いという仮説を検証する。
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
N_CALLS = 10


def main() -> int:
    with threadpoolctl.threadpool_limits(limits=2):
        t0 = time.time()
        model = _train_model(None)
        print(f"[bench] train: {time.time() - t0:.1f}s "
              f"feature_cols={len(model._puyo_feature_cols)}", flush=True)
        score_fn = make_score_fn(model)

        d = np.load(NPZ_PATH, allow_pickle=True)
        grids = d["grids"]
        b1 = Board.from_list(grids[100].tolist())
        b2 = Board.from_list(grids[101].tolist())

        for i in range(N_CALLS):
            t1 = time.time()
            score_fn(b1, b2)
            print(f"[bench] call {i + 1}: {(time.time() - t1) * 1000:.1f}ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
