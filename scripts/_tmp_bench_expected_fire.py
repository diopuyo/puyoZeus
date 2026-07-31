"""expected_fire_power (XVI 平均ツモ期待火力) のコストベンチマーク。

user設計変更 (2026-07-22): K=1,2 は全ツモ色パターンの厳密列挙 (乱数不要)、
K=3,4 はモンテカルロ近似に変更した。本スクリプトは
    (a) 実盤面 (異なる盤面、ChainSimulator 共有キャッシュ=収集パイプラインと
        同じ使用条件) でのコスト実測
    (b) K=3,4 モンテカルロの mc_beam_width / mc_n_samples 別コストと
        再現性 (独立re-seed間のばらつき)
を記録し、src/indicators_v2.py の EXPECTED_FIRE_MC_BEAM_WIDTH=2/
EXPECTED_FIRE_MC_N_SAMPLES=24 の根拠を残す (正直な記録)。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_bench_expected_fire
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from scripts._tmp_ama_builder import (  # noqa: E402
    _compute_active_colors_by_game, _load_candidate_boards,
)

BOARDS_NPZ = Path("data/indicators_v2/boards/v29.npz")
N_BOARDS = 10


def _load_sample_boards() -> "list[tuple]":
    boards = _load_candidate_boards([BOARDS_NPZ], sample_per_file=30)
    active = _compute_active_colors_by_game(BOARDS_NPZ)
    out = []
    for board, game_key in boards[:N_BOARDS]:
        colors = active.get(game_key, (1, 2, 3, 4))
        out.append((board, colors))
    return out


def bench_real_boards(mc_beam_width: int, mc_n_samples: int) -> None:
    """異なる実盤面 N_BOARDS 件・共有キャッシュでの実測 (収集パイプライン相当)。"""
    samples = _load_sample_boards()
    times = []
    for board, colors in samples:
        t0 = time.perf_counter()
        iv.expected_fire_power(
            board, active_colors=colors, mc_beam_width=mc_beam_width, mc_n_samples=mc_n_samples,
        )
        times.append(time.perf_counter() - t0)
    print(
        f"  mc_beam_width={mc_beam_width} mc_n_samples={mc_n_samples}: "
        f"mean={statistics.mean(times)*1000:.0f}ms max={max(times)*1000:.0f}ms "
        f"(異なる{len(samples)}盤面、キャッシュ共有あり=収集パイプライン相当)",
    )


def bench_mc_convergence(mc_beam_width: int, mc_n_samples: int, n_reseeds: int = 4) -> None:
    """1盤面固定・独立re-seedを繰り返し、K=3,4 平均値の再現性を見る。"""
    samples = _load_sample_boards()
    board, colors = samples[0]
    means_k3, means_k4 = [], []
    t0 = time.perf_counter()
    for seed in range(n_reseeds):
        result = iv.expected_fire_power(
            board, active_colors=colors, k_levels=(3, 4),
            mc_beam_width=mc_beam_width, mc_n_samples=mc_n_samples, rng_seed=seed * 100 + 1,
        )
        means_k3.append(result.values[3].raw)
        means_k4.append(result.values[4].raw)
    cost_ms = (time.perf_counter() - t0) / n_reseeds * 1000.0
    print(
        f"  mc_beam_width={mc_beam_width} mc_n_samples={mc_n_samples}: cost={cost_ms:.0f}ms  "
        f"K3 mean={statistics.mean(means_k3):.1f} sd={statistics.pstdev(means_k3):.1f}  "
        f"K4 mean={statistics.mean(means_k4):.1f} sd={statistics.pstdev(means_k4):.1f}",
    )


def main() -> None:
    print("=== (a) 実盤面10件でのコスト実測 (共有キャッシュ、収集パイプライン相当) ===")
    for bw, n in ((8, 48), (2, 24), (2, 16), (1, 16)):
        bench_real_boards(bw, n)

    print("\n=== (b) K=3,4 モンテカルロの独立re-seed間ばらつき (1盤面固定) ===")
    for bw, n in ((1, 24), (1, 48), (2, 24), (2, 48), (4, 24), (4, 48), (8, 24), (8, 48)):
        bench_mc_convergence(bw, n)


if __name__ == "__main__":
    main()
