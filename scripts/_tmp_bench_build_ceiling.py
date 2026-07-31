"""build_ceiling_chain (XII-1b 本来の飽和・ビームサーチ近似) の micro-benchmark。

新規指標統合前に、1 STABLE snapshot あたりの計算コストを実データ盤面
(data/indicators_v2/boards/v29.npz) で計測する。depth × beam_width の
組み合わせを振って 200ms 予算 (仕様書指定) との比較を行う。

計測パターン (scripts/_tmp_bench_xii.py と同様の warm/cold 区別):
    - warm: 既存指標群でキャッシュを温めた後の呼び出し時間 (実運用に近い)。
    - cold: 新品 ChainSimulator (キャッシュ空) での呼び出し時間 (最悪ケース)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_bench_build_ceiling
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

# ベンチ対象盤面のぷよ数レンジ (「典型盤面 30-60 個」指定に一致)
PUYO_COUNT_MIN: int = 30
PUYO_COUNT_MAX: int = 60
# 1 盤面あたりの計測反復回数 (ジッタ低減)
N_REPEATS: int = 3
# 対象 npz (小さい 1 動画で十分。実データ分布の代表として v29 を使用)
SAMPLE_NPZ: Path = Path("data/indicators_v2/boards/v29.npz")
# 200ms 予算 (仕様書指定の閾値)
BUDGET_MS: float = 200.0
# 計測する (depth, beam_width) の組み合わせ
DEPTH_BEAM_GRID: "list[tuple[int, int]]" = [
    (2, 8), (3, 8), (2, 16),
]
# サンプル盤面数上限
MAX_SAMPLE_BOARDS: int = 8


def _load_sample_boards() -> list[Board]:
    """v29.npz からぷよ数 30-60 の盤面を抽出し Board リストに変換する。"""
    data = np.load(str(SAMPLE_NPZ), allow_pickle=True)
    grids = data["grids"]
    counts = (grids > 0).sum(axis=(1, 2))
    idxs = np.where((counts >= PUYO_COUNT_MIN) & (counts <= PUYO_COUNT_MAX))[0]
    if len(idxs) > MAX_SAMPLE_BOARDS:
        idxs = idxs[np.linspace(0, len(idxs) - 1, MAX_SAMPLE_BOARDS).astype(int)]
    return [Board.from_list(grids[i].tolist()) for i in idxs]


def _run_existing_pipeline_warmup(board: Board, sim: ChainSimulator) -> None:
    """既存指標群の呼び出し (ChainSimulator キャッシュ温め用)。"""
    iv.current_max_chain(board, sim)
    iv.immediate_fire_power(board, 0.0, sim)
    iv.saturated_chain_count(board, sim)
    iv.potential_fire_power(board, 0.0, sim)


def _bench_one(board: Board, depth: int, beam_width: int, warm: bool) -> float:
    """1 盤面 × 1 (depth, beam) 設定の所要時間 (ms) を返す。"""
    sim = ChainSimulator()
    if warm:
        _run_existing_pipeline_warmup(board, sim)  # 計測対象外
    t0 = time.perf_counter()
    iv.build_ceiling_chain(board, depth=depth, beam_width=beam_width, simulator=sim)
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    boards = _load_sample_boards()
    print(f"[bench] サンプル盤面数: {len(boards)} (ぷよ数 {PUYO_COUNT_MIN}-{PUYO_COUNT_MAX})")

    for depth, beam_width in DEPTH_BEAM_GRID:
        warm_times: list[float] = []
        cold_times: list[float] = []
        for board in boards:
            warm_reps = [_bench_one(board, depth, beam_width, warm=True) for _ in range(N_REPEATS)]
            cold_reps = [_bench_one(board, depth, beam_width, warm=False) for _ in range(N_REPEATS)]
            warm_times.append(float(np.median(warm_reps)))
            cold_times.append(float(np.median(cold_reps)))
        avg_warm = float(np.mean(warm_times))
        avg_cold = float(np.mean(cold_times))
        max_cold = float(np.max(cold_times))
        verdict = "OK" if avg_warm <= BUDGET_MS else "NG (予算超過)"
        print(
            f"[bench] depth={depth} beam={beam_width:2d} "
            f"warm_avg={avg_warm:7.2f}ms cold_avg={avg_cold:7.2f}ms "
            f"cold_max={max_cold:7.2f}ms 判定={verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
