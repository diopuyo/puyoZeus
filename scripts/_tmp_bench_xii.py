"""XII board sim 5指標 (saturated_chain_count 等) の micro-benchmark。

collect_indicators_v2.py への統合前に、1 STABLE snapshot あたりの
追加コストを実データ盤面 (data/indicators_v2/boards/*.npz) で計測する。

計測パターン:
    - warm: 既存 _fill_indicator_columns 相当の呼び出し (ChainSimulator の
      simulate キャッシュを温める) を先に実行した後の XII 5指標のみの時間。
      実運用の collect パイプラインに最も近い条件。
    - cold: XII 5指標だけを新品 ChainSimulator (キャッシュ空) で呼んだ時間。
      他指標のキャッシュ共有を仮定しない最悪ケース上限。

使い方:
    python -m scripts._tmp_bench_xii
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

# ベンチ対象盤面のぷよ数レンジ (「典型盤面 30-50 個」指定に一致)
PUYO_COUNT_MIN: int = 30
PUYO_COUNT_MAX: int = 50
# 1 盤面あたりの計測反復回数 (ジッタ低減)
N_REPEATS: int = 5
# 対象 npz (小さい 1 動画で十分。実データ分布の代表として v29 を使用)
SAMPLE_NPZ: Path = Path("data/indicators_v2/boards/v29.npz")
# 200ms 予算 (仕様書指定の閾値)
BUDGET_MS: float = 200.0


def _load_sample_boards() -> list[Board]:
    """v29.npz からぷよ数 30-50 の盤面を抽出し Board リストに変換する。"""
    data = np.load(str(SAMPLE_NPZ), allow_pickle=True)
    grids = data["grids"]
    counts = (grids > 0).sum(axis=(1, 2))
    idxs = np.where((counts >= PUYO_COUNT_MIN) & (counts <= PUYO_COUNT_MAX))[0]
    # 分布を均等にサンプルするため等間隔に間引く (最大 10 枚)
    if len(idxs) > 10:
        idxs = idxs[np.linspace(0, len(idxs) - 1, 10).astype(int)]
    boards = [Board.from_list(grids[i].tolist()) for i in idxs]
    return boards


def _run_existing_pipeline(board: Board, sim: ChainSimulator) -> None:
    """既存 _fill_indicator_columns 相当の呼び出し (XII を除く全指標)。

    ChainSimulator のキャッシュ温め用。実際の collect_indicators_v2.py の
    呼び出し順序を模倣する (省略可能なメタ専用指標は除く)。
    """
    total_conn, _ = iv.connectivity_observation(board, sim)
    iv.current_max_chain(board, sim)
    iv.immediate_fire_power(board, 0.0, sim)
    iv.chain_efficiency(board, 0.0, sim)
    iv.min_puyos_to_ignite(board, sim)
    iv.second_chain_potential(board, sim)
    iv.dig_resistance(board, sim)
    iv.absorption_capacity(board)
    iv.ojama_disruption(board, simulator=sim)
    iv.main_linked_pair_count(board, sim)
    iv.isolated_pair_count(board, sim)
    iv.main_linked_ratio(board, sim)
    iv.ukeyasusa(board, sim)


def _run_xii(board: Board, sim: ChainSimulator) -> None:
    """XII 5指標をまとめて呼ぶ (collect 側で追加する呼び出し)。"""
    iv.saturated_chain_count(board, sim)
    iv.ignition_point_count(board, sim)
    iv.multi_color_ignition(board, sim)
    iv.sub_chain_count(board, sim)
    iv.simultaneous_pop_richness(board, sim)


def _bench_warm(board: Board) -> float:
    """既存指標でキャッシュを温めた後の XII 5指標の所要時間 (ms)。"""
    sim = ChainSimulator()
    _run_existing_pipeline(board, sim)  # キャッシュ温め (計測対象外)
    t0 = time.perf_counter()
    _run_xii(board, sim)
    return (time.perf_counter() - t0) * 1000.0


def _bench_cold(board: Board) -> float:
    """新品 ChainSimulator (キャッシュ空) での XII 5指標の所要時間 (ms)。"""
    sim = ChainSimulator()
    t0 = time.perf_counter()
    _run_xii(board, sim)
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    boards = _load_sample_boards()
    print(f"[bench] サンプル盤面数: {len(boards)} (ぷよ数 {PUYO_COUNT_MIN}-{PUYO_COUNT_MAX})")
    warm_times: list[float] = []
    cold_times: list[float] = []
    for i, board in enumerate(boards):
        n_puyo = int((board._grid > 0).sum())
        # 反復平均 (最初の 1 回は JIT/import warmup の影響があるため後半を使う)
        warm_reps = [_bench_warm(board) for _ in range(N_REPEATS)]
        cold_reps = [_bench_cold(board) for _ in range(N_REPEATS)]
        warm_ms = float(np.median(warm_reps))
        cold_ms = float(np.median(cold_reps))
        warm_times.append(warm_ms)
        cold_times.append(cold_ms)
        print(f"[bench] board#{i} puyo={n_puyo:2d} warm={warm_ms:7.2f}ms cold={cold_ms:7.2f}ms")

    avg_warm = float(np.mean(warm_times))
    avg_cold = float(np.mean(cold_times))
    print("=" * 60)
    print(f"[bench] 平均 warm (既存パイプライン後): {avg_warm:.2f} ms")
    print(f"[bench] 平均 cold (キャッシュ空・最悪ケース): {avg_cold:.2f} ms")
    print(f"[bench] 予算: {BUDGET_MS:.0f} ms")
    verdict = "OK (シンプル実装のまま可)" if avg_warm <= BUDGET_MS else "NG (共有キャッシュ実装が必要)"
    print(f"[bench] 判定 (warm 基準): {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
