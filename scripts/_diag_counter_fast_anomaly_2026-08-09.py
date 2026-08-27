"""fast 版の速度異常を切り分ける (2026-08-09).

user 指摘:
  1. 厳密計算と簡易計算の差が「連結ボーナスだけ」なのに、 ここまで速度差が
     出るのは不具合を疑う
  2. **K=4 の方が K=1,2 より速いのはなぜか** (大きい K の方が仕事量は多いはず)

## 切り分ける仮説
A. k_levels でフィルタしきれず、 要求していない K も計算している
   (K=1 を頼むと exact 枝が K=1,2 を両方計算する / K=3 を頼むと MC 枝が
    K=3,4 を両方計算する、 という粒度の粗さ)
   → もしそうなら「K=1 は exact 枝 (全列挙 16+256 通り) を丸ごと回すので遅く、
      K=4 は MC 枝 (サンプル数固定) だけなので速い」という**逆転**が説明できる
B. 候補盤面の生成が純 Python ループで、 そこが支配的
C. 厳密版との差は連結ボーナスだけでなく、 探索の広さ (ビーム幅・列挙数) が
   そもそも違う

読み取り専用。 各枝の実際の評価件数と所要時間を出す。
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, Board  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

import src.indicators_v2 as iv  # noqa: E402

THRESHOLD = 12.0


def _make_board(fill_rows: int = 6, seed: int = 0) -> Board:
    rng = np.random.RandomState(seed)
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - fill_rows, BOARD_ROWS):
        for c in range(BOARD_COLS):
            g[r][c] = int(rng.choice([1, 2, 3, 4]))
    return Board.from_list(g)


def main() -> int:
    b = _make_board()

    print("=== 1. K ごとの所要時間と評価件数 ===")
    print("(n_evaluated は実装が『実際に評価した件数』を返す値)")
    print(f"{'要求K':>6s} {'時間':>9s}  評価件数")
    print("-" * 44)
    for k in (1, 2, 3, 4):
        t0 = time.perf_counter()
        r = iv.counter_reach_probability_fast(b, THRESHOLD, k_levels=(k,))
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"{k:6d} {ms:8.1f}ms  {dict(r.n_evaluated)}")

    print()
    print("=== 2. 複数 K を同時要求した場合 ===")
    for ks in ((1, 2), (3, 4), (1, 2, 3, 4)):
        t0 = time.perf_counter()
        r = iv.counter_reach_probability_fast(b, THRESHOLD, k_levels=ks)
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"  K={ks}: {ms:8.1f}ms  {dict(r.n_evaluated)}")
    print()
    print("  → 単独要求と同時要求で時間が変わらなければ、")
    print("    『要求していない K も計算している』ことの証拠になる。")

    print()
    print("=== 3. 厳密版との比較 (同じ K) ===")
    for k in (2, 4):
        t0 = time.perf_counter()
        iv.counter_reach_probability(b, THRESHOLD, k_levels=(k,))
        ms_exact = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        iv.counter_reach_probability_fast(b, THRESHOLD, k_levels=(k,))
        ms_fast = (time.perf_counter() - t0) * 1000.0
        ratio = ms_exact / ms_fast if ms_fast > 0 else float("nan")
        print(f"  K={k}: 厳密 {ms_exact:8.1f}ms / 近似 {ms_fast:8.1f}ms "
              f"→ 近似は {ratio:.2f}倍速")

    print()
    print("=== 4. K=1 のプロファイル 上位10 ===")
    pr = cProfile.Profile()
    pr.enable()
    iv.counter_reach_probability_fast(b, THRESHOLD, k_levels=(1,))
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(10)
    for line in s.getvalue().splitlines()[4:18]:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
