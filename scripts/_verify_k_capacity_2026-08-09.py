"""3点を実測する (2026-08-09、 user確認事項).

1. **全置き方を探索できているか** — 列挙漏れが無いか、 理論値と一致するか
2. **秒間2回 (500ms) の性能を満たすか**
3. **統計的に有意なサンプル数を確保したうえで、 K はいくつまで増やせるか**

「有意な数」の定義: 二項比率の Wilson 信頼区間 (95%) の半幅が目標以内。
目標を変えて必要サンプル数も変わるため、 複数の目標で測る。

読み取り専用。
"""
from __future__ import annotations

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
from src.counter_reach_adaptive import (  # noqa: E402
    estimate_with_budget,
    required_samples,
    wilson_half_width,
)

THRESHOLD = 12.0
BUDGET_SEC = 0.5  # 秒間2回


def _mk(rows: int = 6, seed: int = 0) -> Board:
    rng = np.random.RandomState(seed)
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - rows, BOARD_ROWS):
        for c in range(BOARD_COLS):
            g[r][c] = int(rng.choice([1, 2, 3, 4]))
    return Board.from_list(g)


def _brute_force_placements(board: Board, pair: "tuple[int, int]") -> int:
    """物理的に可能な配置を素朴に数える (実装の検証用の独立実装)。

    軸ぷよの列 (6) × 回転 (4)。 横向き (回転 1,3) は 2 列使うので
    右端の列には置けない。 置いた結果が窒息なら除外。
    """
    n = 0
    for rot in range(4):
        max_col = BOARD_COLS if rot in (0, 2) else BOARD_COLS - 1
        for col in range(max_col):
            placed = iv._place_pair_to_board(board, pair, col, rot)
            if placed is not None and not placed.is_dead():
                n += 1
    return n


def main() -> int:
    print("=" * 62)
    print("1. 全置き方を探索できているか")
    print("=" * 62)
    print(f"{'積み段数':>8s} {'実装':>6s} {'独立実装':>8s} {'一致':>6s} {'異なる盤面':>10s}")
    ok_all = True
    for rows in (0, 3, 6, 9, 11, 12):
        b = _mk(rows, seed=rows) if rows else Board.from_list(
            [[0] * BOARD_COLS for _ in range(BOARD_ROWS)])
        impl = iv._enumerate_placement_boards(b, (1, 2))
        brute = _brute_force_placements(b, (1, 2))
        uniq = len({x._grid.tobytes() for x in impl})
        same = len(impl) == brute
        ok_all = ok_all and same
        print(f"{rows:8d} {len(impl):6d} {brute:8d} {'○' if same else '✕':>6s} {uniq:10d}")
    print(f"\n  → 列挙漏れ: {'なし' if ok_all else '**あり**'}")
    print("  (理論値: 回転4方向 × 列。 縦6列+横5列+縦6列+横5列 = 22)")

    print()
    print("=" * 62)
    print(f"2-3. 秒間2回 ({BUDGET_SEC * 1000:.0f}ms) で K はいくつまで伸ばせるか")
    print("=" * 62)
    board = _mk(6, seed=7)
    NEXT = ((1, 2), (3, 4))
    print(f"{'目標精度':>8s} {'必要n':>7s} {'到達K':>6s} {'確率':>7s} {'実半幅':>8s} {'実測':>9s} {'打切り理由':>10s}")
    for target in (0.10, 0.05, 0.03, 0.01):
        need = required_samples(target)
        t0 = time.perf_counter()
        r = estimate_with_budget(
            board, THRESHOLD, budget_sec=BUDGET_SEC,
            target_half_width=target, known_pairs=NEXT,
        )
        ms = (time.perf_counter() - t0) * 1000
        why = ("予算切れ" if r.truncated_by_budget else
               ("精度不足" if r.truncated_by_precision else "完走"))
        hw = "厳密" if r.exact else f"±{r.half_width:.3f}"
        print(f"{target:8.2f} {need:7d} {r.achieved_k:6d} {r.probability:7.3f} "
              f"{hw:>8s} {ms:8.1f}ms {why:>10s}")

    print()
    print("=" * 62)
    print("4. 予算を増やすと K はどこまで伸びるか (上限の把握)")
    print("=" * 62)
    print(f"{'予算':>8s} {'到達K':>6s} {'確率':>7s} {'半幅':>8s} {'サンプル':>9s} {'実測':>9s}")
    for budget in (0.5, 1.0, 2.0, 5.0, 10.0):
        t0 = time.perf_counter()
        r = estimate_with_budget(
            board, THRESHOLD, budget_sec=budget, known_pairs=NEXT,
        )
        ms = (time.perf_counter() - t0) * 1000
        hw = "厳密" if r.exact else f"±{r.half_width:.3f}"
        print(f"{budget * 1000:7.0f}ms {r.achieved_k:6d} {r.probability:7.3f} "
              f"{hw:>8s} {r.n_samples:9d} {ms:8.1f}ms")

    print()
    print("=" * 62)
    print("5. 盤面の埋まり具合による差 (予算 500ms)")
    print("=" * 62)
    for rows in (3, 6, 9, 11):
        b = _mk(rows, seed=rows)
        r = estimate_with_budget(
            b, THRESHOLD, budget_sec=BUDGET_SEC, known_pairs=NEXT,
        )
        hw = "厳密" if r.exact else f"±{r.half_width:.3f}"
        print(f"  {rows:2d}段積み: K={r.achieved_k} 確率={r.probability:.3f} "
              f"{hw} 実測{r.elapsed_sec * 1000:6.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
