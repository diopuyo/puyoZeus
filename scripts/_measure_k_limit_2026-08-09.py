"""現実的に到達できる K の上限を実測する (2026-08-09).

user 確認:
> k12 は計算量的に不可能なのは知っています。 大切なのは **どう妥協するか**、
> そして **現実的に可能な K はいくつまでか** を知ることです

## 測ること
1. **K ごとの実測コスト** — 1 手増やすと何倍重くなるか (増加率)
2. **秒間2回 (500ms) で到達できる K の上限**
3. 探索を絞った場合 (ビーム幅を狭める) にどこまで伸びるか

計算量の見積もりだけでなく **実測**で出す。 ビーム幅を変えて、
「精度をどれだけ捨てれば K がいくつ伸びるか」のトレードオフを示す。

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

BUDGET_MS: float = 500.0  # 秒間2回
THRESHOLD = 12.0


def _mk(rows: int = 6, seed: int = 0) -> Board:
    rng = np.random.RandomState(seed)
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - rows, BOARD_ROWS):
        for c in range(BOARD_COLS):
            g[r][c] = int(rng.choice([1, 2, 3, 4]))
    return Board.from_list(g)


def _time_beam(board: Board, k: int, beam: int, next_pair, dnext_pair) -> float:
    """near_future_fire_power を K・ビーム幅指定で回して所要 ms を返す。"""
    t0 = time.perf_counter()
    iv.near_future_fire_power(
        board, next_pair=next_pair, dnext_pair=dnext_pair,
        beam_width=beam, k_levels=(k,),
    )
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    b = _mk(6, seed=3)
    NEXT, DNEXT = (1, 2), (3, 4)

    print("=" * 66)
    print(f"1. K ごとの実測コスト (既定ビーム幅) — 予算 {BUDGET_MS:.0f}ms")
    print("=" * 66)
    print(f"{'K':>3s} {'実測':>10s} {'前K比':>8s} {'予算比':>8s}  判定")
    prev = None
    reachable = 0
    for k in (1, 2, 3, 4, 5):
        ms = _time_beam(b, k, iv.NEAR_FUTURE_BEAM_WIDTH, NEXT, DNEXT)
        ratio = (ms / prev) if prev else float("nan")
        within = ms <= BUDGET_MS
        if within:
            reachable = k
        print(f"{k:3d} {ms:9.1f}ms {ratio:7.2f}x {ms / BUDGET_MS:7.2f}x  "
              f"{'○ 収まる' if within else '✕ 超過'}")
        prev = ms
    print(f"\n  → 既定ビーム幅で 500ms 内に収まる K の上限: **K={reachable}**")
    print(f"  (実手数は ネクスト+ダブルネクスト+K = {reachable + 2} 手ぶん)")

    print()
    print("=" * 66)
    print("2. ビーム幅を絞るとどこまで伸びるか (精度とのトレードオフ)")
    print("=" * 66)
    print(f"{'ビーム幅':>8s} " + " ".join(f"{'K=' + str(k):>9s}" for k in (3, 5, 7, 9)))
    for beam in (iv.NEAR_FUTURE_BEAM_WIDTH, 8, 4, 2, 1):
        cells = []
        for k in (3, 5, 7, 9):
            try:
                ms = _time_beam(b, k, beam, NEXT, DNEXT)
                mark = "○" if ms <= BUDGET_MS else "✕"
                cells.append(f"{ms:7.0f}{mark}")
            except Exception:
                cells.append("     err")
        print(f"{beam:8d} " + " ".join(f"{c:>9s}" for c in cells))
    print("\n  ○ = 500ms 内 / ✕ = 超過")

    print()
    print("=" * 66)
    print("3. ビーム幅を絞ると火力の見積もりがどれだけ変わるか")
    print("=" * 66)
    print("  (K=5 固定。 既定ビーム幅の結果を基準にした差)")
    base = None
    for beam in (iv.NEAR_FUTURE_BEAM_WIDTH, 8, 4, 2, 1):
        r = iv.near_future_fire_power(
            b, next_pair=NEXT, dnext_pair=DNEXT, beam_width=beam, k_levels=(5,),
        )
        v = float(r.scores.get(5, 0.0)) if hasattr(r, "scores") else float("nan")
        if base is None:
            base = v
        diff = v - base
        print(f"  ビーム幅 {beam:3d}: 火力 {v:.4f}  基準との差 {diff:+.4f}")
    print()
    print("  → 差が小さければ、 ビームを絞って K を伸ばす妥協が有効。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
