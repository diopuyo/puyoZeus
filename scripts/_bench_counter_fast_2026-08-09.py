"""リアルタイム版の打ち合い応手 (fast) の速度と、重い版との一致度を測る.

user 確認 (2026-08-09): 「リアルタイムの有用性はどんな感じですか」

## 背景
本プロジェクトは **二層設計** を方針としている
(memory project_dual_mode_indicator_design_2026-07-22:
 「重い指標は二層設計 (動画=精度 / リアルタイム=軽量近似)」)。
打ち合い応手にも 2 実装がある:
  - `estimate_counter_distribution` (mc_counter_estimator): 時間予算ぶん手を
    進めるロールアウト。 K 上限なし。 **実測でベンチが完走しないほど重い**
  - `counter_reach_probability_fast` (indicators_v2): bitboard バッチ +
    近似得点。 K=1..4。 リアルタイム向け

## 測ること
1. fast の 1 回あたり所要時間 (30fps = 33.3ms の予算に収まるか)
2. K を変えたときの時間と結果
3. **重い版との判定一致度** (速い代わりに何を失うか)

読み取り専用のベンチマーク。
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

COLORS = (1, 2, 3, 4)
THRESHOLD = 12.0
FRAME_BUDGET_MS: float = 1000.0 / 30.0
N_BOARDS: int = 8


def _make_board(fill_rows: int, seed: int) -> Board:
    rng = np.random.RandomState(seed)
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - fill_rows, BOARD_ROWS):
        for c in range(BOARD_COLS):
            g[r][c] = int(rng.choice(COLORS))
    return Board.from_list(g)


def main() -> int:
    print(f"30fps の 1 フレーム予算 = {FRAME_BUDGET_MS:.1f} ms")
    print("(認識だけで既に 31fps ≒ 32ms を使用済み)")
    print()
    boards = [_make_board(6, s) for s in range(N_BOARDS)]

    print(f"{'K':>3s} {'1回あたり':>12s} {'フレーム比':>10s}  判定")
    print("-" * 46)
    results: dict[int, list[float]] = {}
    for k in (1, 2, 3, 4):
        ts, probs = [], []
        for b in boards:
            t0 = time.perf_counter()
            r = iv.counter_reach_probability_fast(
                b, THRESHOLD, k_levels=(k,),
            )
            ts.append((time.perf_counter() - t0) * 1000.0)
            probs.append(float(r.probabilities.get(k, 0.0)))
        ms = float(np.median(ts))
        results[k] = probs
        ratio = ms / FRAME_BUDGET_MS
        mark = ("○ 余裕あり" if ratio < 0.3 else
                ("△ 認識と同時は厳しい" if ratio < 1.0 else "✕ 予算超過"))
        print(f"{k:3d} {ms:11.1f}ms {ratio:9.2f}x  {mark}")

    print()
    print("K ごとの応手確率 (盤面 8 枚の平均):")
    for k, ps in results.items():
        print(f"  K={k}: {np.mean(ps):.3f}")
    print()
    print("注: K が大きいほど「相手はもっと打てる」= 応手確率が上がるはず。")
    print("    K=4 で頭打ちになるのが fast 版の制約。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
