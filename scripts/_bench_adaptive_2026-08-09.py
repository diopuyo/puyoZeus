"""実ツモを使う応手判定の速度と、色を全列挙する従来版との比較 (2026-08-09)."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
from src.board import BOARD_COLS, BOARD_ROWS, Board
from src.console_init import init_console
init_console()
from src.counter_reach_adaptive import estimate_with_budget
import src.indicators_v2 as iv

def mk(rows=6, seed=0):
    rng=np.random.RandomState(seed)
    g=[[0]*BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS-rows, BOARD_ROWS):
        for c in range(BOARD_COLS): g[r][c]=int(rng.choice([1,2,3,4]))
    return Board.from_list(g)

b=mk()
NEXT=((1,2),(3,4))
print("=== K=1,2: 色を全列挙 vs 実ツモを使用 ===")
for k in (1,2):
    t0=time.perf_counter()
    iv.counter_reach_probability_fast(b, 12.0, k_levels=(k,))
    ms_all=(time.perf_counter()-t0)*1000
    t0=time.perf_counter()
    r=estimate_with_budget(b, 12.0, budget_sec=5.0, k_hard_max=k, known_pairs=NEXT)
    ms_known=(time.perf_counter()-t0)*1000
    print(f"  K={k}: 全列挙 {ms_all:7.1f}ms / 実ツモ {ms_known:7.1f}ms "
          f"→ {ms_all/max(ms_known,0.001):5.1f}倍速  (評価件数 {r.n_samples})")

print()
print("=== 時間予算ごとの到達K (実ツモあり) ===")
for budget in (0.05, 0.1, 0.3, 0.5, 1.0):
    r=estimate_with_budget(b, 12.0, budget_sec=budget, known_pairs=NEXT)
    tag = "厳密" if r.exact else f"±{r.half_width:.3f}"
    lim = "予算切れ" if r.truncated_by_budget else ("精度不足" if r.truncated_by_precision else "完走")
    print(f"  予算 {budget*1000:6.0f}ms → K={r.achieved_k} 確率={r.probability:.3f} "
          f"{tag} 実測{r.elapsed_sec*1000:6.1f}ms  {lim}")
