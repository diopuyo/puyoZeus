# XII 5指標を「満杯盤面(中盤・後半)」でベンチし、病的遅延の有無を切り分ける
import time
from pathlib import Path

import numpy as np

import src.indicators_v2 as iv
from src.board import Board

# boards_lean_fixed から実データの STABLE 盤面を集める(中盤の満杯盤面が含まれる)
npz_dir = Path("data/indicators_v2/boards_lean_fixed")
files = sorted(npz_dir.glob("*.npz"))[:6]
grids = []
for f in files:
    d = np.load(f, allow_pickle=False)
    g = d["grids"]
    grids.append(g)
allg = np.concatenate(grids, axis=0)
puyos = (allg > 0).sum(axis=(1, 2))
print(f"総盤面: {len(allg)}  ぷよ数分布: min={puyos.min()} med={int(np.median(puyos))} max={puyos.max()}")

# ぷよ数帯別にサンプルして XII 5指標の所要時間を測る
bands = [(0, 20), (20, 40), (40, 55), (55, 100)]
for lo, hi in bands:
    idx = np.where((puyos >= lo) & (puyos < hi))[0]
    if len(idx) == 0:
        continue
    sample = idx[:: max(1, len(idx) // 30)][:30]
    times = []
    for i in sample:
        b = Board.from_list(allg[i].astype(int).tolist())
        t0 = time.perf_counter()
        iv.saturated_chain_count(b)
        iv.ignition_point_count(b)
        iv.multi_color_ignition(b)
        iv.sub_chain_count(b)
        iv.simultaneous_pop_richness(b)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"ぷよ{lo:>2}-{hi:<3}: n={len(sample):2d} "
          f"median={times[len(times)//2]:7.1f}ms  max={times[-1]:8.1f}ms  "
          f"合計={sum(times)/1000:.1f}s")
