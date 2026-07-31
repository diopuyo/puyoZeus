import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path('.').resolve()))
from src.board import Board
from src.chain import ChainSimulator
import src.indicators_v2 as iv

data = np.load('data/indicators_v2/boards/v29.npz', allow_pickle=True)
grids = data['grids']
rng = np.random.default_rng(2)
idxs = rng.choice(len(grids), size=15, replace=False)
boards = [Board.from_list(grids[i].tolist()) for i in idxs if not Board.from_list(grids[i].tolist()).is_dead()]
sim = ChainSimulator()
for bw in (6, 12, 20, 30):
    t0=time.perf_counter()
    raws=[]
    for b in boards:
        v = iv.saturation_chain(b, beam_width=bw, simulator=sim)
        raws.append(v.raw)
    dt=time.perf_counter()-t0
    print(f'beam_width={bw}: mean={np.mean(raws):.2f} max={np.max(raws):.0f} time_per_board={dt/len(boards)*1000:.1f}ms')
