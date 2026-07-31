"""フェーズ2ゲート: c1/c3 npz の next列カバレッジと基本統計を確認する。"""
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
for stem in ("c1", "c3"):
    p = PROJ / "data" / "indicators_v2" / "boards_lean_next" / f"{stem}.npz"
    d = np.load(p, allow_pickle=True)
    n = len(d["grids"])
    next_a = d["next1_a"]
    valid = np.isin(next_a, [1, 2, 3, 4, 5]).sum()
    absent = (next_a == -1).sum()
    misdetect = n - valid - absent
    print(f"{stem}: n={n} valid={valid}({valid/n*100:.1f}%) absent={absent}({absent/n*100:.1f}%) "
          f"misdetect={misdetect}({misdetect/n*100:.1f}%)")
    print(f"  keys={list(d.keys())}")
    print(f"  won distribution: {np.unique(d['won'][~np.isnan(d['won'])], return_counts=True)}")
