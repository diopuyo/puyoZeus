"""効果測定(c) npz とアンカー npz の bit 一致検証 (2026-08-04、使い捨て)。

中間集計で (c) の誤りセルが OFF と完全一致 (51→51、変化ゼロ) だったため、
「4条件ゲートが一度も作動していないのでは」を切り分ける。
grids が全行 bit 一致なら (c) 走行はアンカーと同一出力 = ゲート実質不作動。
"""
from pathlib import Path

import numpy as np

ANCHOR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
C_DIR = Path("data/verify/effect_gate_2026-08-04_c/on_full")


def main() -> None:
    for p in sorted(C_DIR.glob("*.npz")):
        stem = p.stem
        a = np.load(ANCHOR / f"{stem}.npz", allow_pickle=True)
        c = np.load(p, allow_pickle=True)
        ga, gc = a["grids"], c["grids"]
        n = min(len(ga), len(gc))
        same = sum(1 for i in range(n) if np.array_equal(ga[i], gc[i]))
        print(f"{stem}: anchor={len(ga)}行 (c)={len(gc)}行 "
              f"共通先頭{n}行中一致={same} ({same / n:.2%})")


if __name__ == "__main__":
    main()
