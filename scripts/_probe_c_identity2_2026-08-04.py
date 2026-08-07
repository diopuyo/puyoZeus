"""効果測定(c) npz vs アンカー npz: frame_idx 突合での差分率検証 (2026-08-04、使い捨て)。

位置比較はスナップショット集合の増減でズレるため、同一 (side, game_idx,
frame_idx) 同士で grids を比較する。ゲートが実際に確定盤面を変えた頻度と、
スナップショット集合の増減 (c のみ / anchor のみ) を分離して報告する。
"""
from pathlib import Path

import numpy as np

ANCHOR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
C_DIR = Path("data/verify/effect_gate_2026-08-04_c/on_full")


def _key_arrays(z: "np.lib.npyio.NpzFile") -> "np.ndarray":
    return z["side"], z["game_idx"], z["frame_idx"], z["grids"]


def main() -> None:
    tot_common = tot_diff = 0
    for p in sorted(C_DIR.glob("*.npz")):
        stem = p.stem
        a = np.load(ANCHOR / f"{stem}.npz", allow_pickle=True)
        c = np.load(p, allow_pickle=True)
        sa, ga_i, fa, ga = _key_arrays(a)
        sc, gc_i, fc, gc = _key_arrays(c)
        amap = {}
        for i in range(len(fa)):
            amap[(str(sa[i]), int(ga_i[i]), int(fa[i]))] = i
        # (c) は max-sec 打ち切りのため、anchor 側も (c) の最大 frame_idx までに限定
        fc_max_by_side = {}
        common = diff = c_only = 0
        for j in range(len(fc)):
            key = (str(sc[j]), int(gc_i[j]), int(fc[j]))
            i = amap.get(key)
            if i is None:
                c_only += 1
                continue
            common += 1
            if not np.array_equal(ga[i], gc[j]):
                diff += 1
        print(f"{stem}: (c)行={len(fc)} 共通frame={common} 差分あり={diff} "
              f"({(diff / common if common else 0):.2%}) (c)のみ={c_only}")
        tot_common += common
        tot_diff += diff
    print(f"\n合計: 共通frame={tot_common} 差分={tot_diff} "
          f"({(tot_diff / tot_common if tot_common else 0):.2%})")


if __name__ == "__main__":
    main()
