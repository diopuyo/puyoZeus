"""ビームの枝刈りが「同点だらけ」で機能していない仮説を実測する (2026-08-09).

## 仮説
ビームは各手で **シミュレート後の得点** 降順 top-N を残す
(src/indicators_v2.py:2340-2383)。 しかし連鎖を組んでいる途中の盤面は
**まだ何も消えない=得点0** が大半。 同点が何百通りもある中で
「上位N件」を選ぶのは、 実質 **先頭N件を取っているだけ**。

これが正しければ:
- 幅を広げても当たりを引く確率はほとんど上がらない (実測と一致)
- **真因は深さでも幅でもなく、 評価関数が平坦なこと**

## 測ること
各手ごとに: 候補数 / 得点0の割合 / 得点が同点の最大群サイズ
読み取り専用。
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
from src.board import BOARD_COLS, BOARD_ROWS, Board
from src.chain import ChainSimulator
from src.console_init import init_console
init_console()
import src.indicators_v2 as iv
from src.scoring import OJAMA_RATE_STANDARD, calculate_chain_score

COLORS = (1, 2, 3, 4)

def _mk(rows, seed):
    rng = np.random.RandomState(seed)
    g = [[0]*BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS-rows, BOARD_ROWS):
        for c in range(BOARD_COLS): g[r][c] = int(rng.choice(COLORS))
    return Board.from_list(g)

def _score(b, sim):
    res = sim.simulate(b)
    if res.chain_count < 1: return 0.0
    return calculate_chain_score(res).total_score / OJAMA_RATE_STANDARD

def main() -> int:
    sim = ChainSimulator()
    print("ビーム各手の候補の得点分布 (幅8で枝刈りする直前)")
    print()
    print(f"{'積み段':>6s} {'手':>3s} {'候補数':>7s} {'得点0の割合':>12s} {'最大同点群':>10s} {'異なる得点':>10s}")
    print("-" * 56)
    for rows in (3, 6, 9):
        b0 = _mk(rows, seed=rows)
        frontier = [b0]
        for hand in range(1, 5):
            cands = []
            for b in frontier:
                for a in COLORS:
                    for bb in COLORS:
                        cands.extend(iv._enumerate_placement_boards(b, (a, bb)))
            if not cands: break
            scores = [_score(x, sim) for x in cands]
            cnt = Counter(round(s, 4) for s in scores)
            zero = sum(1 for s in scores if s <= 1e-9) / len(scores)
            print(f"{rows:6d} {hand:3d} {len(cands):7d} {zero:11.1%} "
                  f"{max(cnt.values()):10d} {len(cnt):10d}")
            order = np.argsort([-s for s in scores])[:8]
            frontier = [cands[i] for i in order]
        print()
    print("読み方: 得点0の割合が高く最大同点群が幅8を大きく超えるなら、")
    print("        枝刈りは実質ランダム選択。 幅を広げても改善しないのは必然。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
