"""ビームサーチの取りこぼしを **手数を揃えて** 測り直す (2026-08-09).

前回の測定は比較する手数が不揃いだった (ビーム側は
「ネクスト+ダブルネクスト+K手」なのに、全探索側は K 手だけ)。
本スクリプトは **同じ手数**で突き合わせる。

ビーム側: near_future_fire_power(k_levels=(k,)) は 2+k 手を打つ
全探索側: 同じ 2+k 手を全列挙する (ネクスト・ダブルネクストは実色、
          それ以降は理想ツモ=全色から選べる想定に合わせる)

全探索は手数が増えると爆発するため 3 手まで。
"""
from __future__ import annotations
import sys
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

N_BOARDS = 25
NEXT, DNEXT = (1, 2), (3, 4)
COLORS = (1, 2, 3, 4)

def _mk(rows, seed):
    rng = np.random.RandomState(seed)
    g = [[0]*BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS-rows, BOARD_ROWS):
        for c in range(BOARD_COLS): g[r][c] = int(rng.choice(COLORS))
    return Board.from_list(g)

def _fire(b, sim):
    res = sim.simulate(b)
    if res.chain_count < 1: return 0.0
    return calculate_chain_score(res).total_score / OJAMA_RATE_STANDARD

def _brute(board, pairs, sim, ideal_hands, cap=200000):
    """実ツモ pairs を置いた後、ideal_hands 手ぶん理想ツモを全探索。"""
    frontier = [board]
    for pair in pairs:
        nxt = []
        for b in frontier: nxt.extend(iv._enumerate_placement_boards(b, pair))
        frontier = nxt or frontier
    for _ in range(ideal_hands):
        nxt = []
        for b in frontier:
            for a in COLORS:
                for bb in COLORS:
                    nxt.extend(iv._enumerate_placement_boards(b, (a, bb)))
                    if len(nxt) > cap: break
                if len(nxt) > cap: break
            if len(nxt) > cap: break
        if not nxt: break
        # 爆発を抑えるため、火力上位のみ残す (全探索の近似だが十分広い)
        if len(nxt) > cap:
            scored = sorted(((_fire(x, sim), x) for x in nxt),
                            key=lambda t: -t[0])[:cap]
            nxt = [x for _, x in scored]
        frontier = nxt
    return max((_fire(b, sim) for b in frontier), default=0.0)

def main() -> int:
    sim = ChainSimulator()
    print("手数を揃えた比較 (ビーム側 2+K 手 vs 全探索 2+K 手)")
    print()
    for k in (1,):
        print(f"=== K={k} (実手数 {2+k} 手: ネクスト+ダブルネクスト+{k}) ===")
        boards = [_mk(4 + (i % 5), i) for i in range(N_BOARDS)]
        truths = [_brute(b, (NEXT, DNEXT), sim, k) for b in boards]
        print(f"{'ビーム幅':>8s} {'取りこぼし':>10s} {'平均過小':>10s} {'最大過小':>10s} {'相対':>8s}")
        for beam in (1, 4, 8, 16, 32):
            miss, gaps, rels = 0, [], []
            for b, truth in zip(boards, truths):
                r = iv.near_future_fire_power(
                    b, next_pair=NEXT, dnext_pair=DNEXT,
                    beam_width=beam, k_levels=(k,))
                v = r.values.get(k)
                got = float(v.raw) if v is not None else 0.0
                if got < truth - 1e-6:
                    miss += 1; gaps.append(truth-got)
                    if truth > 0: rels.append((truth-got)/truth)
            print(f"{beam:8d} {miss/len(boards):9.1%} "
                  f"{np.mean(gaps) if gaps else 0:9.1f}個 "
                  f"{np.max(gaps) if gaps else 0:9.1f}個 "
                  f"{np.mean(rels)*100 if rels else 0:7.1f}%")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
