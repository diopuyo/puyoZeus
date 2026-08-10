"""ビームサーチの取りこぼしを全探索と突き合わせて実測する (2026-08-09).

user 指摘: 「ビームサーチで探索しきれないものもあるのでは?」

## なぜ取りこぼすか
ビームサーチは各手で **評価が上位 N 個の盤面だけ**を残して次の手に進む。
そのため「**途中は評価が低いが、 後で大きく伸びる**」手順が枝刈りされる。
連鎖は「今は消えないが後で繋がる」形を作るゲームなので、 この性質と
相性が悪い可能性がある。

## 測ること
K=1,2 は全探索が現実的 (22 / 484 通り) なので、 **全探索の真値**と
**ビームサーチの結果**を突き合わせる。
- 何 % の盤面で真値を取りこぼすか
- 取りこぼした場合、 どれだけ過小評価するか
- ビーム幅を変えると取りこぼしがどう変わるか

K>=3 は全探索ができないため、 K=1,2 の結果から傾向を推定する。

読み取り専用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

import src.indicators_v2 as iv  # noqa: E402
from src.scoring import OJAMA_RATE_STANDARD, calculate_chain_score  # noqa: E402

N_BOARDS: int = 40
NEXT, DNEXT = (1, 2), (3, 4)


def _mk(rows: int, seed: int) -> Board:
    rng = np.random.RandomState(seed)
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - rows, BOARD_ROWS):
        for c in range(BOARD_COLS):
            g[r][c] = int(rng.choice([1, 2, 3, 4]))
    return Board.from_list(g)


def _best_ojama(board: Board, sim: ChainSimulator) -> float:
    """この盤面を即発火したときのお邪魔換算得点。"""
    res = sim.simulate(board)
    if res.chain_count < 1:
        return 0.0
    return calculate_chain_score(res).total_score / OJAMA_RATE_STANDARD


def _brute_force(board: Board, pairs, sim: ChainSimulator) -> float:
    """ツモ列を全探索して到達できる最大お邪魔量を返す (真値)。"""
    frontier = [board]
    for pair in pairs:
        nxt = []
        for b in frontier:
            nxt.extend(iv._enumerate_placement_boards(b, pair))
        if not nxt:
            return 0.0
        frontier = nxt
    return max(_best_ojama(b, sim) for b in frontier)


def main() -> int:
    sim = ChainSimulator()
    print("ビームサーチ vs 全探索 (K=1: 22通り / K=2: 484通り)")
    print()
    for n_hands, pairs in ((1, (NEXT,)), (2, (NEXT, DNEXT))):
        print(f"=== {n_hands} 手先 ===")
        print(f"{'ビーム幅':>8s} {'取りこぼし':>10s} {'平均過小':>10s} {'最大過小':>10s}")
        truths = []
        boards = []
        for i in range(N_BOARDS):
            b = _mk(rows=4 + (i % 6), seed=i)
            boards.append(b)
            truths.append(_brute_force(b, pairs, sim))
        for beam in (1, 2, 4, 8, iv.NEAR_FUTURE_BEAM_WIDTH, 32):
            miss = 0
            gaps = []
            for b, truth in zip(boards, truths):
                r = iv.near_future_fire_power(
                    b, next_pair=NEXT,
                    dnext_pair=DNEXT if n_hands >= 2 else None,
                    beam_width=beam, k_levels=(1,),
                )
                # k_levels=(1,) は「ネクスト+ダブルネクスト+1手」なので
                # 手数が違う。 raw (お邪魔換算) を取り出して比較する。
                v = r.values.get(1)
                got = float(v.raw) if v is not None else 0.0
                if got < truth - 1e-6:
                    miss += 1
                    gaps.append(truth - got)
            rate = miss / len(boards)
            avg = float(np.mean(gaps)) if gaps else 0.0
            mx = float(np.max(gaps)) if gaps else 0.0
            print(f"{beam:8d} {rate:9.1%} {avg:9.1f}個 {mx:9.1f}個")
        print()
    print("注: ビームサーチは各手で上位 N 個しか残さないため、")
    print("    『途中は低評価だが後で伸びる』手順を枝刈りする。")
    print("    取りこぼし率が高ければ、 深く読んでも真値に届かない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
