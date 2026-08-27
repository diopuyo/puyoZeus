"""連結ボーナスを 0 に近似すると得点がどれだけズレるかを実測する (2026-08-09).

user 確認: 「これでそんなに変わるんですか?」

## 背景
`chain_bitboard.simulate_batch_with_approx_score` は速度のために
**連結ボーナスを 0 (=4連結相当) に近似**している。 bitboard は
「4 つ以上つながっているか」しか判定せず、 グループの個数 (5,6,7...) を
保持しないため。 連鎖ボーナス・色数ボーナスは公式テーブルで厳密。

連結ボーナスは 4連結=0 / 5=2 / 6=3 / 7=4 / 8=5 / 9=6 / 10=7 / 11+=10。

## 測ること
実データ (Phase L の盤面) を使い、 厳密計算 (ChainSimulator +
calculate_chain_score) と近似計算の得点差を出す。
- 得点の相対誤差の分布
- **おじゃま換算での差** (実際に効くのはこちら。 70点=1個)
- 誤差が大きい盤面の特徴 (大きい連結が多いか)

読み取り専用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.chain_bitboard import (  # noqa: E402
    batch_from_boards,
    simulate_batch_with_approx_score,
)
from src.scoring import OJAMA_RATE_STANDARD, calculate_chain_score  # noqa: E402

NPZ_DIR = _ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-07"
# サンプル数 (盤面)
N_SAMPLES: int = 3000


def main() -> int:
    npzs = sorted(NPZ_DIR.glob("*.npz"))[:20]
    grids = []
    for p in npzs:
        d = np.load(p, allow_pickle=True)
        g = d["grids"]
        if len(g) == 0:
            continue
        idx = np.linspace(0, len(g) - 1, min(200, len(g))).astype(int)
        grids.extend(g[idx])
    grids = grids[:N_SAMPLES]
    if not grids:
        print("盤面が無い")
        return 1
    boards = [Board.from_list([list(map(int, r)) for r in g]) for g in grids]

    sim = ChainSimulator()
    exact_scores, approx_scores, chain_counts = [], [], []
    batch = batch_from_boards(boards)
    approx = simulate_batch_with_approx_score(batch)
    for b, ap in zip(boards, approx):
        res = sim.simulate(b)
        if res.chain_count < 1:
            continue
        exact_scores.append(float(calculate_chain_score(res).total_score))
        approx_scores.append(float(ap.score_approx))
        chain_counts.append(int(res.chain_count))

    if not exact_scores:
        print("連鎖が起きる盤面が無かった")
        return 1
    e = np.array(exact_scores)
    a = np.array(approx_scores)
    cc = np.array(chain_counts)
    diff = e - a                      # 正 = 近似が過小
    rel = np.where(e > 0, diff / e, 0.0)
    ojama_diff = diff / OJAMA_RATE_STANDARD

    print(f"連鎖が起きた盤面 {len(e)} 件 (連鎖数 中央値 {np.median(cc):.0f})")
    print()
    print("=== 得点の相対誤差 (正 = 近似が過小) ===")
    for q in (50, 75, 90, 99):
        print(f"  p{q}: {np.percentile(rel, q):6.1%}")
    print(f"  平均: {rel.mean():6.1%}  最大: {rel.max():6.1%}")
    print()
    print("=== おじゃま換算の差 (70点 = 1個) ===")
    for q in (50, 75, 90, 99):
        print(f"  p{q}: {np.percentile(ojama_diff, q):6.1f} 個")
    print(f"  平均: {ojama_diff.mean():6.1f} 個  最大: {ojama_diff.max():6.1f} 個")
    print()
    print("=== 連鎖数別の平均誤差 ===")
    for lo, hi in ((1, 2), (2, 4), (4, 7), (7, 10), (10, 99)):
        m = (cc >= lo) & (cc < hi)
        if m.sum() < 10:
            continue
        print(f"  {lo:2d}〜{hi - 1:2d}連鎖 (n={int(m.sum()):5d}): "
              f"相対 {rel[m].mean():6.1%} / おじゃま {ojama_diff[m].mean():5.1f} 個")
    print()
    print("判定の目安: 応手判定の閾値は 12 個。 おじゃま換算の差がこれに対して")
    print("            どれだけ大きいかで、 実用上の影響が決まる。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
