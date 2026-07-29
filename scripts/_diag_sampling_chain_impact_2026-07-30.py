"""間引きが連鎖系指標をどれだけ狂わせるかを、対になった盤面で直接測る。

追加の収集は不要。`sampling_rate_2026-07-30` の間引きあり/なしのnpzを突き合わせ、
同じ時刻に有効な盤面それぞれで current_max_chain (今1手で撃てる最大連鎖) を計算し比較する。

狙い: 「盤面が11.9%で列欠損する」ことが連鎖系指標に実害を与えるかを金額換算する。
密度系は雑音で済む可能性があるが、連鎖系は列が1本欠けると計算が破綻するため。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import src.indicators_v2 as iv
from src.board import Board

DIR = Path("data/indicators_v2/sampling_rate_2026-07-30")
STALE_LIMIT_SEC = 3.0
SIDES = ("1P", "2P")


def load(path: Path) -> tuple:
    """npzから side/t_sec/grids を取り出す。"""
    z = np.load(path, allow_pickle=True)
    return (np.asarray(z["side"]).astype(str),
            np.asarray(z["t_sec"], dtype=float),
            np.asarray(z["grids"]))


def max_chain(grid: np.ndarray) -> int:
    """盤面の「今1手で撃てる最大連鎖」を返す。"""
    return int(iv.current_max_chain(Board.from_list(grid.tolist())).raw)


def collect_pairs(vid: str) -> list[tuple[int, int]]:
    """(間引きあり, 間引きなし) の最大連鎖の対を集める。"""
    p_s, p_a = DIR / f"{vid}_sampled.npz", DIR / f"{vid}_allframes.npz"
    if not (p_s.exists() and p_a.exists()):
        return []
    s_side, s_t, s_g = load(p_s)
    a_side, a_t, a_g = load(p_a)
    pairs: list[tuple[int, int]] = []
    for side in SIDES:
        ms, ma = s_side == side, a_side == side
        ts, gs = s_t[ms], s_g[ms]
        ta, ga = a_t[ma], a_g[ma]
        if len(ta) == 0:
            continue
        order = np.argsort(ta)
        ta_s, ga_s = ta[order], ga[order]
        for i, t0 in enumerate(ts):
            k = int(np.searchsorted(ta_s, t0, side="right")) - 1
            if k < 0 or (t0 - ta_s[k]) > STALE_LIMIT_SEC:
                continue
            g1, g2 = gs[i], ga_s[k]
            if np.array_equal(g1, g2):
                pairs.append((0, 0))  # 同一なら差は0 (計算を省く印)
                continue
            pairs.append((max_chain(g1), max_chain(g2)))
    return pairs


def main() -> None:
    """動画ごとに集計して表示する。"""
    vids = sorted({p.name.split("_")[0] for p in DIR.glob("*_allframes.npz")})
    tot_n = tot_diff = tot_under = tot_over = 0
    gaps: list[int] = []
    print(f"{'動画':<6}{'対':<6}{'連鎖数が違う':<14}"
          f"{'間引きが過小':<14}{'間引きが過大':<14}")
    for vid in vids:
        pairs = collect_pairs(vid)
        n = len(pairs)
        d = [(a, b) for a, b in pairs if a != b]
        under = sum(1 for a, b in d if a < b)
        over = sum(1 for a, b in d if a > b)
        gaps += [b - a for a, b in d]
        tot_n += n; tot_diff += len(d); tot_under += under; tot_over += over
        pct = len(d) / n * 100 if n else 0.0
        print(f"{vid:<6}{n:<6}{len(d):>5} ({pct:5.1f}%)   "
              f"{under:>5}        {over:>5}")
    if tot_n:
        print(f"\n合計: 対{tot_n}件, 連鎖数が違う {tot_diff}件 "
              f"({tot_diff/tot_n*100:.1f}%)")
        print(f"  間引きが過小に出る: {tot_under}件 / "
              f"過大: {tot_over}件")
    if gaps:
        a = np.array(gaps)
        print(f"  差(全フレーム - 間引き): 中央値{np.median(a):.0f} "
              f"平均{a.mean():+.2f} p90={np.percentile(a, 90):.0f} 最大{a.max()}")


if __name__ == "__main__":
    main()
