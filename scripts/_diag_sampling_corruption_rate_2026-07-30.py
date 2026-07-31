"""間引きが盤面を壊す割合を、時刻を合わせた盤面比較で定量する。

間引きあり(--sample-interval 0.2 = 実質5fps)と間引きなし(全フレーム)を
同一開始時刻・同一長さで収集したnpzを突き合わせる。
間引きあり側の各snapshotに対し、間引きなし側で最も時刻が近いsnapshotを対応させ、
盤面が一致するか・列がまるごと違うかを数える。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DIR = Path("data/indicators_v2/sampling_rate_2026-07-30")
# npzのsnapshotは「この時刻にこの盤面になった」という記録なので、ある時刻に有効な盤面は
# 「その時刻以前の直近のsnapshot」である。単純な最近傍対応では時刻のズレ(30fpsで数フレーム)を
# 盤面の違いと誤認するため、この「有効な盤面」同士を比べる。
STALE_LIMIT_SEC = 3.0  # 直近snapshotがこれより古い場合は比較対象にしない
SIDES = ("1P", "2P")


def load(path: Path) -> tuple:
    """npzから side/t_sec/grids を取り出す。"""
    z = np.load(path, allow_pickle=True)
    return (np.asarray(z["side"]).astype(str),
            np.asarray(z["t_sec"], dtype=float),
            np.asarray(z["grids"]))


def compare_one(vid: str) -> dict | None:
    """1動画分を比較して集計を返す。対応が取れなければ None。"""
    p_s, p_a = DIR / f"{vid}_sampled.npz", DIR / f"{vid}_allframes.npz"
    if not (p_s.exists() and p_a.exists()):
        return None
    s_side, s_t, s_g = load(p_s)
    a_side, a_t, a_g = load(p_a)

    matched = diff_board = diff_col = 0
    col_gaps: list[int] = []
    cell_diffs: list[int] = []
    for side in SIDES:
        ms, ma = s_side == side, a_side == side
        ts, gs = s_t[ms], s_g[ms]
        ta, ga = a_t[ma], a_g[ma]
        if len(ta) == 0:
            continue
        order = np.argsort(ta)
        ta_s, ga_s = ta[order], ga[order]
        for i, t0 in enumerate(ts):
            # t0 以前の直近snapshot = その時刻に有効な盤面
            k = int(np.searchsorted(ta_s, t0, side="right")) - 1
            if k < 0 or (t0 - ta_s[k]) > STALE_LIMIT_SEC:
                continue
            j = k
            matched += 1
            g1, g2 = gs[i], ga_s[j]
            if not np.array_equal(g1, g2):
                diff_board += 1
                cell_diffs.append(int((g1 != g2).sum()))
                for c in range(g1.shape[1]):
                    n1 = int((g1[:, c] != 0).sum())
                    n2 = int((g2[:, c] != 0).sum())
                    # 一方が完全に空で他方に4個以上ある = 列まるごとの欠損
                    if (n1 == 0) != (n2 == 0) and max(n1, n2) >= 4:
                        diff_col += 1
                        col_gaps.append(abs(n1 - n2))
    return {"video": vid, "n_sampled": int((s_side != "").sum()),
            "n_allframes": int((a_side != "").sum()), "matched": matched,
            "diff_board": diff_board, "diff_col": diff_col,
            "col_gaps": col_gaps, "cell_diffs": cell_diffs}


def main() -> None:
    """完走している動画すべてを比較して表示する。"""
    vids = sorted({p.name.split("_")[0] for p in DIR.glob("*_allframes.npz")})
    if not vids:
        print("全フレーム版がまだ無い")
        sys.exit(0)
    tot_m = tot_b = tot_c = 0
    all_gaps: list[int] = []
    all_cells: list[int] = []
    print(f"{'動画':<6}{'snapshot数(間引き/全)':<22}{'対応':<7}"
          f"{'盤面が違う':<12}{'列まるごと欠損':<14}")
    for vid in vids:
        r = compare_one(vid)
        if r is None:
            continue
        tot_m += r["matched"]; tot_b += r["diff_board"]; tot_c += r["diff_col"]
        all_gaps += r["col_gaps"]; all_cells += r["cell_diffs"]
        pct_b = r["diff_board"] / r["matched"] * 100 if r["matched"] else 0.0
        print(f"{vid:<6}{r['n_sampled']:>4} / {r['n_allframes']:<15}"
              f"{r['matched']:>5}  {r['diff_board']:>5} ({pct_b:5.1f}%) "
              f"{r['diff_col']:>10}")
    if tot_m:
        print(f"\n合計: 対応{tot_m}件, 盤面が違う {tot_b}件 "
              f"({tot_b/tot_m*100:.1f}%), 列まるごと欠損 {tot_c}件 "
              f"({tot_c/tot_m*100:.1f}%)")
    if all_cells:
        a = np.array(all_cells)
        print(f"盤面が違うとき何セル違うか: 中央値{np.median(a):.0f} "
              f"p90={np.percentile(a, 90):.0f} 最大{a.max()} "
              f"(78セル中) / 1-2セルのみ={int((a <= 2).sum())}件"
              f"({(a <= 2).mean()*100:.0f}%)")
    if all_gaps:
        a = np.array(all_gaps)
        print(f"列欠損時のセル数差: 中央値{np.median(a):.0f} "
              f"p90={np.percentile(a, 90):.0f} 最大{a.max()}")


if __name__ == "__main__":
    main()
