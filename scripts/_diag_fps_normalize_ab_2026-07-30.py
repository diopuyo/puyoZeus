"""fps正規化(stride2) A/B比較 (2026-07-30)。

data/verify/fps_normalize_ab_2026-07-30/ に置いた
「60fps動画を全フレーム処理した npz」対「2フレームおき(stride2)で処理した npz」
を突き合わせ、盤面・next検出・連鎖検知が同等かを定量する。

既存 scripts/_diag_sampling_corruption_rate_2026-07-30.py と同じ「直近有効
snapshot対応」方式を流用し、ファイル名のみ本検証用に差し替えている
(collect_boards_lean.py / collect_indicators_v2.py 本体は変更していない)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DIR = Path("data/verify/fps_normalize_ab_2026-07-30")
# 全フレーム版と比較する際、直近snapshotがこれより古ければ「対応なし」とする。
STALE_LIMIT_SEC: float = 3.0
SIDES = ("1P", "2P")


def load(path: Path) -> tuple:
    """npzから比較に必要な列を取り出す。"""
    z = np.load(path, allow_pickle=True)
    return (
        np.asarray(z["side"]).astype(str),
        np.asarray(z["t_sec"], dtype=float),
        np.asarray(z["grids"]),
        np.asarray(z["next1_a"]),
        np.asarray(z["next1_b"]),
        np.asarray(z["chain_trigger_sec"], dtype=float),
    )


def compare(all_path: Path, stride_path: Path) -> None:
    """全フレーム版を基準に stride 版の乖離を表示する。"""
    a_side, a_t, a_g, a_n1a, a_n1b, a_chain = load(all_path)
    s_side, s_t, s_g, s_n1a, s_n1b, s_chain = load(stride_path)
    print(f"snapshot数: allframes={len(a_t)} stride2={len(s_t)}")

    matched = diff_board = diff_col = diff_next = 0
    cell_diffs: list[int] = []
    col_gaps: list[int] = []
    for side in SIDES:
        ma, ms = a_side == side, s_side == side
        ta, ga, n1a_a, n1b_a = a_t[ma], a_g[ma], a_n1a[ma], a_n1b[ma]
        ts, gs, n1a_s, n1b_s = s_t[ms], s_g[ms], s_n1a[ms], s_n1b[ms]
        if len(ta) == 0 or len(ts) == 0:
            continue
        order = np.argsort(ta)
        ta_o, ga_o, n1a_o, n1b_o = ta[order], ga[order], n1a_a[order], n1b_a[order]
        for i, t0 in enumerate(ts):
            k = int(np.searchsorted(ta_o, t0, side="right")) - 1
            if k < 0 or (t0 - ta_o[k]) > STALE_LIMIT_SEC:
                continue
            matched += 1
            g1, g2 = gs[i], ga_o[k]
            if not np.array_equal(g1, g2):
                diff_board += 1
                cell_diffs.append(int((g1 != g2).sum()))
                for c in range(g1.shape[1]):
                    n1 = int((g1[:, c] != 0).sum())
                    n2 = int((g2[:, c] != 0).sum())
                    if (n1 == 0) != (n2 == 0) and max(n1, n2) >= 4:
                        diff_col += 1
                        col_gaps.append(abs(n1 - n2))
            if n1a_s[i] != -1 and n1a_o[k] != -1 and (
                (n1a_s[i], n1b_s[i]) != (n1a_o[k], n1b_o[k])
            ):
                diff_next += 1

    print(f"対応snapshot数: {matched}")
    if matched:
        print(f"盤面が違う: {diff_board} ({diff_board/matched*100:.1f}%)")
        print(f"列まるごと欠損: {diff_col} ({diff_col/matched*100:.1f}%)")
        print(f"next不一致(両方実値取得時): {diff_next}")
    if cell_diffs:
        arr = np.array(cell_diffs)
        print(f"盤面が違うときのセル差: 中央値{np.median(arr):.0f} 最大{arr.max()} (78セル中)")
    if col_gaps:
        arr = np.array(col_gaps)
        print(f"列欠損時のセル数差: 中央値{np.median(arr):.0f} 最大{arr.max()}")

    a_rate = float(np.mean(~np.isnan(a_chain))) if len(a_chain) else 0.0
    s_rate = float(np.mean(~np.isnan(s_chain))) if len(s_chain) else 0.0
    print(f"chain_trigger_sec 非NaN率: allframes={a_rate*100:.1f}% stride2={s_rate*100:.1f}%")


def main() -> None:
    """DIR 配下の *_allframes.npz / *_stride2.npz ペアを全て比較する。"""
    all_paths = sorted(DIR.glob("*_allframes.npz"))
    if not all_paths:
        print("全フレーム版npzがまだ無い (収集ジョブ完走待ち)")
        return
    for all_path in all_paths:
        stride_path = DIR / all_path.name.replace("_allframes.npz", "_stride2.npz")
        if not stride_path.exists():
            print(f"{all_path.name}: 対応するstride2版が無い (スキップ)")
            continue
        print(f"=== {all_path.stem.replace('_allframes', '')} ===")
        compare(all_path, stride_path)


if __name__ == "__main__":
    main()
