# -*- coding: utf-8 -*-
"""「クロスチェック緩和=score単独に戻す」案の誤ラベルリスク実測 (2026-08-19)。

断片化された実試合 (ギャップ<6sで連結した game 群) について、
- 各断片の「score単独勝者」(断片末尾スコア比較 = 試合途中のリード側)
- 実試合の真の勝者 (最終断片の won ラベル、あればそれを真とみなす)
の一致率を測る。不一致率 = score単独に戻した場合に混入する誤ラベル率。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEW_DIR = ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
GAP = 6.0


def main() -> None:
    n_frag_eval = 0
    n_agree = 0
    n_disagree = 0
    n_snap_agree = 0
    n_snap_disagree = 0
    for f in sorted(NEW_DIR.glob("*.npz")):
        d = dict(np.load(f, allow_pickle=False))
        if len(d["game_idx"]) == 0:
            continue
        gidxs = sorted(set(int(g) for g in d["game_idx"]))
        # game情報
        info = {}
        for g in gidxs:
            rows = np.where(d["game_idx"] == g)[0]
            won = d["won"][rows]
            finals = {}
            for side in ("1P", "2P"):
                srows = rows[d["side"][rows] == side]
                finals[side] = int(d["score"][srows[-1]]) if len(srows) else None
            sw = None
            if finals["1P"] is not None and finals["2P"] is not None \
                    and finals["1P"] != finals["2P"]:
                sw = "1P" if finals["1P"] > finals["2P"] else "2P"
            w1 = d["won"][rows[d["side"][rows] == "1P"]]
            label = None
            if len(w1) and not np.isnan(w1[-1]):
                label = "1P" if w1[-1] == 1.0 else "2P"
            info[g] = dict(
                t0=float(d["t_sec"][rows[0]]), t1=float(d["t_sec"][rows[-1]]),
                sw=sw, label=label, n=len(rows),
                missing=bool(np.all(np.isnan(won))),
            )
        # ギャップ<6s で連結 → 実試合クラスタ
        clusters: list[list[int]] = [[gidxs[0]]]
        for j in range(1, len(gidxs)):
            if info[gidxs[j]]["t0"] - info[gidxs[j - 1]]["t1"] < GAP:
                clusters[-1].append(gidxs[j])
            else:
                clusters.append([gidxs[j]])
        for cl in clusters:
            if len(cl) < 2:
                continue
            true_w = info[cl[-1]]["label"]  # 最終断片のラベル = 真の勝者
            if true_w is None:
                continue
            for g in cl[:-1]:
                i = info[g]
                if not i["missing"] or i["sw"] is None:
                    continue
                n_frag_eval += 1
                if i["sw"] == true_w:
                    n_agree += 1
                    n_snap_agree += i["n"]
                else:
                    n_disagree += 1
                    n_snap_disagree += i["n"]
    print(f"評価対象断片 (欠損かつscore勝者計算可、真の勝者既知): {n_frag_eval}")
    print(f"score単独勝者 = 真の勝者: {n_agree} "
          f"({100.0 * n_agree / max(n_frag_eval, 1):.1f}%)")
    print(f"score単独勝者 ≠ 真の勝者 (誤ラベルになる): {n_disagree} "
          f"({100.0 * n_disagree / max(n_frag_eval, 1):.1f}%)")
    print(f"snapshot数: 正 {n_snap_agree} / 誤 {n_snap_disagree}")


if __name__ == "__main__":
    main()
