# -*- coding: utf-8 -*-
"""won欠損の残り851件を説明するギャップベース断片判定 (2026-08-19)。

断片判定v2: game g と g+1 の間の snapshot 時刻ギャップ
(g+1先頭t_sec - g末尾t_sec)。実境界はリザルト画面+メニューで大きく空く。
試合中の誤発火境界はほぼ連続 (数秒以内)。

開始スコア (v1) と組み合わせ、欠損904件 (game0/最終試合を除く中間game) を
「次gameが低ギャップ or 高開始スコア = 断片」で説明できる割合を出す。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEW_DIR = ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
OUT_TSV = ROOT / "logs" / "_diag_won_missing_gap_analysis_2026-08-19.tsv"

FRAGMENT_START_SCORE = 2000
# 実境界の最短ギャップ: 勝敗演出+リザルト+メニュー+レディーゴーで通常10秒超
FRAGMENT_GAP_SEC = 6.0


def main() -> None:
    rows_out = ["video\tgame\tmissing\tstart_score\tgap_before\tgap_after\t"
                "next_frag_score\tnext_frag_gap\tdur"]
    stats = {  # (next_is_frag, missing) の2x2
        (True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0,
    }
    self_frag_miss = {  # 自分自身が断片Bかどうか x missing
        (True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0,
    }
    gap_hist_miss = []
    gap_hist_ok = []
    for f in sorted(NEW_DIR.glob("*.npz")):
        vid = f.stem
        d = dict(np.load(f, allow_pickle=False))
        if len(d["game_idx"]) == 0:
            continue
        gidxs = sorted(set(int(g) for g in d["game_idx"]))
        info: dict[int, dict] = {}
        for g in gidxs:
            rows = np.where(d["game_idx"] == g)[0]
            won = d["won"][rows]
            starts = []
            for side in ("1P", "2P"):
                srows = rows[d["side"][rows] == side]
                if len(srows):
                    s = int(d["score"][srows[0]])
                    if s >= 0:
                        starts.append(s)
            info[g] = dict(
                missing=bool(np.all(np.isnan(won))),
                start_score=max(starts) if starts else -1,
                t_first=float(d["t_sec"][rows[0]]),
                t_last=float(d["t_sec"][rows[-1]]),
            )
        for j, g in enumerate(gidxs):
            i = info[g]
            gap_before = (
                i["t_first"] - info[gidxs[j - 1]]["t_last"] if j > 0 else -1.0
            )
            nxt = info.get(gidxs[j + 1]) if j + 1 < len(gidxs) else None
            gap_after = nxt["t_first"] - i["t_last"] if nxt else -1.0
            nfs = bool(nxt and nxt["start_score"] >= FRAGMENT_START_SCORE)
            nfg = bool(nxt and 0 <= gap_after < FRAGMENT_GAP_SEC)
            self_frag = bool(
                i["start_score"] >= FRAGMENT_START_SCORE
                or (0 <= gap_before < FRAGMENT_GAP_SEC)
            )
            dur = i["t_last"] - i["t_first"]
            if g != 0 and j != len(gidxs) - 1 and nxt is not None:
                next_is_frag = nfs or nfg
                stats[(next_is_frag, i["missing"])] += 1
                self_frag_miss[(self_frag, i["missing"])] += 1
                if nxt:
                    (gap_hist_miss if i["missing"] else gap_hist_ok).append(gap_after)
            rows_out.append(
                f"{vid}\t{g}\t{int(i['missing'])}\t{i['start_score']}\t"
                f"{gap_before:.1f}\t{gap_after:.1f}\t{int(nfs)}\t{int(nfg)}\t{dur:.1f}"
            )
    OUT_TSV.write_text("\n".join(rows_out), encoding="utf-8")
    n_frag = stats[(True, True)] + stats[(True, False)]
    n_norm = stats[(False, True)] + stats[(False, False)]
    print("--- 中間game (game0/最終を除く) ---")
    print(f"P(missing | next=断片[スコア or ギャップ<{FRAGMENT_GAP_SEC}s]) = "
          f"{stats[(True, True)]}/{n_frag} = "
          f"{100.0 * stats[(True, True)] / max(n_frag, 1):.1f}%")
    print(f"P(missing | next=正常境界) = "
          f"{stats[(False, True)]}/{n_norm} = "
          f"{100.0 * stats[(False, True)] / max(n_norm, 1):.1f}%")
    sf = self_frag_miss
    print(f"参考: 自分が断片B: missing {sf[(True, True)]}/"
          f"{sf[(True, True)] + sf[(True, False)]}, "
          f"自分が正常: missing {sf[(False, True)]}/"
          f"{sf[(False, True)] + sf[(False, False)]}")
    gm = np.array(gap_hist_miss)
    go = np.array(gap_hist_ok)
    if len(gm):
        print(f"gap_after 分布 (missing): median={np.median(gm):.1f}s, "
              f"p25={np.percentile(gm, 25):.1f}, p75={np.percentile(gm, 75):.1f}")
    if len(go):
        print(f"gap_after 分布 (labeled): median={np.median(go):.1f}s, "
              f"p25={np.percentile(go, 25):.1f}, p75={np.percentile(go, 75):.1f}")
    print(f"TSV -> {OUT_TSV}")


if __name__ == "__main__":
    main()
