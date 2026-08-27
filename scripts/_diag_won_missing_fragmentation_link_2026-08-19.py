# -*- coding: utf-8 -*-
"""won欠損と試合断片化の因果リンク診断 (2026-08-19)。

仮説: multisignal境界が試合中に誤発火 → 1試合が複数game_idxに分割 →
パネルクロスチェックの比較窓 (game g の開始 vs game g+1 の開始) が
「同一試合の途中同士」になりパネル数字差分ゼロ → panel_winner=None →
2系統一致要求で won=NaN。

検証ロジック:
- 各 game g の「開始スコア」(両side先頭snapshotのscore最大値) を見る。
  本物の試合開始なら ≈0。断片B (試合途中から始まる偽game) なら高値。
- 断片B が存在する ⇔ その直前 game (断片A) の won が欠損するはず。
- P(missing(g) | g+1が断片B) vs P(missing(g) | g+1が正常開始) を比較。

出力: 標準出力サマリ + logs/_diag_won_missing_fragmentation_link_2026-08-19.tsv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEW_DIR = ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
OUT_TSV = ROOT / "logs" / "_diag_won_missing_fragmentation_link_2026-08-19.tsv"

# 試合開始直後スコアの上限: これを超えて始まる game は「試合途中からの断片」とみなす
FRAGMENT_START_SCORE = 2000


def main() -> None:
    rows_out = ["video\tgame\tmissing\tstart_score\tis_fragment_start\tnext_is_fragment"]
    # 集計: 条件付き欠損率
    n_g_nextfrag_miss = 0
    n_g_nextfrag = 0
    n_g_nextnorm_miss = 0
    n_g_nextnorm = 0
    n_game0_miss = 0
    n_game0 = 0
    n_lastgame_miss = 0
    n_lastgame = 0
    n_frag_start_games = 0
    n_games = 0
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
            missing = bool(np.all(np.isnan(won)))
            # 開始スコア: 各sideの先頭snapshotのscore (>=0のみ)
            starts = []
            for side in ("1P", "2P"):
                srows = rows[d["side"][rows] == side]
                if len(srows):
                    s = int(d["score"][srows[0]])
                    if s >= 0:
                        starts.append(s)
            start_score = max(starts) if starts else -1
            info[g] = dict(missing=missing, start_score=start_score,
                           frag=(start_score >= FRAGMENT_START_SCORE))
        for j, g in enumerate(gidxs):
            n_games += 1
            i = info[g]
            if i["frag"]:
                n_frag_start_games += 1
            nxt = info.get(gidxs[j + 1]) if j + 1 < len(gidxs) else None
            nxt_frag = bool(nxt and nxt["frag"])
            if g == 0:
                n_game0 += 1
                if i["missing"]:
                    n_game0_miss += 1
            elif j == len(gidxs) - 1:
                n_lastgame += 1
                if i["missing"]:
                    n_lastgame_miss += 1
            elif nxt is not None:
                # 中間game (game0と最終を除く) のみ条件付き比較
                if nxt_frag:
                    n_g_nextfrag += 1
                    if i["missing"]:
                        n_g_nextfrag_miss += 1
                else:
                    n_g_nextnorm += 1
                    if i["missing"]:
                        n_g_nextnorm_miss += 1
            rows_out.append(
                f"{vid}\t{g}\t{int(i['missing'])}\t{i['start_score']}\t"
                f"{int(i['frag'])}\t{int(nxt_frag)}"
            )
    OUT_TSV.write_text("\n".join(rows_out), encoding="utf-8")
    print(f"total games: {n_games}")
    print(f"fragment-start games (start_score>={FRAGMENT_START_SCORE}): "
          f"{n_frag_start_games} ({100.0 * n_frag_start_games / n_games:.1f}%)")
    print(f"game0: missing {n_game0_miss}/{n_game0}")
    print(f"last-game: missing {n_lastgame_miss}/{n_lastgame}")
    if n_g_nextfrag:
        print(f"P(missing | next game is fragment) = "
              f"{n_g_nextfrag_miss}/{n_g_nextfrag} = "
              f"{100.0 * n_g_nextfrag_miss / n_g_nextfrag:.1f}%")
    if n_g_nextnorm:
        print(f"P(missing | next game starts normally) = "
              f"{n_g_nextnorm_miss}/{n_g_nextnorm} = "
              f"{100.0 * n_g_nextnorm_miss / n_g_nextnorm:.1f}%")
    print(f"TSV -> {OUT_TSV}")


if __name__ == "__main__":
    main()
