# -*- coding: utf-8 -*-
"""won欠損490試合の内訳診断 (2026-08-19)。

新方式42本 (boards_lean_subset50_2026-08-19) の各 (video, game_idx) について:
- won 欠損 (全NaN) か
- npz末尾スコアから score系統の勝者が計算できたか (offline再現)
- 窒息フォールバック (_winner_by_survival 相当) が効くか
- lockdown比率・試合長・snapshot数
を集計し、旧方式 (boards_lean_phase_l_2026-08-11) の同試合と対応づける。

出力: logs/_diag_won_missing_breakdown_2026-08-19.tsv + 標準出力サマリ
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEW_DIR = ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
OLD_DIR = ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-11"
OUT_TSV = ROOT / "logs" / "_diag_won_missing_breakdown_2026-08-19.tsv"

_DEATH_ROW, _DEATH_COL = 1, 2


def game_rows(d: dict, gidx: int) -> np.ndarray:
    return np.where(d["game_idx"] == gidx)[0]


def offline_score_winner(d: dict, rows: np.ndarray) -> str | None:
    """npz内の最終snapshot scoreからscore系統勝者を近似再現。"""
    finals: dict[str, int | None] = {"1P": None, "2P": None}
    for side in ("1P", "2P"):
        srows = rows[d["side"][rows] == side]
        if len(srows) == 0:
            continue
        s = int(d["score"][srows[-1]])
        finals[side] = s if s >= 0 else None
    s1, s2 = finals["1P"], finals["2P"]
    if s1 is not None and s2 is not None and s1 != s2:
        return "1P" if s1 > s2 else "2P"
    return None


def offline_survival_winner(d: dict, rows: np.ndarray) -> str | None:
    """_winner_by_survival 相当: 各sideの末尾gridで窒息セル確認。"""
    choked: dict[str, bool | None] = {"1P": None, "2P": None}
    for side in ("1P", "2P"):
        srows = rows[d["side"][rows] == side]
        if len(srows) == 0:
            continue
        g = d["grids"][srows[-1]]
        choked[side] = bool(g[_DEATH_ROW, _DEATH_COL] != 0)
    c1, c2 = choked["1P"], choked["2P"]
    if c1 is None or c2 is None:
        return None
    if c1 and not c2:
        return "2P"
    if c2 and not c1:
        return "1P"
    return None


def main() -> None:
    lines = ["video\tgame_idx\tnew_missing\tn_snap\tdur_sec\tlock_ratio\t"
             "score_winner_offline\tsurvival_winner_offline\told_present\told_won_1p\tlast_gidx"]
    tot_games = 0
    miss_games = 0
    per_video: dict[str, list[int]] = {}
    miss_score_ok = 0
    miss_score_ng = 0
    miss_surv_ok = 0
    miss_is_lastgame = 0
    npz_files = sorted(NEW_DIR.glob("*.npz"))
    for f in npz_files:
        vid = f.stem
        d = dict(np.load(f, allow_pickle=False))
        if len(d["game_idx"]) == 0:
            continue
        old_f = OLD_DIR / f.name
        od = dict(np.load(old_f, allow_pickle=False)) if old_f.exists() else None
        gidxs = sorted(set(int(g) for g in d["game_idx"]))
        last_g = gidxs[-1]
        for g in gidxs:
            rows = game_rows(d, g)
            won = d["won"][rows]
            missing = bool(np.all(np.isnan(won)))
            partial = (not missing) and bool(np.any(np.isnan(won)))
            if partial:
                print(f"[warn] partial missing: {vid} game {g}")
            tot_games += 1
            lock = d.get("post_match_lockdown_active")
            lock_ratio = float(np.mean(lock[rows] == 1)) if lock is not None else -1.0
            dur = float(d["t_sec"][rows[-1]] - d["t_sec"][rows[0]])
            sw = offline_score_winner(d, rows)
            vw = offline_survival_winner(d, rows)
            old_present = ""
            old_won = ""
            if od is not None:
                orows = np.where(od["game_idx"] == g)[0]
                if len(orows):
                    ow = od["won"][orows]
                    old_present = "all_nan" if np.all(np.isnan(ow)) else "ok"
                    if old_present == "ok":
                        w1 = ow[od["side"][orows] == "1P"]
                        old_won = str(float(w1[-1])) if len(w1) else ""
                else:
                    old_present = "no_game"
            if missing:
                miss_games += 1
                per_video.setdefault(vid, []).append(g)
                if sw is not None:
                    miss_score_ok += 1
                else:
                    miss_score_ng += 1
                    if vw is not None:
                        miss_surv_ok += 1
                if g == last_g:
                    miss_is_lastgame += 1
            lines.append(
                f"{vid}\t{g}\t{int(missing)}\t{len(rows)}\t{dur:.1f}\t"
                f"{lock_ratio:.3f}\t{sw}\t{vw}\t{old_present}\t{old_won}\t{int(g == last_g)}"
            )
    OUT_TSV.parent.mkdir(exist_ok=True)
    OUT_TSV.write_text("\n".join(lines), encoding="utf-8")
    print(f"total games: {tot_games}, missing(won all-NaN): {miss_games} "
          f"({100.0 * miss_games / max(tot_games, 1):.1f}%)")
    print(f"missing games with offline score-winner computable: {miss_score_ok}")
    print(f"missing games score-NG: {miss_score_ng} (survival-computable: {miss_surv_ok})")
    print(f"missing & is-last-game-of-video: {miss_is_lastgame}")
    print("--- per-video missing count (top 20) ---")
    for vid, gs in sorted(per_video.items(), key=lambda kv: -len(kv[1]))[:20]:
        d = dict(np.load(NEW_DIR / f"{vid}.npz", allow_pickle=False))
        n_games = len(set(int(x) for x in d["game_idx"]))
        print(f"{vid}: {len(gs)}/{n_games} missing, games={gs[:12]}{'...' if len(gs) > 12 else ''}")
    print(f"videos with missing: {len(per_video)}/{len(npz_files)}")
    print(f"TSV -> {OUT_TSV}")


if __name__ == "__main__":
    main()
