"""boards_lean_fixed 内蔵 won (--max-sec 1200 打ち切り収集) と、WIN★パネル方式
(extract_match_winners.py, 動画フル長) の勝者判定を game 単位で突合する検証専用スクリプト。

読み取り専用。既存 npz / labeled_win.csv には一切書き込まない。

判定手順:
  1. boards_lean_fixed/{vid}.npz の各 game_idx について、その t_sec 範囲の
     中心時刻を計算し、winners JSON の試合区間 [start_sec, end_sec) にマップする。
  2. npz 側の勝者 (1P 視点 won の最終行から復元) と winners JSON 側の winner を比較。
  3. 一致率、および「npz側 t_sec_max が 1200 秒付近(打ち切り近接)のゲームで
     不一致が集中するか」を集計する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.label_win_from_winners import load_winners, find_winner_for_t

NPZ_DIR = Path("data/indicators_v2/boards_lean_fixed")
WINNERS_DIR = Path("data/verify/winners_probe_2026-07-23")
TARGET_VIDEOS: list[str] = ["c1", "c4", "c34", "c82"]

MAX_SEC_CUTOFF: float = 1200.0
CUTOFF_MARGIN: float = 5.0  # 打ち切り近接とみなす秒数マージン


def npz_winner_for_game(t_sec: np.ndarray, side: np.ndarray, won: np.ndarray, game_idx: np.ndarray, gidx: int) -> str | None:
    """npz の game_idx から 1P 視点 won の最終行を使って winner ("1P"/"2P") を復元する。"""
    mask = game_idx == gidx
    if mask.sum() == 0:
        return None
    # 1P 側の最終行を優先。無ければ 2P 側最終行から反転して求める。
    mask_1p = mask & (side == "1P")
    if mask_1p.sum() > 0:
        idx_last = np.where(mask_1p)[0][-1]
        w = won[idx_last]
        if np.isnan(w):
            return None
        return "1P" if w >= 0.5 else "2P"
    mask_2p = mask & (side == "2P")
    if mask_2p.sum() > 0:
        idx_last = np.where(mask_2p)[0][-1]
        w = won[idx_last]
        if np.isnan(w):
            return None
        return "2P" if w >= 0.5 else "1P"
    return None


def main() -> None:
    video_ids = [f"video_{v}" for v in TARGET_VIDEOS]
    winners_map = load_winners(WINNERS_DIR, video_ids)

    total_compared = 0
    total_match = 0
    total_mismatch_cutoff = 0
    total_cutoff = 0

    for vid in TARGET_VIDEOS:
        npz_path = NPZ_DIR / f"{vid}.npz"
        if not npz_path.exists():
            print(f"[{vid}] npz なし、skip")
            continue
        data = np.load(str(npz_path), allow_pickle=True)
        t_sec = data["t_sec"]
        side = data["side"]
        won = data["won"]
        game_idx = data["game_idx"]

        games_wp = winners_map.get(f"video_{vid}", [])
        print(f"=== video_{vid} ===  (winners JSON 試合数: {len(games_wp)})")

        for gidx in sorted(np.unique(game_idx)):
            mask = game_idx == gidx
            t_g = t_sec[mask]
            t_min, t_max = float(t_g.min()), float(t_g.max())
            t_mid = (t_min + t_max) / 2.0
            is_cutoff = t_max >= MAX_SEC_CUTOFF - CUTOFF_MARGIN

            npz_winner = npz_winner_for_game(t_sec, side, won, game_idx, gidx)
            wp_winner = find_winner_for_t(games_wp, t_mid)

            status: str
            if npz_winner is None or wp_winner is None:
                status = "N/A(片方None)"
            else:
                total_compared += 1
                if is_cutoff:
                    total_cutoff += 1
                if npz_winner == wp_winner:
                    total_match += 1
                    status = "MATCH"
                else:
                    status = "MISMATCH"
                    if is_cutoff:
                        total_mismatch_cutoff += 1

            print(
                f"  game_idx={gidx:>2}  t_sec=[{t_min:>7.1f},{t_max:>7.1f}]"
                f"  cutoff近接={'YES' if is_cutoff else 'no ':<3}"
                f"  npz_won={str(npz_winner):<6}  win_panel={str(wp_winner):<6}  {status}"
            )
        print()

    print("=== 総合 ===")
    if total_compared > 0:
        print(f"  比較可能ゲーム数: {total_compared}")
        print(f"  一致: {total_match}  不一致: {total_compared - total_match}  "
              f"一致率: {total_match/total_compared:.1%}")
        n_mismatch = total_compared - total_match
        if n_mismatch > 0:
            print(f"  不一致のうち打ち切り近接ゲームの割合: "
                  f"{total_mismatch_cutoff}/{n_mismatch} = {total_mismatch_cutoff/n_mismatch:.1%}")
        print(f"  打ち切り近接ゲーム総数(比較対象内): {total_cutoff}")
    else:
        print("  比較可能ゲームなし")


if __name__ == "__main__":
    main()
