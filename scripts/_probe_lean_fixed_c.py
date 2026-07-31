"""boards_lean_fixed の c1/c4/c34/c82 npz を読み、game_idx 別の t_sec 範囲と
won ラベル分布を確認する (--max-sec 1200 打ち切りの影響を数値で見るための検証専用スクリプト)。

読み取り専用。既存 npz / labeled_win.csv には一切書き込まない。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TARGET_VIDEOS: list[str] = ["c1", "c4", "c34", "c82"]
NPZ_DIR = Path("data/indicators_v2/boards_lean_fixed")

MAX_SEC_CUTOFF: float = 1200.0


def main() -> None:
    for vid in TARGET_VIDEOS:
        path = NPZ_DIR / f"{vid}.npz"
        if not path.exists():
            print(f"[{vid}] npz なし: {path}")
            continue
        data = np.load(str(path), allow_pickle=True)
        t_sec = data["t_sec"]
        game_idx = data["game_idx"]
        won = data["won"]
        side = data["side"]

        print(f"=== video_{vid} ===")
        print(f"  総行数: {len(t_sec)}  t_sec範囲: {t_sec.min():.1f} - {t_sec.max():.1f}")
        n_games = len(np.unique(game_idx))
        print(f"  game_idx種類数: {n_games}")

        for gidx in sorted(np.unique(game_idx)):
            mask = game_idx == gidx
            t_g = t_sec[mask]
            won_g = won[mask]
            side_g = side[mask]
            t_max = t_g.max()
            near_cutoff = t_max >= MAX_SEC_CUTOFF - 5.0
            # 1P/2P 側それぞれ最終行の won
            def last_won(s: str) -> float:
                m2 = mask & (side == s)
                if m2.sum() == 0:
                    return float("nan")
                idx_last = np.where(m2)[0][-1]
                return float(won[idx_last])
            print(
                f"    game_idx={gidx}: n={mask.sum():>5}  t_sec=[{t_g.min():.1f},{t_max:.1f}]"
                f"  cutoff近接={'YES' if near_cutoff else 'no'}"
                f"  won(1P最終)={last_won('1P')}  won(2P最終)={last_won('2P')}"
            )
        print()


if __name__ == "__main__":
    main()
