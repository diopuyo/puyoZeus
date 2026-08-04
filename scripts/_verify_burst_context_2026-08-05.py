"""userレビュー裏取り: 未説明5シーンのonset直前に相手スコア急増があるか (2026-08-05、使い捨て)。

userの目視分類=全シーン「相手のお邪魔送付バースト演出」。正しければ onset 直前に
相手の連鎖発火 (スコア急増) が存在するはず。診断の no_time_window 判定は
chain_trigger_sec (ChainEventベース) の欠落・不完全性による見かけだった、を検証する。
"""
from pathlib import Path

import numpy as np

ANCHOR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
# (video, 誤りside, onset_t_sec)
SCENES: list[tuple[str, str, float]] = [
    ("c18", "2P", 845.77),
    ("c13", "2P", 2524.23),
    ("c29", "2P", 611.40),
    ("c36", "2P", 168.03),
    ("c29", "2P", 2405.83),
]
LOOKBACK_SEC: float = 12.0
LOOKAHEAD_SEC: float = 2.0


def main() -> None:
    for stem, err_side, onset in SCENES:
        z = np.load(ANCHOR / f"{stem}.npz", allow_pickle=True)
        opp = "1P" if err_side == "2P" else "2P"
        mask = (z["side"] == opp)
        t = z["t_sec"][mask].astype(float)
        s = z["score"][mask].astype(float)
        win = (t >= onset - LOOKBACK_SEC) & (t <= onset + LOOKAHEAD_SEC)
        tw, sw = t[win], s[win]
        if len(sw) < 2:
            print(f"{stem} onset={onset}: 相手スコアサンプル不足")
            continue
        valid = sw >= 0
        deltas = np.diff(sw[valid])
        total_up = float(deltas[deltas > 0].sum()) if len(deltas) else 0.0
        # 相手側の chain_trigger (診断が使った窓の元データ) がこの近傍にあるか
        ct = z["chain_trigger_sec"][mask].astype(float)
        ct = ct[np.isfinite(ct) & (ct > 0)]
        near_ct = ct[(ct >= onset - LOOKBACK_SEC) & (ct <= onset + LOOKAHEAD_SEC)]
        print(f"{stem} onset={onset:8.1f}: 相手スコア増分(直前{LOOKBACK_SEC:.0f}秒)={total_up:8.0f} "
              f"(お邪魔換算 約{total_up / 70:.0f}個) / ChainEvent検出={len(np.unique(near_ct))}件")


if __name__ == "__main__":
    main()
