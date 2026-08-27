# 診断: 試合開始+1〜3秒窓の「非空かつ完全不変」窓の列挙 (2026-08-19)
# 品質ゲート (scripts/phase_l_video_quality_gate.py) の検査1と同一の窓定義で
# subset50 npz 全体から (video, game_idx, side) 窓を列挙し、
#   - 窓内に非空セルがあるか
#   - 窓内の grid が全レコードで完全一致か (不変)
# を分類して TSV に書き出す。本体コードは変更しない。
from __future__ import annotations

import glob
import os

import numpy as np

BASE = "data/indicators_v2/boards_lean_subset50_2026-08-19"
OUT = "logs/_diag_startwindow_invariant_cases_2026-08-19.tsv"

OFFSET_LO = 1.0
OFFSET_HI = 3.0
ROW_LO = 1
ROW_HI = 10
ANCHOR_SCORE_MAX = 50
SCORE_UNREADABLE = -1


def main() -> None:
    rows = []
    n_windows = 0
    n_nonempty = 0
    n_invariant = 0
    n_invariant_le4 = 0
    for path in sorted(glob.glob(os.path.join(BASE, "*.npz"))):
        vid = os.path.basename(path)[:-4]
        d = np.load(path, allow_pickle=True)
        grids = d["grids"]
        t_sec = d["t_sec"]
        game_idx = d["game_idx"]
        side = d["side"]
        score = d["score"] if "score" in d else np.array([])
        for g in np.unique(game_idx):
            gm = game_idx == g
            start_sec = float(t_sec[gm].min())
            lo, hi = start_sec + OFFSET_LO, start_sec + OFFSET_HI
            for s in ("1P", "2P"):
                m = gm & (side == s) & (t_sec >= lo) & (t_sec <= hi)
                if not m.any():
                    continue
                # anchor score 除外 (ゲートと同一)
                sgm = gm & (side == s)
                if score.size:
                    tt = t_sec[sgm]
                    a = int(score[sgm][int(np.argmin(np.abs(tt - start_sec)))])
                    if a != SCORE_UNREADABLE and a > ANCHOR_SCORE_MAX:
                        continue
                n_windows += 1
                sub = grids[m][:, ROW_LO:ROW_HI, :]
                nonempty = int((sub != 0).sum())
                if nonempty == 0:
                    continue
                n_nonempty += 1
                invariant = bool((sub == sub[0]).all())
                # 窓内最初の grid の非空セル (row1-9)
                g0 = grids[m][0]
                cells = [
                    f"r{r}c{c}={int(g0[r, c])}"
                    for r, c in np.argwhere(g0[ROW_LO:ROW_HI] != 0) + [ROW_LO, 0]
                ]
                n_cells0 = int((g0[ROW_LO:ROW_HI] != 0).sum())
                if invariant:
                    n_invariant += 1
                    if n_cells0 <= 4:
                        n_invariant_le4 += 1
                tw = t_sec[m]
                fw = d["frame_idx"][m]
                rows.append(
                    "\t".join(
                        [
                            vid,
                            str(int(g)),
                            s,
                            f"{start_sec:.2f}",
                            f"{tw.min():.2f}",
                            f"{tw.max():.2f}",
                            str(int(fw.min())),
                            str(int(fw.max())),
                            str(int(m.sum())),
                            str(n_cells0),
                            "INVARIANT" if invariant else "changing",
                            ";".join(cells[:12]),
                        ]
                    )
                )
    header = (
        "video\tgame_idx\tside\tstart_sec\twin_t_lo\twin_t_hi\twin_f_lo\t"
        "win_f_hi\tn_records\tn_cells_first\tinvariance\tcells_first"
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
    print(f"windows={n_windows} nonempty={n_nonempty} invariant={n_invariant} invariant_le4cells={n_invariant_le4}")
    print(f"out={OUT}")


if __name__ == "__main__":
    main()
