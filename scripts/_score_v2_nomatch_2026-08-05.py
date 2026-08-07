"""v2 no_match盤面の直接採点 (2026-08-05、使い捨て)。

Stage1.5ガードが認識内容を変えた盤面は frame_idx 一致が失敗し no_match に落ちる
(=効果が出た盤面ほど集計から抜ける)。ラベル時点の実効盤面 = 「その時刻以前の
最新スナップショット」で correct_grid と直接採点し、OFF の誤り数と比較する。
時刻窓は [anchor_t - LOOKBACK_SEC, anchor_t + EPS] (盤面はスナップショット間で
持続するという実効意味論、bit一致に依存しない明示的な近似)。
"""
import csv
from pathlib import Path

import numpy as np

ANCHOR_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
V2_DIR = Path("data/verify/burst_guard_2026-08-05/on_v2_full")
LOOKBACK_SEC: float = 15.0
EPS_SEC: float = 0.05

LABELS = [
    ("data/verify/full_board_label_sheet_2026-08-02/labeling_result.csv",
     "data/verify/full_board_label_sheet_2026-08-02/labeling_sheet.csv"),
    ("data/verify/full_board_label_sheet_batch2_2026-08-03/labeling_result.csv",
     "data/verify/full_board_label_sheet_batch2_2026-08-03/labeling_sheet.csv"),
]


def _err_count(grid: "np.ndarray", correct_rows: "list[str]") -> "tuple[int, int]":
    """(全体誤り, burst層row1-3誤り) を返す。'U' セルは分母除外。"""
    total = burst = 0
    for r in range(13):
        for c in range(6):
            cv = correct_rows[r][c]
            if cv == "U":
                continue
            v = int(grid[r][c])
            if str(v) != cv:
                total += 1
                if r in (1, 2, 3):
                    burst += 1
    return total, burst


def main() -> None:
    sheets: dict = {}
    results: dict = {}
    for res_csv, sheet_csv in LABELS:
        for r in csv.DictReader(open(sheet_csv, encoding="utf-8-sig")):
            sheets[(r["video_id"], r["side"], r["t_sec"])] = r
        for r in csv.DictReader(open(res_csv, encoding="utf-8-sig")):
            if r["status"] == "fixed":
                results[(r["video_id"], r["side"], r["t_sec"])] = r["correct_grid"]

    grand_off = grand_v2 = 0
    for (vid, side, t), correct in sorted(results.items()):
        stem = vid.replace("video_", "")
        v2_path = V2_DIR / f"{stem}.npz"
        if not v2_path.exists():
            continue
        anchor_t = float(t)
        z = np.load(v2_path, allow_pickle=True)
        mask = (z["side"] == side) & (z["t_sec"] <= anchor_t + EPS_SEC) \
            & (z["t_sec"] >= anchor_t - LOOKBACK_SEC)
        idxs = np.where(mask)[0]
        correct_rows = correct.split("/")
        a = np.load(ANCHOR_DIR / f"{stem}.npz", allow_pickle=True)
        am = (a["side"] == side) & (np.abs(a["t_sec"].astype(float) - anchor_t) < 0.5)
        ai = np.where(am)[0]
        off_err = off_burst = None
        if len(ai):
            off_err, off_burst = _err_count(a["grids"][ai[-1]], correct_rows)
        if len(idxs) == 0:
            print(f"{stem} {side} t={t}: v2に時刻窓内スナップショット無し (OFF={off_err})")
            continue
        v2_err, v2_burst = _err_count(z["grids"][idxs[-1]], correct_rows)
        lag = anchor_t - float(z["t_sec"][idxs[-1]])
        print(f"{stem} {side} t={t}: OFF={off_err} (burst層{off_burst}) -> v2={v2_err} "
              f"(burst層{v2_burst})  [v2スナップ遅れ {lag:.2f}s]")
        if off_err is not None:
            grand_off += off_err
            grand_v2 += v2_err
    print(f"\n合計 (fixed盤面・着弾済み分): OFF={grand_off} -> v2={grand_v2} ({grand_v2 - grand_off:+d})")


if __name__ == "__main__":
    main()
