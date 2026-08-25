"""旧CSVを「新方式が実際に記録した時刻」だけに間引く (2026-08-19)。

目的: 公平条件 (locked窓転写) でも新方式が AUC -0.031 劣後する原因を
「サンプリング (どの時刻の盤面を持っているか)」と「盤面内容の質」に分離する。

作り方: 新CSVの学習到達行 (won非欠損) の (video, side, t_sec) を基準に、
旧locked窓転写後CSVから各基準時刻の最近傍1行 (|Δt|<=TOL) だけを残す。
- AUCが旧のまま (~0.659) → サンプリングは無罪、新盤面の内容が悪い
- AUCが新に近づく (~0.628) → 時刻被覆の偏りが主因

出力: data/verify/retrain_subset42_2026-08-19/old_timematched/labeled_win_old_timematched.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NEW_CSV = ROOT / "data/verify/labeled_win_subset42_2026-08-19/labeled.csv"
OLD_FILT_CSV = ROOT / "data/verify/retrain_subset42_2026-08-19/old_lockedfilt/labeled_win_old_lockedfilt.csv"
OUT = ROOT / "data/verify/retrain_subset42_2026-08-19/old_timematched/labeled_win_old_timematched.csv"

TOL_SEC: float = 0.7  # 新は1手≈1-2秒間隔、旧は密 → 最近傍1行で十分


def main() -> int:
    new = pd.read_csv(NEW_CSV, usecols=["video_id", "side", "t_sec", "won"])
    new = new.dropna(subset=["won"])
    old = pd.read_csv(OLD_FILT_CSV)

    keep = np.zeros(len(old), dtype=bool)
    n_ref_total, n_ref_matched = 0, 0
    for (vid, side), g_new in new.groupby(["video_id", "side"]):
        m = ((old["video_id"] == vid) & (old["side"] == side)).to_numpy()
        idx_old = np.where(m)[0]
        if len(idx_old) == 0:
            continue
        t_old = old.loc[idx_old, "t_sec"].to_numpy(dtype=float)
        order = np.argsort(t_old)
        t_sorted = t_old[order]
        refs = np.sort(g_new["t_sec"].to_numpy(dtype=float))
        n_ref_total += len(refs)
        pos = np.searchsorted(t_sorted, refs)
        for r, p in zip(refs, pos):
            best, bestd = -1, TOL_SEC + 1
            for q in (p - 1, p):
                if 0 <= q < len(t_sorted):
                    d = abs(t_sorted[q] - r)
                    if d < bestd:
                        bestd, best = d, q
            if best >= 0 and bestd <= TOL_SEC:
                keep[idx_old[order[best]]] = True
                n_ref_matched += 1

    out = old[keep]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"新基準時刻 {n_ref_total:,}件中 旧側に最近傍あり {n_ref_matched:,} ({n_ref_matched/n_ref_total:.1%})")
    print(f"旧locked窓転写後 {len(old):,}行 -> 時刻マッチ {len(out):,}行 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
