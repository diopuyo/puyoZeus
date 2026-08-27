# -*- coding: utf-8 -*-
"""
148動画再収集(2026-08-18, 8フラグ全群ON)の品質診断。
既存npzの読み取り専用。src/配下・collect_boards_lean.pyは一切変更しない。
新旧ディレクトリを突き合わせ、行数/列/値域/フラグ列を比較する。
"""
import numpy as np
from pathlib import Path

OLD_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
NEW_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-18")

def load(path):
    return dict(np.load(path, allow_pickle=True))

def summarize(video_id):
    old_p = OLD_DIR / f"{video_id}.npz"
    new_p = NEW_DIR / f"{video_id}.npz"
    if not old_p.exists() or not new_p.exists():
        print(f"{video_id}: SKIP (missing old={old_p.exists()} new={new_p.exists()})")
        return None
    old = load(old_p)
    new = load(new_p)
    n_old = len(old["grids"])
    n_new = len(new["grids"])
    ratio = n_new / n_old if n_old else float("nan")

    # 値域チェック (新データ)
    grids = new["grids"]
    nan_inf = 0
    for k, v in new.items():
        if v.dtype.kind in "fc":
            bad = (~np.isfinite(v)).sum()
            nan_inf += bad
    color_min, color_max = grids.min(), grids.max()

    # おじゃま列
    ojama_neg = 0
    if "ojama_net_balance" in new:
        ojama_neg = (new["ojama_net_balance"] < 0).sum()

    # 新設フラグ列の非ゼロ率
    flag_stats = {}
    for col in ["match_end_locked", "post_match_lockdown_active", "all_clear_pending", "ojama_forecast", "tsumo_count"]:
        if col in new:
            v = new[col]
            nz = int((v != 0).sum())
            flag_stats[col] = f"{nz}/{len(v)} ({100*nz/len(v):.1f}%)"
        else:
            flag_stats[col] = "MISSING"

    # side別行数
    side_counts_new = {s: int((new["side"] == s).sum()) for s in np.unique(new["side"])}
    side_counts_old = {s: int((old["side"] == s).sum()) for s in np.unique(old["side"])}

    print(f"\n=== {video_id} ===")
    print(f"  rows: old={n_old} new={n_new} ratio={ratio:.3f}")
    print(f"  side_counts old={side_counts_old} new={side_counts_new}")
    print(f"  color range new=[{color_min},{color_max}] nan/inf={nan_inf}")
    print(f"  ojama_net_balance negative count={ojama_neg}")
    print(f"  flags: {flag_stats}")
    return dict(video_id=video_id, n_old=n_old, n_new=n_new, ratio=ratio)

if __name__ == "__main__":
    new_files = sorted(p.stem for p in NEW_DIR.glob("*.npz"))
    results = []
    for vid in new_files:
        r = summarize(vid)
        if r:
            results.append(r)
    print("\n=== まとめ ===")
    for r in results:
        print(r)
    ratios = [r["ratio"] for r in results]
    print(f"\n行数比率(new/old): min={min(ratios):.3f} max={max(ratios):.3f} mean={sum(ratios)/len(ratios):.3f}")
