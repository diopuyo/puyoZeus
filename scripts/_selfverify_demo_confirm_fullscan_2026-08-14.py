"""検収セルフベリファイ (demo_fixed_3match.mp4 = 確認デモ集大成版): 全域スキャン。

demo_fixed_3match.mp4 と完全同一条件 (--dump-timeline 相当) で全区間
(source 162-310s = デモ全域) を --no-render で再計算し、
- 極端値 (p1>=97% or p1<=3%) の出現頻度
- 主因(top1)列名の分布 (新列名=diff系が自然に出ているか)
を集計する (read-only)。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visualize_advantage_overlay import generate, save_timeline_dump

START_SEC = 162.0
END_SEC = 310.0
DUMP_PATH = Path("data/verify/demo_fixed_2026-08-13/demo_confirm_fullscan_dump.npz")


def main() -> int:
    video = Path("data/frames/review_demo_2026-08-12.mp4")
    out = Path("data/verify/demo_fixed_2026-08-13/_unused_selfverify_fullscan.mp4")
    generate(
        video, out, max_sec=0.0, sample_interval=0.0,
        start_sec=START_SEC, end_sec=END_SEC,
        show_recognition=True,
        enable_early_fire_reaction=True, enable_per_side_settled=True,
        disable_score_lead_bias=True, disable_pressure=True,
        enable_counter_remaining_time=True, enable_counter_defender_only=True,
        stable_majority_window=True,
        enable_ojama_fall_placement_override=True,
        enable_ojama_fall_entry_hardening=True,
        enable_ojama_fall_scoped_exit=True,
        enable_resolved_exchange_eval=True,
        enable_resolved_decisive_amplify=True,
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
        dump_timeline_path=DUMP_PATH,
    )
    from scripts.visualize_advantage_overlay import load_timeline_dump
    video_id, rows = load_timeline_dump(DUMP_PATH)
    print(f"video_id={video_id} n_settled_updates={len(rows)}")
    n_extreme_p1 = sum(1 for r in rows if r.p1 >= 0.97)
    n_extreme_p2 = sum(1 for r in rows if r.p1 <= 0.03)
    n_total = len(rows)
    print(f"極端値(p1>=97%): {n_extreme_p1}/{n_total} ({n_extreme_p1/n_total*100:.1f}%)")
    print(f"極端値(p1<=3% ,2P側): {n_extreme_p2}/{n_total} ({n_extreme_p2/n_total*100:.1f}%)")
    driver_counter = Counter(r.drivers_top1_name for r in rows)
    print("主因(top1) 列名分布 (上位20):")
    for name, cnt in driver_counter.most_common(20):
        print(f"  {name:30s} {cnt:5d} ({cnt/n_total*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
