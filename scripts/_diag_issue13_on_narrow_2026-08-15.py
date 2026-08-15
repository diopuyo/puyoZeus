"""指摘13修正 (着弾済み仮想盤面) の短縮検証 — ON側のみ・162-250秒窓。

シャットダウン前の時間制約用: OFF側の値は取得済み (logs/_diag_issue13_verify_2026-08-15.log、
t=234.9で disp_adv=-33.55 固定) のため再実行しない。勝率換算は表示と同じ
adv_to_winprob (S字較正) を使う (線形近似 0.5+adv/200 は誤り、2026-08-15検収の教訓)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

START_SEC = 162.0
END_SEC = 250.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue13_narrow.mp4")
DUMP = Path("data/verify/demo_fixed_2026-08-13/diag_issue13_on_narrow_2026-08-15.npz")

ISSUE12_LO, ISSUE12_HI = 234.87, 245.5
ISSUE10_LO, ISSUE10_HI = 194.53, 200.0
BUCKET_SEC = 0.5


def main() -> int:
    history: list[tuple[float, float]] = []
    ov.generate(
        VIDEO, OUT, max_sec=0.0, sample_interval=0.0,
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
        enable_resolved_live_defender=True,
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
        debug_history_out=history,
        dump_timeline_path=DUMP,
    )
    for label, lo, hi in (("指摘12窓 (234.87-245.5s)", ISSUE12_LO, ISSUE12_HI),
                          ("指摘10窓 (194.53-200s)", ISSUE10_LO, ISSUE10_HI)):
        print(f"\n---- ON修正版 {label} ----")
        last_bucket = None
        for t_sec, disp_adv in history:
            if not (lo <= t_sec <= hi):
                continue
            bucket = round(t_sec / BUCKET_SEC)
            if bucket == last_bucket:
                continue
            last_bucket = bucket
            p1 = ov.adv_to_winprob(disp_adv)
            print(f"  t={t_sec:7.2f} disp_adv={disp_adv:+7.2f} "
                  f"p1(1P)={p1 * 100:5.1f}% p2(2P)={100 - p1 * 100:5.1f}%")
    print("\nDONE_ISSUE13_NARROW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
