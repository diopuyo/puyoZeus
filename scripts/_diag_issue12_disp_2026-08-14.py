"""指摘12 診断 (1/2): 2試合目 (デモ56-116s = source 218-278s) の disp_adv/p1 を
0.25秒刻みでダンプし、2Pが97%超になる瞬間を特定する。

demo_fixed_3match.mp4 と完全同一の生成条件 (scripts/_gen_demo_fixed_2026-08-13.sh)
で generate(render=False, debug_history_out=...) を実行する
(scripts/_selfverify_demo_confirm_dump_2026-08-14.py と同じ確立手法)。

前面実行のみ・バックグラウンド分離禁止 (規約)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visualize_advantage_overlay import generate

START_SEC = 162.0   # デモ t=0 (source絶対時刻)
WINDOW_LO = 218.0   # デモ t=56s (2試合目開始付近)
WINDOW_HI = 278.0   # デモ t=116s (2試合目終了付近)


def main() -> int:
    video = Path("data/frames/review_demo_2026-08-12.mp4")
    out = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue12.mp4")
    history: list[tuple[float, float]] = []
    generate(
        video, out, max_sec=0.0, sample_interval=0.0,
        start_sec=START_SEC, end_sec=WINDOW_HI + 2.0,
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
        debug_history_out=history,
    )
    print(f"n_samples={len(history)}")
    last_bucket = None
    worst_t, worst_adv = None, 0.0
    for t_sec, disp_adv in history:
        if t_sec < WINDOW_LO or t_sec > WINDOW_HI:
            continue
        t_out = t_sec - START_SEC
        bucket = round(t_out * 4) / 4.0  # 0.25秒刻み
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        p1 = 0.5 + disp_adv / 200.0
        print(f"t_out={t_out:6.2f}s (src={t_sec:7.2f}) disp_adv={disp_adv:+7.2f} "
              f"p1={p1*100:5.1f}% p2={100-p1*100:5.1f}%")
        if disp_adv < worst_adv:
            worst_adv, worst_t = disp_adv, t_sec
    print(f"\n[worst for 1P] src_t={worst_t} disp_adv={worst_adv:+.2f} "
          f"p2={100*(0.5 - worst_adv/200.0):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
