"""検収セルフベリファイ (demo_fixed_3match.mp4 = 確認デモ集大成版):
指摘#10/#11の対象区間 (デモt=28-48s = source t=190-210s) を --no-render で
再計算し、debug_history_out (実際の表示値 disp_adv の全サンプル) を
0.5秒刻みに整形して出力する (read-only、動画は書き出さない)。

demo_fixed_3match.mp4 と完全に同一の生成条件
(scripts/_gen_demo_fixed_2026-08-13.sh / logs/demo_confirm_2026-08-14.log の
[cmd] 行と同一フラグ) で再計算するので、動画の実表示値と一致するはず。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visualize_advantage_overlay import generate

START_SEC = 162.0  # デモ t=0 (source絶対時刻)
WINDOW_LO = 190.0  # デモ t=28s
WINDOW_HI = 210.0  # デモ t=48s


def main() -> int:
    video = Path("data/frames/review_demo_2026-08-12.mp4")
    out = Path("data/verify/demo_fixed_2026-08-13/_unused_selfverify_confirm.mp4")
    history: list[tuple[float, float]] = []
    generate(
        video, out, max_sec=0.0, sample_interval=0.0,
        start_sec=START_SEC, end_sec=210.0,
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
    for t_sec, disp_adv in history:
        if t_sec < WINDOW_LO or t_sec > WINDOW_HI:
            continue
        t_out = t_sec - START_SEC
        bucket = round(t_out * 2) / 2.0  # 0.5秒刻み
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        p1 = 0.5 + disp_adv / 200.0
        print(f"t_out={t_out:6.2f}s (src={t_sec:7.2f}) disp_adv={disp_adv:+7.2f} p1={p1*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
