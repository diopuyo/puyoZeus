"""final4 デモの自己検収 (2026-08-15、read-only)。

feedback_selfverify_before_user_review: userレビュー依頼の前に、指定シーン
全時間帯の数値突合を済ませる (1フレームのスポット確認は禁止)。

検収項目:
  1. 指摘14の窓 (絶対194.53-201秒) で 2P 19% の退行が消えているか
  2. 全域の極端値・凍結が異常でないか (デグレ検査)
  3. れんさ数表示の材料 (chain_event) が実際に流れているか

フラグ構成は scripts/_gen_demo_final4_2026-08-15.sh と完全同一。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_final4_2026-08-15/_unused_selfverify.mp4")
DUMP = Path("data/verify/demo_final4_2026-08-15/selfverify_final4_dump.npz")
ISSUE14_LO, ISSUE14_HI = 193.0, 203.0

history: list[tuple[float, float]] = []
ov.generate(
    VIDEO, OUT, max_sec=0.0, sample_interval=0.0,
    start_sec=162.0, end_sec=310.0,
    show_recognition=True,
    enable_early_fire_reaction=True, enable_per_side_settled=True,
    disable_score_lead_bias=True, disable_pressure=True,
    enable_counter_remaining_time=True, enable_counter_defender_only=True,
    enable_ojama_fall_placement_override=True,
    enable_resolved_exchange_eval=True,
    enable_resolved_decisive_amplify=True,
    enable_pseudo_chain_score_fill=True,
    enable_resolved_live_defender=True,
    enable_resolved_live_defender_strict=True,
    enable_resolved_kill_override=True,
    show_chain_count=True,
    layout="panel",
    render=False,
    debug_history_out=history,
    dump_timeline_path=DUMP,
)

print(f"n_history_samples={len(history)}")
print(f"\n===== 指摘14窓 ({ISSUE14_LO}-{ISSUE14_HI}秒): 表示値 (0.5秒刻み) =====")
print("t_abs   1P%    2P%")
last = None
for t_sec, disp_adv in history:
    if not (ISSUE14_LO <= t_sec <= ISSUE14_HI):
        continue
    b = round(t_sec * 2)
    if b == last:
        continue
    last = b
    p1 = ov.adv_to_winprob(disp_adv) * 100.0
    print(f"{t_sec:6.2f} {p1:5.1f}% {100.0 - p1:5.1f}%")
print("SELFVERIFY_FINAL4_DONE")
