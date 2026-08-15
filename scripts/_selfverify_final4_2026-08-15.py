"""final4 デモの自己検収 (2026-08-15、read-only)。

feedback_selfverify_before_user_review: userレビュー依頼の前に、指定シーン
全時間帯の数値突合を済ませる (1フレームのスポット確認は禁止)。

検収項目:
  1. 指摘14の窓 (絶対194.53-201秒) で 2P 19% の退行が消えているか
  2. 全域で極端値・凍結が異常に増えていないか (デグレ検査)
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

ov.generate(
    video_path=VIDEO,
    out_path=OUT,
    start_sec=162.0,
    end_sec=310.0,
    render=False,
    layout="panel",
    show_recognition=True,
    sample_interval=0,
    early_fire_reaction=True,
    per_side_settled=True,
    no_score_lead_bias=True,
    no_pressure=True,
    enable_counter_remaining_time=True,
    enable_counter_defender_only=True,
    enable_ojama_fall_placement_override=True,
    enable_resolved_exchange_eval=True,
    enable_resolved_decisive_amplify=True,
    enable_pseudo_chain_score_fill=True,
    enable_resolved_live_defender=True,
    enable_resolved_live_defender_strict=True,
    enable_resolved_kill_override=True,
    show_chain_count=True,
    dump_timeline_path=DUMP,
)
print("SELFVERIFY_FINAL4_DUMP_DONE")
