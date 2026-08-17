"""指摘19 状態ゲート版: t=200.5-204.0 の r.p1.state/r.p2.state を直接計測する
(read-only、臆測せず実測で確認)。

hold_after_kill_override 呼び出し直前の state1/state2 (= r.p1.state/
r.p2.state、generate() が実際に渡す値) を全てダンプする。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
LO, HI = 200.5, 204.0

_orig = ov.ResolvedExchangeTracker.hold_after_kill_override


def _traced(self, b1, b2, state1=None, state2=None):
    t = self._t_sec
    adv, p1 = _orig(self, b1, b2, state1=state1, state2=state2)
    if LO <= t <= HI:
        room1, room2 = ov.board_room(b1), ov.board_room(b2)
        print(f"t={t:7.2f} state1={state1} state2={state2} "
              f"inc1={self._incoming_total_p1:.1f} inc2={self._incoming_total_p2:.1f} "
              f"room1={room1} room2={room2} hold_adv={self.hold_adv:7.2f} "
              f"final_adv={adv:7.2f} p1={p1 * 100:.1f}%")
    return adv, p1


ov.ResolvedExchangeTracker.hold_after_kill_override = _traced

hist: list[tuple[float, float]] = []
ov.generate(
    VIDEO, Path("data/verify/_unused_issue19_state_trace.mp4"),
    max_sec=0.0, sample_interval=0.0, start_sec=162.0, end_sec=207.0,
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
    enable_resolved_kill_override_counter_aware=True,
    layout="panel", render=False,
    debug_history_out=hist,
)
print("DIAG_ISSUE19_STATE_TRACE_DONE")
