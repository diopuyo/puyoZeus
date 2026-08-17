"""指摘19: C構成 (フレッシュ再計算版) で減衰量が小さい原因を計装で特定する
(read-only)。

`_kill_dampen_counter_prob` 呼び出し時の
(t_sec, victim_side, incoming, budget_sec, room, prob) を全てダンプする。
start_sec は本番A/B/C比較スクリプトと同じ 162.0 (warmup差によるdesync混入を
避けるため)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
LO, HI = 200.5, 204.0

_orig = ov.ResolvedExchangeTracker._kill_dampen_counter_prob


def _traced(self, b1, b2, victim_side, incoming, attacker_event):
    t = self._t_sec
    prob = _orig(self, b1, b2, victim_side, incoming, attacker_event)
    if LO <= t <= HI:
        elapsed = t - attacker_event.trigger_sec
        budget = ov._chain_remaining_time_budget_sec(
            attacker_event.chain_count, attacker_event.trigger_sec, t, self._chain_len_table)
        board = b1 if victim_side == "1P" else b2
        room = ov.board_room(board)
        print(f"t={t:7.2f} victim={victim_side} incoming={incoming:.1f} "
              f"attacker_chain_count={attacker_event.chain_count} "
              f"attacker_trigger_sec={attacker_event.trigger_sec:.2f} elapsed={elapsed:.2f} "
              f"budget={budget:.2f} room={room} prob={prob}")
    return prob


ov.ResolvedExchangeTracker._kill_dampen_counter_prob = _traced

hist: list[tuple[float, float]] = []
ov.generate(
    VIDEO, Path("data/verify/_unused_issue19_trace.mp4"),
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
print("DIAG_ISSUE19_TRACE_DONE")
