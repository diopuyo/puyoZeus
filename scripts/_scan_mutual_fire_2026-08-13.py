"""review_demo 内で「両側 chain_event が同時に CHAIN_TOTAL_MIN_SCORE 以上」に
なる真の両者発火区間を探す診断スクリプト (使い捨て、コミット対象外)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from scripts.visualize_advantage_overlay import generate  # noqa: E402

_orig_update = vao.ResolvedExchangeTracker.update
_events: list[tuple[float, float, float]] = []


def _traced_update(self, r_p1, r_p2, snap, elapsed_sec):
    was_active = self._active
    active, just_deactivated = _orig_update(self, r_p1, r_p2, snap, elapsed_sec)
    if not was_active and active:
        _events.append(("activate", self.hold_adv, self._pred_score1))
    return active, just_deactivated


vao.ResolvedExchangeTracker.update = _traced_update

# generate() 自体は t を外に出さないため、update() 呼び出し時刻を別途記録する。
_orig_resolve = vao.ResolvedExchangeTracker._resolve
_t_ref = {"t": 0.0}


def _traced_resolve(self, snap, elapsed_sec, score1, score2):
    _orig_resolve(self, snap, elapsed_sec, score1, score2)
    print(f"[resolve] elapsed={elapsed_sec:.1f}s score1={score1:.0f} score2={score2:.0f} "
          f"adv={self.hold_adv:.1f}")


vao.ResolvedExchangeTracker._resolve = _traced_resolve

BASE_KWARGS = dict(
    enable_early_fire_reaction=True,
    enable_per_side_settled=True,
    disable_score_lead_bias=True,
    disable_pressure=True,
    enable_counter_remaining_time=True,
    enable_counter_defender_only=True,
    stable_majority_window=True,
    enable_ojama_fall_placement_override=True,
    enable_ojama_fall_entry_hardening=True,
    enable_ojama_fall_scoped_exit=True,
    enable_resolved_exchange_eval=True,
)

generate(
    Path("data/frames/review_demo_2026-08-12.mp4"),
    Path("data/verify/_unused_scan.mp4"),
    max_sec=0.0, sample_interval=0.0,
    start_sec=162.0, end_sec=310.0, warmup_sec=10.0,
    render=False,
    **BASE_KWARGS,
)
