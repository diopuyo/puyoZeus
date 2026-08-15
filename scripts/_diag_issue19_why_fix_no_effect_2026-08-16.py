"""指摘19 根治 (--resolved-victim-gen-live) が窓2 (t=201.2-203.4) で無効だった
原因を追跡する (read-only)。_resolve/_maybe_redecide/hold_after_kill_override
の内部値を計装し、なぜ FIX が BASE と bit-identical だったかを特定する。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")

_orig_resolve = ov.ResolvedExchangeTracker._resolve
_orig_maybe_redecide = ov.ResolvedExchangeTracker._maybe_redecide
_orig_kill = ov.ResolvedExchangeTracker.hold_after_kill_override
_log: list[str] = []


def _traced_resolve(self, snap, elapsed_sec, score1, score2):
    r = _orig_resolve(self, snap, elapsed_sec, score1, score2)
    _log.append(
        f"[_resolve] t={elapsed_sec:.2f} score1={score1:.0f} score2={score2:.0f} "
        f"incoming1={self._incoming_total_p1:.1f} incoming2={self._incoming_total_p2:.1f} "
        f"hold_adv={self.hold_adv:.2f}"
    )
    return r


def _traced_maybe_redecide(self, snap, elapsed_sec):
    if snap.chain_end_triggered_p1 or snap.chain_end_triggered_p2:
        _log.append(
            f"[_maybe_redecide EDGE] t={elapsed_sec:.2f} "
            f"chain_end1={snap.chain_end_triggered_p1} total1={snap.chain_total_score_p1} "
            f"redecided1={self._redecided1} pred1={self._pred_score1:.0f} | "
            f"chain_end2={snap.chain_end_triggered_p2} total2={snap.chain_total_score_p2} "
            f"redecided2={self._redecided2} pred2={self._pred_score2:.0f}"
        )
    return _orig_maybe_redecide(self, snap, elapsed_sec)


def _traced_kill(self, b1, b2, state1=None, state2=None):
    before = self.hold_adv
    adv, p1 = _orig_kill(self, b1, b2, state1=state1, state2=state2)
    if adv != before:
        _log.append(
            f"[kill_override] hold_adv={before:.2f} -> adv={adv:.2f} "
            f"incoming1={self._incoming_total_p1:.1f} incoming2={self._incoming_total_p2:.1f} "
            f"state1={state1} state2={state2}"
        )
    return adv, p1


ov.ResolvedExchangeTracker._resolve = _traced_resolve
ov.ResolvedExchangeTracker._maybe_redecide = _traced_maybe_redecide
ov.ResolvedExchangeTracker.hold_after_kill_override = _traced_kill

hist: list[tuple[float, float]] = []
ov.generate(
    VIDEO, Path("data/verify/_unused_issue19_why_no_effect.mp4"),
    max_sec=0.0, sample_interval=0.0, start_sec=162.0, end_sec=204.0,
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
    enable_resolved_victim_gen_live=True,
    layout="panel", render=False,
    debug_history_out=hist,
)

# 201.2-203.4 に関連するログだけ抜き出す (_resolve/_maybe_redecide は
# elapsed_sec、kill_override はタイムスタンプ非公開のため全件出す)。
for line in _log:
    print(line)
print("\n=== hold_adv/winprob 履歴 (t=201.0-203.5) ===")
for t, adv in hist:
    if 201.0 <= t <= 203.5:
        print(f"t={t:.2f} adv={adv:.2f} p1win={ov.adv_to_winprob(adv) * 100.0:.1f}%")
print("DIAG_ISSUE19_WHY_NO_EFFECT_DONE")
