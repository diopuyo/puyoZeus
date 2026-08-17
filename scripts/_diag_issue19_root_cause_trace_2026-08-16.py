"""指摘19 根治調査: ResolvedExchangeTracker._resolve/_maybe_redecide の内部値を
計装トレースし、「victim側の score が途中値のまま固定される」という architect の
仮説 (_redecided1/2 の一回きり制限が2本目以降の確定値を握りつぶす) を実測で
検証する (read-only、production_config.py は変更しない)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")

_orig_resolve = ov.ResolvedExchangeTracker._resolve
_orig_maybe_redecide = ov.ResolvedExchangeTracker._maybe_redecide
_log: list[str] = []


def _traced_resolve(self, snap, elapsed_sec, score1, score2):
    _log.append(
        f"[_resolve] elapsed={elapsed_sec:.2f} score1={score1:.0f} score2={score2:.0f} "
        f"redecided1={self._redecided1} redecided2={self._redecided2} "
        f"chain_end1={snap.chain_end_triggered_p1} chain_end2={snap.chain_end_triggered_p2} "
        f"chain_total1={snap.chain_total_score_p1} chain_total2={snap.chain_total_score_p2}"
    )
    return _orig_resolve(self, snap, elapsed_sec, score1, score2)


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


ov.ResolvedExchangeTracker._resolve = _traced_resolve
ov.ResolvedExchangeTracker._maybe_redecide = _traced_maybe_redecide

hist: list[tuple[float, float]] = []
ov.generate(
    VIDEO, Path("data/verify/_unused_issue19_root_cause_trace.mp4"),
    max_sec=0.0, sample_interval=0.0, start_sec=193.0, end_sec=205.0,
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

print("\n".join(_log))
print("\n=== hold_adv/winprob 履歴 (t=200.0-204.0) ===")
for t, adv in hist:
    if 200.0 <= t <= 204.0:
        print(f"t={t:.2f} adv={adv:.2f} p1win={ov.adv_to_winprob(adv) * 100.0:.1f}%")
print("DIAG_ISSUE19_ROOT_CAUSE_TRACE_DONE")
