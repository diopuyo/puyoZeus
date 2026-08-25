"""source t=192〜200秒 の生 chain_event (両側) をフレーム毎に trace する
診断スクリプト (使い捨て、コミット対象外)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from scripts.visualize_advantage_overlay import generate  # noqa: E402

_orig_update = vao.ResolvedExchangeTracker.update
_n = {"i": 0}
_last1 = {"trig": None}
_last2 = {"trig": None}


def _traced_update(self, r_p1, r_p2, snap, elapsed_sec):
    _n["i"] += 1
    ev1, ev2 = r_p1.chain_event, r_p2.chain_event
    t1 = ev1.trigger_sec if ev1 is not None else None
    t2 = ev2.trigger_sec if ev2 is not None else None
    if t1 != _last1["trig"] or t2 != _last2["trig"]:
        c1 = f"cc={ev1.chain_count} score={ev1.total_score} trig={ev1.trigger_sec:.2f}" if ev1 else "None"
        c2 = f"cc={ev2.chain_count} score={ev2.total_score} trig={ev2.trigger_sec:.2f}" if ev2 else "None"
        print(f"[i={_n['i']}] ev1=({c1}) ev2=({c2}) "
              f"state1={r_p1.state.name} state2={r_p2.state.name} "
              f"score1={r_p1.score} score2={r_p2.score}")
        _last1["trig"], _last2["trig"] = t1, t2
    return _orig_update(self, r_p1, r_p2, snap, elapsed_sec)


vao.ResolvedExchangeTracker.update = _traced_update

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
    enable_pseudo_chain_score_fill=True,  # 根治① 検証用 (2026-08-13)
)

generate(
    Path("data/frames/review_demo_2026-08-12.mp4"),
    Path("data/verify/_unused_trace.mp4"),
    max_sec=0.0, sample_interval=0.0,
    start_sec=192.0, end_sec=200.0, warmup_sec=10.0,
    render=False,
    **BASE_KWARGS,
)
