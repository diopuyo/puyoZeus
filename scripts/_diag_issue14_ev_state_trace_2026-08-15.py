"""指摘14 案1やり直し用の計装 (2026-08-15): 誤爆時点の ev1/ev2/state1/state2 の
実際の値を確定する (coordinator 指示、推測で直さない)。

対象窓: 絶対 t=193〜202秒 (final3_m1 で 2P に誤って生存率18.9%が5.2秒表示
された区間、195.33-200.53 を含む)。

`ResolvedExchangeTracker.update` を monkeypatch し、呼び出しごとに
(t_sec, ev1 is None?, ev2 is None?, r_p1.state, r_p2.state,
 _decisive_defender の結果) を記録する。本体コードは一切変更しない。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

START_SEC = 162.0
END_SEC = 202.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue14_ev_state.mp4")
WINDOW = (193.0, 202.0)

_orig_update = ov.ResolvedExchangeTracker.update
_records: list[dict] = []


def _patched_update(self, r_p1, r_p2, snap, elapsed_sec, t_sec=None, b1=None, b2=None):
    tt = elapsed_sec if t_sec is None else t_sec
    ev1, ev2 = r_p1.chain_event, r_p2.chain_event
    active_before = self._active
    if WINDOW[0] <= tt <= WINDOW[1]:
        defender_side = None
        if active_before and self._result is not None:
            defender_side, _incoming = self._decisive_defender(self._result)
        rec = {
            "t": tt,
            "ev1_none": ev1 is None,
            "ev2_none": ev2 is None,
            "ev1_cc": None if ev1 is None else ev1.chain_count,
            "ev2_cc": None if ev2 is None else ev2.chain_count,
            "ev1_trigger": None if ev1 is None else ev1.trigger_sec,
            "ev2_trigger": None if ev2 is None else ev2.trigger_sec,
            "state1": r_p1.state.name,
            "state2": r_p2.state.name,
            "active_before": active_before,
            "defender_side": defender_side,
        }
        _records.append(rec)
    return _orig_update(self, r_p1, r_p2, snap, elapsed_sec, t_sec=t_sec, b1=b1, b2=b2)


ov.ResolvedExchangeTracker.update = _patched_update


def main() -> int:
    ov.generate(
        VIDEO, OUT, max_sec=0.0, sample_interval=0.0,
        start_sec=START_SEC, end_sec=END_SEC,
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
        enable_resolved_live_defender=True,
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
    )
    print(f"\n[records] n={len(_records)} in window {WINDOW}")
    print("t        ev1None ev2None ev1_cc ev2_cc ev1_trig ev2_trig state1        state2        active defender")
    last_printed = None
    for r in _records:
        bucket = round(r["t"] / 0.2)
        if bucket == last_printed:
            continue
        last_printed = bucket
        print(
            f"{r['t']:7.2f}  {str(r['ev1_none']):5s}  {str(r['ev2_none']):5s}  "
            f"{str(r['ev1_cc']):>5s}  {str(r['ev2_cc']):>5s}  "
            f"{str(r['ev1_trigger']):>7s}  {str(r['ev2_trigger']):>7s}  "
            f"{r['state1']:12s}  {r['state2']:12s}  {str(r['active_before']):5s}  "
            f"{str(r['defender_side'])}"
        )
    print("\nDONE_ISSUE14_EV_STATE_TRACE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
