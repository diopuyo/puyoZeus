"""指摘14 案1やり直し用の追加計装 (2026-08-15): `_reevaluate_live_defender` の
呼び出し1回1回 (フレーム間引き無し) の ev1/ev2/state1/state2/hold_adv 遷移を
確定する。coordinator 指示: 「defender の own event が None」だけで判定して
いた旧strict案がなぜ効かなかったかを正確に特定するための追加確認。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402

START_SEC = 162.0
END_SEC = 202.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue14_reeval_calls.mp4")
WINDOW = (193.5, 201.0)

_orig_reeval = ov.ResolvedExchangeTracker._reevaluate_live_defender
_orig_update = ov.ResolvedExchangeTracker.update
_rows: list[str] = []
_last_state: dict[str, "object"] = {}


def _patched_update(self, r_p1, r_p2, snap, elapsed_sec, t_sec=None, b1=None, b2=None):
    tt = elapsed_sec if t_sec is None else t_sec
    _last_state["state1"] = r_p1.state
    _last_state["state2"] = r_p2.state
    return _orig_update(self, r_p1, r_p2, snap, elapsed_sec, t_sec=t_sec, b1=b1, b2=b2)


def _patched_reeval(self, b1, b2, snap=None, ev1=None, ev2=None):
    before_adv = self.hold_adv
    tt = self._t_sec
    _orig_reeval(self, b1, b2, snap, ev1=ev1, ev2=ev2)
    if WINDOW[0] <= tt <= WINDOW[1]:
        changed = self.hold_adv != before_adv
        defender_side = None
        if self._result is not None:
            defender_side, _ = self._decisive_defender(self._result)
        s1 = _last_state.get("state1")
        s2 = _last_state.get("state2")
        _rows.append(
            f"t={tt:7.3f} ev1None={ev1 is None!s:5s} ev2None={ev2 is None!s:5s} "
            f"ev1_cc={getattr(ev1, 'chain_count', None)!s:>4s} "
            f"ev2_cc={getattr(ev2, 'chain_count', None)!s:>4s} "
            f"state1={getattr(s1, 'name', s1)!s:14s} state2={getattr(s2, 'name', s2)!s:14s} "
            f"defender={defender_side!s:4s} changed={changed!s:5s} "
            f"adv_before={before_adv:+7.2f} adv_after={self.hold_adv:+7.2f}"
        )


ov.ResolvedExchangeTracker.update = _patched_update
ov.ResolvedExchangeTracker._reevaluate_live_defender = _patched_reeval


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
    print(f"\n[reeval calls] n={len(_rows)} in window {WINDOW}")
    for row in _rows:
        print(row)
    print("\nDONE_ISSUE14_REEVAL_CALLS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
