"""指摘12窓 (t=234.87) の hold_adv/hold_p1 不一致の原因を確認する軽量診断。

_diag_issue12_fix_verify_2026-08-14.py と同じスタブ手法 (HeavyAdvCache/
ライブ CounterReachTracker をスタブ化し計算コストを削減、決着ホールド専用の
_counter_tracker は本物のまま) で、_amplify_decisive の amp 適用前後の
(adv, p1) を直接ログする。本体コードは変更しない。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402
iv = ov.iv

START_SEC = 162.0
END_SEC = 237.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_amp_breakdown.mp4")

_CUR_T = {"t": None}
LOG: list[dict] = []


def _patch_t_tracker() -> None:
    from src.recognition_pipeline import RecognitionPipeline
    orig = RecognitionPipeline.update

    def patched(self, fi, t, frame):
        _CUR_T["t"] = t
        return orig(self, fi, t, frame)

    RecognitionPipeline.update = patched


def _patch_hcache_stub() -> None:
    def patched(self, b1, b2, snap, sp1, sp2, elapsed):
        return (0.0, 0.0, [], 0.0, 0.0, 0.0, 0.0)
    ov.HeavyAdvCache.update = patched


def _tag_decisive_counter_tracker() -> None:
    orig_init = ov.ResolvedExchangeTracker.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._counter_tracker._is_decisive_amplify_tracker = True

    ov.ResolvedExchangeTracker.__init__ = patched_init


def _patch_live_counter_tracker_stub() -> None:
    orig = ov.CounterReachTracker._update_defender_only

    def patched(self, b1, b2, budget_sec, next1, next2, t_sec, defender_side, threshold_ojama):
        if not getattr(self, "_is_decisive_amplify_tracker", False):
            self.last_hands = 0.0
            return (0.0, float("nan"), float("nan"))
        return orig(self, b1, b2, budget_sec, next1, next2, t_sec, defender_side, threshold_ojama)

    ov.CounterReachTracker._update_defender_only = patched


def _patch_amplify_logger() -> None:
    orig = ov.ResolvedExchangeTracker._amplify_decisive

    def patched(self, adv, result):
        defender_side, incoming = self._decisive_defender(result)
        adv_before = adv
        p1_before = ov.adv_to_winprob(adv_before)
        new_adv, new_p1 = orig(self, adv, result)
        LOG.append({
            "t": _CUR_T["t"], "adv_before": adv_before, "p1_before": p1_before,
            "adv_after": new_adv, "p1_after": new_p1,
            "adv_to_winprob_of_after": ov.adv_to_winprob(new_adv),
            "defender_side": defender_side, "incoming": incoming,
            "defender_prob": self.hold_defender_prob,
        })
        return new_adv, new_p1

    ov.ResolvedExchangeTracker._amplify_decisive = patched


def main() -> int:
    _tag_decisive_counter_tracker()
    _patch_t_tracker()
    _patch_hcache_stub()
    _patch_live_counter_tracker_stub()
    _patch_amplify_logger()
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
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
    )
    print(f"n_amplify_calls={len(LOG)}")
    for e in LOG:
        print(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
