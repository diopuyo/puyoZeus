"""指摘19 窓 (絶対t=201.2-203.4) で実際に -100/0.7% を作っている経路を特定する
(read-only)。update()/_reevaluate_live_defender/hold_after_kill_override の
全呼び出しを計装し、絶対時刻でフィルタして出力する。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")

_orig_update = ov.ResolvedExchangeTracker.update
_orig_reeval = ov.ResolvedExchangeTracker._reevaluate_live_defender
_orig_kill = ov.ResolvedExchangeTracker.hold_after_kill_override
_cur_t = {"v": 0.0}  # generate() 側の絶対 t (フレームループ) を橋渡しする


def _traced_update(self, r_p1, r_p2, snap, elapsed_sec, t_sec=None, b1=None, b2=None):
    before = self.hold_adv
    ev1_before = "None" if r_p1.chain_event is None else "ev"
    ev2_before = "None" if r_p2.chain_event is None else "ev"
    active_before = self._active
    ret = _orig_update(self, r_p1, r_p2, snap, elapsed_sec, t_sec=t_sec, b1=b1, b2=b2)
    t_abs = _cur_t["v"]
    if 200.5 <= t_abs <= 204.0:
        print(
            f"[update] t_abs={t_abs:.2f} active_before={active_before} "
            f"ev1={ev1_before} ev2={ev2_before} state1={r_p1.state} state2={r_p2.state} "
            f"hold_adv {before:.2f}->{self.hold_adv:.2f} active_after={self._active}"
        )
    return ret


def _traced_reeval(self, b1, b2, snap=None, state1=None, state2=None):
    before = self.hold_adv
    ret = _orig_reeval(self, b1, b2, snap=snap, state1=state1, state2=state2)
    t_abs = _cur_t["v"]
    if 200.5 <= t_abs <= 204.0 and self.hold_adv != before:
        print(
            f"[_reevaluate_live_defender] t_abs={t_abs:.2f} "
            f"hold_adv {before:.2f}->{self.hold_adv:.2f} "
            f"defender_side={self.hold_defender_side} incoming={self.hold_incoming_ojama:.1f} "
            f"defender_prob={self.hold_defender_prob}"
        )
    return ret


def _traced_kill(self, b1, b2, state1=None, state2=None):
    before = self.hold_adv
    adv, p1 = _orig_kill(self, b1, b2, state1=state1, state2=state2)
    t_abs = _cur_t["v"]
    if 200.5 <= t_abs <= 204.0:
        print(
            f"[kill_override] t_abs={t_abs:.2f} hold_adv={before:.2f} -> adv={adv:.2f} "
            f"incoming1={self._incoming_total_p1:.1f} incoming2={self._incoming_total_p2:.1f} "
            f"state1={state1} state2={state2}"
        )
    return adv, p1


ov.ResolvedExchangeTracker.update = _traced_update
ov.ResolvedExchangeTracker._reevaluate_live_defender = _traced_reeval
ov.ResolvedExchangeTracker.hold_after_kill_override = _traced_kill

# generate() 内の t (フレームループの絶対秒) を捕まえるため、EarlyFireTracker等
# 既存の何かをフックするのは大変なので、debug_history_out の (t, adv) 系列から
# 「直近の t」を _cur_t に反映する軽量フック (History append 直前に呼ばれる
# generate() 内の t 変数は直接参照できないため、代わりに resolved_tracker.update
# 呼び出しに渡る elapsed_sec と t_sec の関係から絶対 t を推定する)。
_orig_generate = ov.generate


def _t_bridge(*args, **kwargs):
    # generate() 呼び出し前に t_sec を捕まえるフックは複雑なため、簡易策として
    # cv2 フレームループの time.sleep 相当のフックは行わず、update() 呼び出し
    # ごとに t_sec (絶対時刻) を _cur_t に反映する (update() の t_sec 引数は
    # generate() が elapsed_sec と同時に絶対時刻ベースで渡す)。
    return _orig_generate(*args, **kwargs)


# update() の t_sec 引数 (絶対時刻) を直接使う (elapsed_sec は試合内相対)。
def _traced_update2(self, r_p1, r_p2, snap, elapsed_sec, t_sec=None, b1=None, b2=None):
    if t_sec is not None:
        _cur_t["v"] = t_sec
    return _traced_update(self, r_p1, r_p2, snap, elapsed_sec, t_sec=t_sec, b1=b1, b2=b2)


ov.ResolvedExchangeTracker.update = _traced_update2

hist: list[tuple[float, float]] = []
ov.generate(
    VIDEO, Path("data/verify/_unused_issue19_pinpoint.mp4"),
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
print("DIAG_ISSUE19_PINPOINT_DONE")
