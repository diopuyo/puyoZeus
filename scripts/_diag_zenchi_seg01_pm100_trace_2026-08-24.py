"""seg01 (納品レンダ) t=0〜215 を本番同一フラグで再走し、±100 を書いた機構を計装する。

計装対象 (本体コードは無変更、monkeypatch のみ):
  1. kill_override()           — per-frame 経路 (:5335) と hold 経路 (:1531) の全呼び出し
  2. _kill_override_chain_completion_inputs() — 是正後の kpending/kroom (dump 欠損分)
  3. ResolvedExchangeTracker.update()          — 決着ホールドの active 遷移 + 絶対 t 橋渡し
  4. ResolvedExchangeTracker.hold_after_kill_override() — ホールド中の表示上書き
  5. debug_history_out                          — 実際に表示された disp_adv
出力: logs/_diag_zenchi_seg01_pm100_trace_2026-08-24.log (stdout リダイレクト前提)
      logs/_diag_zenchi_seg01_pm100_dump_2026-08-24.npz (新規 dump、既存は触らない)
      logs/_diag_zenchi_seg01_pm100_hist_2026-08-24.csv (disp_adv 時系列)

フラグは logs/zenchi_render_2026-08-21/seg01.log:2 の [flags] 行と同一構成
(warmup 30s は start=0 では no-op のため省略)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

import scripts.visualize_advantage_overlay as ov  # noqa: E402

cv2.setNumThreads(1)  # 本番 main() と同一 (スレッド過剰生成の防止)

VIDEO = Path("data/frames/video_zenchi_c0BQoMJwwQU.mp4")
OUT_UNUSED = Path("logs/_diag_zenchi_seg01_pm100_unused_2026-08-24.mp4")
DUMP = Path("logs/_diag_zenchi_seg01_pm100_dump_2026-08-24.npz")
HIST = Path("logs/_diag_zenchi_seg01_pm100_hist_2026-08-24.csv")
END_SEC = 215.0
# 詳細ログを出す窓 (それ以外は kill 発火の要約のみ)
W0, W1 = 165.0, 215.0

_cur = {"t": 0.0, "in_hold_kill": False, "active": False}

_orig_kill = ov.kill_override
_orig_inputs = ov._kill_override_chain_completion_inputs
_orig_update = ov.ResolvedExchangeTracker.update
_orig_hkill = ov.ResolvedExchangeTracker.hold_after_kill_override


def _traced_kill(adv: float, inc1: float, inc2: float,
                 room1: int, room2: int) -> float:
    """kill_override の全呼び出しを記録する (呼出元: perframe / hold)。"""
    out = _orig_kill(adv, inc1, inc2, room1, room2)
    t = _cur["t"]
    src = "hold" if _cur["in_hold_kill"] else "perframe"
    if out != adv or (W0 <= t <= W1):
        print(f"[kill:{src}] t={t:.2f} adv_in={adv:.1f} -> {out:.1f} "
              f"inc1={inc1:.1f} inc2={inc2:.1f} room1={room1} room2={room2}")
    return out


def _traced_inputs(snap, b1, b2, room1, room2,
                   gen1, gen_before1, gen2, gen_before2):
    """連鎖完走後是正の入出力を記録する。"""
    kroom1, kroom2, kp1, kp2 = _orig_inputs(
        snap, b1, b2, room1, room2, gen1, gen_before1, gen2, gen_before2)
    t = _cur["t"]
    corrected = (kroom1 != room1 or kroom2 != room2
                 or kp1 != float(snap.pending_p1) or kp2 != float(snap.pending_p2))
    if corrected and (W0 <= t <= W1):
        print(f"[cc_inputs] t={t:.2f} pend=({snap.pending_p1},{snap.pending_p2}) "
              f"room=({room1},{room2}) gen=({gen1:.1f},{gen2:.1f}) "
              f"-> kpend=({kp1:.1f},{kp2:.1f}) kroom=({kroom1},{kroom2})")
    return kroom1, kroom2, kp1, kp2


def _traced_update(self, r_p1, r_p2, snap, elapsed_sec,
                   t_sec=None, b1=None, b2=None):
    """絶対 t の橋渡し + ホールド active 遷移の記録。"""
    if t_sec is not None:
        _cur["t"] = t_sec
    prev_active = self._active
    ret = _orig_update(self, r_p1, r_p2, snap, elapsed_sec,
                       t_sec=t_sec, b1=b1, b2=b2)
    if self._active != prev_active:
        print(f"[hold] t={_cur['t']:.2f} active {prev_active}->{self._active} "
              f"hold_adv={self.hold_adv:.1f} inc1={self._incoming_total_p1:.1f} "
              f"inc2={self._incoming_total_p2:.1f}")
    return ret


def _traced_hkill(self, b1, b2, state1=None, state2=None):
    """hold_after_kill_override の入出力を記録する。"""
    _cur["in_hold_kill"] = True
    try:
        adv, p1 = _orig_hkill(self, b1, b2, state1=state1, state2=state2)
    finally:
        _cur["in_hold_kill"] = False
    t = _cur["t"]
    if (W0 <= t <= W1) and adv != self.hold_adv:
        print(f"[hkill] t={t:.2f} hold_adv={self.hold_adv:.1f} -> disp={adv:.1f} "
              f"state1={state1} state2={state2}")
    return adv, p1


ov.kill_override = _traced_kill
ov._kill_override_chain_completion_inputs = _traced_inputs
ov.ResolvedExchangeTracker.update = _traced_update
ov.ResolvedExchangeTracker.hold_after_kill_override = _traced_hkill


def main() -> None:
    """本番フラグ同一構成 (seg01.log:2) で t=0〜215 を再走する。"""
    hist: list[tuple[float, float]] = []
    ov.generate(
        VIDEO, OUT_UNUSED, max_sec=0.0, sample_interval=0.0,
        start_sec=0.0, end_sec=END_SEC,
        model_dir=Path("data/verify/retrain_model62_2026-08-21"),
        force_in_match=False,                      # --no-force-in-match
        enable_resolved_exchange_eval=True,        # --resolved-exchange-eval
        enable_resolved_decisive_amplify=True,     # --resolved-decisive-amplify
        enable_resolved_live_defender=True,        # --resolved-live-defender
        enable_resolved_live_defender_strict=True, # --resolved-live-defender-strict
        enable_resolved_kill_override=True,        # --resolved-kill-override
        enable_kill_override_chain_completion=True,  # --kill-override-chain-completion
        enable_slide_exit_min_display_guard=True,  # --enable-slide-exit-min-display-guard
        enable_early_fire_reaction=True,           # --early-fire-reaction
        enable_per_side_settled=True,              # --per-side-settled
        disable_score_lead_bias=True,              # --no-score-lead-bias
        disable_pressure=True,                     # --no-pressure
        enable_counter_reach=True,                 # --counter-reach
        layout="panel", panel_subtitle_h=0,
        render=False,                              # 描画/書き出しなし (表示値は hist に出る)
        dump_timeline_path=DUMP,
        debug_history_out=hist,
        # normalize_fps_30 / production_recognition / resize_1080p は既定 True
    )
    with HIST.open("w", encoding="utf-8") as fh:
        fh.write("t_sec,disp_adv\n")
        for t, a in hist:
            fh.write(f"{t:.4f},{a:.4f}\n")
    print(f"[done] hist rows={len(hist)} -> {HIST}")


if __name__ == "__main__":
    main()
