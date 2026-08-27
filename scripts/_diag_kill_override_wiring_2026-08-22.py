"""kill_override 配線計装 (2026-08-22)。

目的: 「致死が確定している側を有利と表示する」矛盾の原因追跡。
本体コード (scripts/visualize_advantage_overlay.py) は変更せず、monkeypatch で
下記4点をフレーム単位 (settled更新単位) で記録する:

  1. ResolvedExchangeTracker.update() の呼出結果 (active/just_deactivated/
     hold_adv/hold_p1/_incoming_total_p1/_incoming_total_p2)
  2. ResolvedExchangeTracker.hold_after_kill_override() の呼出回数・
     発動 (override) 回数
  3. モジュール関数 kill_override() の呼出 (adv, inc1, inc2, room1, room2) -> 結果
     (ライブ per-frame 経路、4755行の呼び出し)
  4. RealtimeForecastTracker.update() の結果 inc1/inc2 (kill_override に
     渡る「確定予告」の代役ヒューリスティック値)
  5. _drive_ojama() の戻り値 snap.pending_p1/pending_p2 (OjamaAccountingTracker
     の確定値、走査器 D1b が「真の予告」として使う値。ojama_accounting.py
     docstring 確認済み)

t の対応付け: _drive_ojama は毎フレーム呼ばれるため、そこで捕捉した t を
共有状態 _STATE["t"] に保持し、同一フレーム内で後から呼ばれる
resolved_tracker.update / fctracker.update / kill_override がそれを参照する
(generate() 内の呼び出し順序: _drive_ojama → resolved_tracker.update →
 (settled時) fctracker.update → kill_override、コード確認済み)。

実行構成:
  RUN1: 本番と同一 (--resolved-kill-override なし、既定OFF)
  RUN2: 本番 + --resolved-kill-override (陽性対照の準備、②の呼出確認)
  RUN3: RUN2 + KILL_RATIO_FULL を一時的に 0.1 に monkeypatch (④ 陽性対照:
        計装が「効いている/効いていない」を区別できることの証明)

対象区間: 0〜893.7s (t=887, t=200 の2エピソードを含む、seg1 と同一区間)。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
cv2.setNumThreads(1)

import scripts.visualize_advantage_overlay as vao  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/_diag_killoverride_ab_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 対象窓 (この範囲だけ詳細ログを出す。それ以外は集計のみ)
WATCH_WINDOWS = [(878.0, 902.0), (195.0, 212.0)]


def _in_watch(t: float) -> bool:
    return any(lo <= t <= hi for lo, hi in WATCH_WINDOWS)


_STATE: dict = {"t": None}


def instrument(
    log_path: Path, enable_resolved_kill_override: bool,
    kill_ratio_full_override: float | None = None,
    end_sec: float = 893.7,
) -> dict:
    """monkeypatch を仕込んで generate() を1回実行し、集計結果を返す。"""
    counters = {
        "drive_ojama_calls": 0,
        "resolved_update_calls": 0,
        "resolved_active_true_frames": 0,
        "hold_after_kill_override_calls": 0,
        "hold_after_kill_override_overridden": 0,
        "kill_override_calls": 0,
        "kill_override_fired_partial": 0,  # 0<g<1
        "kill_override_fired_full": 0,     # g=1 (完全上書き)
        "fctracker_update_calls": 0,
    }
    detail_lines: list[str] = []

    orig_drive_ojama = vao._drive_ojama
    orig_resolved_update = vao.ResolvedExchangeTracker.update
    orig_hold_after_kill = vao.ResolvedExchangeTracker.hold_after_kill_override
    orig_kill_override = vao.kill_override
    orig_fctracker_update = vao.RealtimeForecastTracker.update

    if kill_ratio_full_override is not None:
        orig_kill_ratio_full = vao.KILL_RATIO_FULL
        vao.KILL_RATIO_FULL = kill_ratio_full_override

    def patched_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw):
        counters["drive_ojama_calls"] += 1
        snap = orig_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw)
        _STATE["t"] = t
        _STATE["pending_p1"] = snap.pending_p1
        _STATE["pending_p2"] = snap.pending_p2
        _STATE["forecast_p1"] = snap.forecast_p1
        _STATE["forecast_p2"] = snap.forecast_p2
        if _in_watch(t):
            detail_lines.append(
                f"[drive_ojama] t={t:.3f} snap.pending_p1={snap.pending_p1} "
                f"snap.pending_p2={snap.pending_p2} "
                f"snap.forecast_p1={snap.forecast_p1} snap.forecast_p2={snap.forecast_p2}"
            )
        return snap

    def patched_resolved_update(self, r_p1, r_p2, snap, elapsed_sec, t_sec=None, b1=None, b2=None):
        counters["resolved_update_calls"] += 1
        active, just_deact = orig_resolved_update(
            self, r_p1, r_p2, snap, elapsed_sec, t_sec=t_sec, b1=b1, b2=b2)
        t = _STATE.get("t")
        if active:
            counters["resolved_active_true_frames"] += 1
        if t is not None and _in_watch(t):
            detail_lines.append(
                f"[resolved.update] t={t:.3f} active={active} just_deact={just_deact} "
                f"hold_adv={getattr(self, 'hold_adv', None)} "
                f"hold_p1={getattr(self, 'hold_p1', None)} "
                f"_incoming_total_p1={getattr(self, '_incoming_total_p1', None)} "
                f"_incoming_total_p2={getattr(self, '_incoming_total_p2', None)}"
            )
        return active, just_deact

    def patched_hold_after_kill_override(self, b1, b2, state1=None, state2=None):
        counters["hold_after_kill_override_calls"] += 1
        before = self.hold_adv
        adv, p1 = orig_hold_after_kill(self, b1, b2, state1=state1, state2=state2)
        overridden = (adv != before)
        if overridden:
            counters["hold_after_kill_override_overridden"] += 1
        t = _STATE.get("t")
        if t is not None and _in_watch(t):
            detail_lines.append(
                f"[hold_after_kill_override] t={t:.3f} before_hold_adv={before:.2f} "
                f"after_adv={adv:.2f} overridden={overridden} "
                f"_incoming_total_p1={self._incoming_total_p1} "
                f"_incoming_total_p2={self._incoming_total_p2} "
                f"state1={state1} state2={state2}"
            )
        return adv, p1

    def patched_kill_override(adv, inc1, inc2, room1, room2):
        counters["kill_override_calls"] += 1
        result = orig_kill_override(adv, inc1, inc2, room1, room2)
        # g を逆算 (target=+-100、result=(1-g)*adv+g*target)
        fired_full = (abs(result) == 100.0 and result != adv)
        fired_partial = (result != adv and not fired_full)
        if fired_full:
            counters["kill_override_fired_full"] += 1
        elif fired_partial:
            counters["kill_override_fired_partial"] += 1
        t = _STATE.get("t")
        if t is not None and _in_watch(t):
            detail_lines.append(
                f"[kill_override] t={t:.3f} adv_in={adv:.2f} inc1={inc1:.2f} inc2={inc2:.2f} "
                f"room1={room1} room2={room2} adv_out={result:.2f} "
                f"true_pending_p1={_STATE.get('pending_p1')} "
                f"true_pending_p2={_STATE.get('pending_p2')}"
            )
        return result

    def patched_fctracker_update(self, score1, score2, tsumo1, tsumo2, rate=70.0):
        counters["fctracker_update_calls"] += 1
        result = orig_fctracker_update(self, score1, score2, tsumo1, tsumo2, rate=rate)
        t = _STATE.get("t")
        if t is not None and _in_watch(t):
            detail_lines.append(
                f"[fctracker.update] t={t:.3f} inc1={self.inc1:.2f} inc2={self.inc2:.2f} "
                f"true_pending_p1={_STATE.get('pending_p1')} "
                f"true_pending_p2={_STATE.get('pending_p2')}"
            )
        return result

    vao._drive_ojama = patched_drive_ojama
    vao.ResolvedExchangeTracker.update = patched_resolved_update
    vao.ResolvedExchangeTracker.hold_after_kill_override = patched_hold_after_kill_override
    vao.kill_override = patched_kill_override
    vao.RealtimeForecastTracker.update = patched_fctracker_update

    try:
        vao.generate(
            video=VIDEO,
            out=OUT_DIR / "_diag_wiring_dummy.mp4",
            max_sec=0.0,
            sample_interval=0.15,
            start_sec=0.0,
            end_sec=end_sec,
            warmup_sec=30.0,
            model_dir=MODEL_DIR,
            force_in_match=False,
            enable_early_fire_reaction=True,
            enable_resolved_exchange_eval=True,
            enable_resolved_decisive_amplify=True,
            enable_resolved_live_defender=True,
            enable_resolved_live_defender_strict=True,
            enable_resolved_kill_override=enable_resolved_kill_override,
            layout="panel",
            panel_subtitle_h=0,
            render=False,
            dump_timeline_path=None,
        )
    finally:
        vao._drive_ojama = orig_drive_ojama
        vao.ResolvedExchangeTracker.update = orig_resolved_update
        vao.ResolvedExchangeTracker.hold_after_kill_override = orig_hold_after_kill
        vao.kill_override = orig_kill_override
        vao.RealtimeForecastTracker.update = orig_fctracker_update
        if kill_ratio_full_override is not None:
            vao.KILL_RATIO_FULL = orig_kill_ratio_full

    log_path.write_text("\n".join(detail_lines), encoding="utf-8")
    return counters


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["RUN1", "RUN2", "RUN3"], required=True)
    ap.add_argument("--end-sec", type=float, default=893.7,
                    help="スモークテスト用に短縮する場合に指定 (既定=区間全体)")
    a = ap.parse_args()

    if a.run == "RUN1":
        # 本番と同一 (--resolved-kill-override なし)
        counters = instrument(
            OUT_DIR / "RUN1_detail.log", enable_resolved_kill_override=False,
            kill_ratio_full_override=None, end_sec=a.end_sec)
    elif a.run == "RUN2":
        # 本番 + --resolved-kill-override (②の呼出確認用)
        counters = instrument(
            OUT_DIR / "RUN2_detail.log", enable_resolved_kill_override=True,
            kill_ratio_full_override=None, end_sec=a.end_sec)
    else:
        # RUN2 + KILL_RATIO_FULL=0.1 (④ 陽性対照)
        counters = instrument(
            OUT_DIR / "RUN3_detail.log", enable_resolved_kill_override=True,
            kill_ratio_full_override=0.1, end_sec=a.end_sec)

    summary_path = OUT_DIR / f"{a.run}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        for k, v in counters.items():
            f.write(f"{k}\t{v}\n")
    print(f"[{a.run}] summary -> {summary_path}")
    for k, v in counters.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
