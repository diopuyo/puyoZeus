"""修正①適用後にseg01で新規発生した3エピソード (t≈205.3/311.45/817.97) の
内部値を計装する (2026-08-22)。_diag_t6717_kill_override_root_cause_2026-08-22.py
と同一の計装パターンを再利用 (コードは変更しない、monkeypatchのみ)。
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
OUT_DIR = PROJECT_ROOT / "logs/_diag_new_episodes_seg01_root_cause_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEG_START = 0.0
SEG_END = 825.0
WARMUP = 30.0
WATCH_WINDOWS = [(202.0, 208.0), (308.0, 314.0), (814.0, 821.0)]

_STATE: dict = {"t": None}
_detail: list[str] = []


def _in_watch(t: float) -> bool:
    return any(lo <= t <= hi for lo, hi in WATCH_WINDOWS)


def main() -> None:
    orig_drive_ojama = vao._drive_ojama
    orig_inputs_fn = vao._kill_override_chain_completion_inputs
    orig_kill_override = vao.kill_override

    def patched_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw):
        _STATE["t"] = t
        return orig_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw)

    def patched_inputs_fn(r_p1, r_p2, snap, b1, b2, room1, room2, elapsed_sec):
        t = _STATE.get("t")
        watch = t is not None and _in_watch(t)
        if watch:
            ev1 = getattr(r_p1, "chain_event", None)
            ev2 = getattr(r_p2, "chain_event", None)
            _detail.append(
                f"[1,2] t={t:.3f} state1={r_p1.state} state2={r_p2.state} "
                f"ev1={'None' if ev1 is None else f'chain_count={ev1.chain_count} total_score={ev1.total_score} mechanism={ev1.mechanism}'} "
                f"ev2={'None' if ev2 is None else f'chain_count={ev2.chain_count} total_score={ev2.total_score} mechanism={ev2.mechanism}'} "
                f"snap.pending_p1={snap.pending_p1} snap.pending_p2={snap.pending_p2} "
                f"room1_in={room1} room2_in={room2}"
            )
        result = orig_inputs_fn(r_p1, r_p2, snap, b1, b2, room1, room2, elapsed_sec)
        if watch:
            r1, r2, p1, p2 = result
            _detail.append(f"  [5] room1_eff={r1} room2_eff={r2} pending1_eff={p1} pending2_eff={p2}")
        return result

    def patched_kill_override(adv, inc1, inc2, room1, room2):
        t = _STATE.get("t")
        result = orig_kill_override(adv, inc1, inc2, room1, room2)
        if t is not None and _in_watch(t) and result != adv:
            _detail.append(
                f"  [6] FIRED t={t:.3f} adv_in={adv:+.2f} inc1={inc1} inc2={inc2} "
                f"room1={room1} room2={room2} -> adv_out={result:+.2f}"
            )
        return result

    vao._drive_ojama = patched_drive_ojama
    vao._kill_override_chain_completion_inputs = patched_inputs_fn
    vao.kill_override = patched_kill_override
    try:
        vao.generate(
            video=VIDEO, out=OUT_DIR / "_dummy.mp4",
            max_sec=0.0, sample_interval=0.0,
            start_sec=SEG_START, end_sec=SEG_END, warmup_sec=WARMUP,
            model_dir=MODEL_DIR, layout="panel", panel_subtitle_h=0,
            render=False, dump_timeline_path=None,
            enable_early_fire_reaction=True,
            enable_per_side_settled=True,
            disable_score_lead_bias=True,
            disable_pressure=True,
            enable_counter_reach=True,
            normalize_fps_30=True,
            use_production_recognition=True,
            resize_1080p=True,
            enable_resolved_live_defender_strict=True,
            enable_resolved_kill_override=True,
            enable_resolved_exchange_eval=True,
            enable_resolved_decisive_amplify=True,
            enable_resolved_live_defender=True,
            force_in_match=False,
            enable_kill_override_chain_completion=True,
        )
    finally:
        vao._drive_ojama = orig_drive_ojama
        vao._kill_override_chain_completion_inputs = orig_inputs_fn
        vao.kill_override = orig_kill_override

    out_path = OUT_DIR / "detail.log"
    out_path.write_text("\n".join(_detail), encoding="utf-8")
    print(f"[detail] {len(_detail)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
