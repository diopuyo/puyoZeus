"""残存3件がどの呼び出し元 (_stash_and_clear_active_chain の呼び出し行) から
発生しているかを特定する (2026-08-22)。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
cv2.setNumThreads(1)

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/slide_suppress_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEG_START = 6664.17
SEG_END = 6725.0
WARMUP = 30.0
WATCH_LO, WATCH_HI = 6700.0, 6720.0

_STATE: dict = {"t": None}


def main() -> None:
    orig_drive_ojama = vao._drive_ojama

    def patched_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw):
        _STATE["t"] = t
        return orig_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw)

    vao._drive_ojama = patched_drive_ojama

    orig_stash = RecognitionPipeline._stash_and_clear_active_chain

    def patched_stash(self, side):  # type: ignore[no-untyped-def]
        t = _STATE.get("t")
        if (
            side == "1P"
            and self._active_chain_1p is not None
            and t is not None
            and WATCH_LO <= t <= WATCH_HI
        ):
            caller = sys._getframe(1)
            print(
                f"[stash] t={t:.3f} caller_line={caller.f_lineno} "
                f"chain_count={self._active_chain_1p.chain_count} "
                f"mechanism={getattr(self._active_chain_1p, 'mechanism', None)} "
                f"last_next={self._last_seen_next_1p} "
                f"start_next={self._chain_start_next_1p}"
            )
        return orig_stash(self, side)

    RecognitionPipeline._stash_and_clear_active_chain = patched_stash
    try:
        vao.generate(
            video=VIDEO, out=OUT_DIR / "_dummy_residual2.mp4",
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
            enable_slide_exit_min_display_guard=True,
            force_in_match=False,
        )
    finally:
        vao._drive_ojama = orig_drive_ojama
        RecognitionPipeline._stash_and_clear_active_chain = orig_stash


if __name__ == "__main__":
    main()
