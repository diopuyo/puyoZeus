"""大連鎖10ケースの ChainEvent 遷移プローブ・新フラグON版 (2026-08-24 コーダ)。

logs/_analyst_telop_check_2026-08-24/_probe_chain_events.py と同一の起動条件
(residual2 構成 + 修正③ON) に、STABLE凍結デッドロック根治フラグを
load_default 経由で注入して再実行する。stash 時の chain_count / total_score
を before (既存 probe_c*.log) と突合するための after 側データを取る。

使い方: python scripts/_probe_formula_fix_cases_2026-08-24.py <case> <t0> <t1>
出力: logs/_probe_formula_fix_cases_2026-08-24/probe_<case>.log (stdout)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

cv2.setNumThreads(1)

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

CASE = sys.argv[1]
T0 = float(sys.argv[2])
T1 = float(sys.argv[3])
SEG_START = max(0.0, T0 - 36.0)
SEG_END = T1 + 6.0
WARMUP = 30.0
VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/_probe_formula_fix_cases_2026-08-24"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 根治フラグ (ON 検証対象)
NEW_FLAGS = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
)

_STATE: dict = {"t": None, "last": {}}

_orig_load_default = RecognitionPipeline.load_default.__func__


def _patched_load_default(cls, *args, **kwargs):
    kwargs.update(NEW_FLAGS)
    return _orig_load_default(cls, *args, **kwargs)


def main() -> None:
    RecognitionPipeline.load_default = classmethod(_patched_load_default)
    orig = vao._drive_ojama

    def patched(tracker, p1, p2, ps1, ps2, t, **kw):
        _STATE["t"] = t
        pipe = kw.get("pipeline")
        if pipe is not None:
            for attr in ("_active_chain_1p", "_active_chain_2p"):
                ev = getattr(pipe, attr, None)
                cur = None if ev is None else (
                    id(ev) % 100000, ev.chain_count, ev.mechanism,
                    ev.total_score, ev.ojama_sent,
                    round(ev.trigger_sec, 2), ev.score_estimated)
                if _STATE["last"].get(attr) != cur:
                    print(f"[ev] t={t:.3f} {attr} {cur}", flush=True)
                    _STATE["last"][attr] = cur
        return orig(tracker, p1, p2, ps1, ps2, t, **kw)

    vao._drive_ojama = patched
    orig_stash = RecognitionPipeline._stash_and_clear_active_chain

    def pst(self, side):
        t = _STATE.get("t")
        ev = self._active_chain_1p if side == "1P" else self._active_chain_2p
        if ev is not None and t is not None:
            print(
                f"[stash] t={t:.3f} side={side} cc={ev.chain_count} "
                f"mech={ev.mechanism} score={ev.total_score} "
                f"ojama={ev.ojama_sent}", flush=True,
            )
        return orig_stash(self, side)

    RecognitionPipeline._stash_and_clear_active_chain = pst
    try:
        vao.generate(
            video=VIDEO, out=OUT_DIR / f"_dummy_{CASE}.mp4",
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
        vao._drive_ojama = orig
        RecognitionPipeline._stash_and_clear_active_chain = orig_stash
        RecognitionPipeline.load_default = classmethod(_orig_load_default)


if __name__ == "__main__":
    main()
