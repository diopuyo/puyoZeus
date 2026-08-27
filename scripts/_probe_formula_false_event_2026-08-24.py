"""偽連鎖イベント率の before/after 測定プローブ (2026-08-24 コーダ)。

受け入れ条件4:「偽の連鎖イベントを増やしていないこと」の測定器。
同一ウィンドウを 旧構成 (off) / 根治フラグON (on) で流し、
  - 新規 ChainEvent の発生 (mechanism / chain_count / score)
  - score OCR 値の変化タイムライン
を記録する。後段の解析で「イベント trigger 後 SUPPORT_WINDOW 秒以内に
自 side の score が +40 (最小連鎖点) 以上増えたか」で真偽を分類する。

使い方:
  python scripts/_probe_formula_false_event_2026-08-24.py <mode> <t0> <t1> <tag>
出力: logs/_probe_formula_false_event_2026-08-24/probe_<tag>_<mode>.log
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

MODE = sys.argv[1]
assert MODE in ("off", "on"), MODE
T0 = float(sys.argv[2])
T1 = float(sys.argv[3])
TAG = sys.argv[4]
WARMUP = 30.0
VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/_probe_formula_false_event_2026-08-24"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_FLAGS = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
)

_orig_load_default = RecognitionPipeline.load_default.__func__


def _patched_load_default(cls, *args, **kwargs):
    if MODE == "on":
        kwargs.update(NEW_FLAGS)
    return _orig_load_default(cls, *args, **kwargs)


_orig_update = RecognitionPipeline.update
_last = {"ev1": None, "ev2": None, "s1": None, "s2": None}


def _patched_update(self, frame_idx, time_sec, frame):
    r = _orig_update(self, frame_idx, time_sec, frame)
    for key, ev in (("ev1", self._active_chain_1p),
                    ("ev2", self._active_chain_2p)):
        cur = None if ev is None else (
            round(ev.trigger_sec, 2), ev.mechanism, ev.chain_count,
            ev.total_score,
        )
        if _last[key] != cur:
            print(f"[ev] t={time_sec:.3f} {key} {cur}", flush=True)
            _last[key] = cur
    s1 = (self._score_tracker_1p.last_score
          if self._score_tracker_1p is not None else None)
    s2 = (self._score_tracker_2p.last_score
          if self._score_tracker_2p is not None else None)
    if _last["s1"] != s1:
        print(f"[score] t={time_sec:.3f} 1P {s1}", flush=True)
        _last["s1"] = s1
    if _last["s2"] != s2:
        print(f"[score] t={time_sec:.3f} 2P {s2}", flush=True)
        _last["s2"] = s2
    return r


def main() -> None:
    RecognitionPipeline.load_default = classmethod(_patched_load_default)
    RecognitionPipeline.update = _patched_update
    try:
        vao.generate(
            video=VIDEO, out=OUT_DIR / f"_dummy_{TAG}_{MODE}.mp4",
            max_sec=0.0, sample_interval=0.0,
            start_sec=max(0.0, T0 - WARMUP), end_sec=T1, warmup_sec=WARMUP,
            model_dir=MODEL_DIR, layout="panel", panel_subtitle_h=0,
            render=False, dump_timeline_path=None,
            enable_early_fire_reaction=True,
            enable_per_side_settled=True,
            disable_score_lead_bias=True,
            disable_pressure=True,
            enable_counter_reach=False,
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
        RecognitionPipeline.load_default = classmethod(_orig_load_default)
        RecognitionPipeline.update = _orig_update


if __name__ == "__main__":
    main()
