"""t=6717.5 の断片化が「NextSlide信号 (enable_chain_exit_next_signal、
enable_gravity_settle_state=True により強制ON) の誤検知」で起きているかを
直接確認する (2026-08-22)。

RecognitionPipeline._slide_detector_1p.update() をmonkeypatchし、1P連鎖中
(6700-6720s) の slide_motion 判定結果を trigger_sec と突合する。
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

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/_diag_slide_false_positive_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEG_START = 6664.17
SEG_END = 6725.0
WARMUP = 30.0
WATCH_LO, WATCH_HI = 6700.0, 6720.0

_STATE: dict = {"t": None}
_detail: list[str] = []


def main() -> None:
    orig_drive_ojama = vao._drive_ojama

    def patched_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw):
        _STATE["t"] = t
        return orig_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw)

    vao._drive_ojama = patched_drive_ojama

    # SlideDetector.update をパッチして 1P 側の slide_motion 結果を記録する。
    # クラス自体を差し替えず、生成された pipe インスタンスの属性 (_slide_
    # detector_1p) にアクセスするため、RecognitionPipeline.load_default を
    # ラップして生成直後の pipe を横取りする。
    orig_load_default = RecognitionPipeline.load_default
    _pipe_holder: dict = {}

    def patched_load_default(*a, **kw):
        pipe = orig_load_default(*a, **kw)
        _pipe_holder["pipe"] = pipe
        if pipe._slide_detector_1p is not None:
            orig_update = pipe._slide_detector_1p.update

            def patched_slide_update(prev_frame, frame):
                result = orig_update(prev_frame, frame)
                t = _STATE.get("t")
                if t is not None and WATCH_LO <= t <= WATCH_HI and result.slide_motion:
                    _detail.append(f"[slide_1p=True] t={t:.3f}")
                return result

            pipe._slide_detector_1p.update = patched_slide_update
        return pipe

    RecognitionPipeline.load_default = patched_load_default
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
        )
    finally:
        vao._drive_ojama = orig_drive_ojama
        RecognitionPipeline.load_default = orig_load_default

    out_path = OUT_DIR / "detail.log"
    out_path.write_text("\n".join(_detail), encoding="utf-8")
    print(f"[detail] {len(_detail)} 行 (slide_1p=True の回数) -> {out_path}")
    for line in _detail:
        print(line)


if __name__ == "__main__":
    main()
