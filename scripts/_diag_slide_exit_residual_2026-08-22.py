"""修正③根治の残存3件 (t=6706.667/6711.667/6716.667) の原因を特定する
(2026-08-22)。_should_suppress_slide_exit の corroboration 判定に渡された
current_next/start_next を記録し、なぜ「NEXT が変化した」と判定されたのかを
確認する。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
cv2.setNumThreads(1)

import scripts.visualize_advantage_overlay as vao  # noqa: E402
import src.recognition_pipeline as rp  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/slide_suppress_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEG_START = 6664.17
SEG_END = 6725.0
WARMUP = 30.0
WATCH_LO, WATCH_HI = 6700.0, 6720.0

_STATE: dict = {"t": None}
_records: list[str] = []


def main() -> None:
    orig_drive_ojama = vao._drive_ojama

    def patched_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw):
        _STATE["t"] = t
        return orig_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw)

    vao._drive_ojama = patched_drive_ojama

    orig_fn = rp._should_suppress_slide_exit

    def patched(*args, **kwargs):  # type: ignore[no-untyped-def]
        result = orig_fn(*args, **kwargs)
        t = _STATE.get("t")
        if t is not None and WATCH_LO <= t <= WATCH_HI:
            _records.append(
                f"t={t:.3f} suppress={result} "
                f"current_next={kwargs.get('current_next')} "
                f"start_next={kwargs.get('start_next')} "
                f"chain_entry_t={kwargs.get('chain_entry_t')} "
                f"chain_count={kwargs.get('chain_count')}"
            )
        return result

    rp._should_suppress_slide_exit = patched
    try:
        vao.generate(
            video=VIDEO, out=OUT_DIR / "_dummy_residual.mp4",
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
        rp._should_suppress_slide_exit = orig_fn

    # suppress=False (= 抑止されず通過した) の行だけ抽出して表示
    passed = [r for r in _records if "suppress=False" in r]
    out_path = OUT_DIR / "residual_detail.log"
    out_path.write_text("\n".join(_records), encoding="utf-8")
    print(f"[全記録] {len(_records)} 件 -> {out_path}")
    print(f"[通過(suppress=False)] {len(passed)} 件")
    for r in passed:
        print(r)


if __name__ == "__main__":
    main()
