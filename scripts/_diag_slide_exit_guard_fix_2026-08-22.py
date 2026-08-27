"""修正③根治 (enable_slide_exit_min_display_guard) の短区間確認 (2026-08-22)。

t=6700-6720 (1Pの15連鎖区間) で:
  - guard OFF (既定): 従来通り断片化が再現すること (陽性対照)
  - guard ON: 断片化が解消され、最終的な chain_count が正しく積み上がること

RecognitionPipeline._stash_and_clear_active_chain をmonkeypatchし、1P側の
active_chain がクリアされた瞬間 (= 断片化イベント) を time_sec と
chain_count 付きで記録する。
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
OUT_DIR = PROJECT_ROOT / "logs/slide_suppress_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEG_START = 6664.17
SEG_END = 6725.0
WARMUP = 30.0
WATCH_LO, WATCH_HI = 6700.0, 6720.0

_STATE: dict = {"t": None}


def run(guard: bool) -> list[tuple[float, int]]:
    """guard ON/OFF で1回走査し、1P側の断片化イベント (t, chain_count) 一覧を返す。"""
    events: list[tuple[float, int]] = []
    _STATE["t"] = None

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
            events.append((t, int(self._active_chain_1p.chain_count)))
        return orig_stash(self, side)

    RecognitionPipeline._stash_and_clear_active_chain = patched_stash
    try:
        vao.generate(
            video=VIDEO, out=OUT_DIR / f"_dummy_guard{int(guard)}.mp4",
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
            enable_slide_exit_min_display_guard=guard,
            force_in_match=False,
        )
    finally:
        vao._drive_ojama = orig_drive_ojama
        RecognitionPipeline._stash_and_clear_active_chain = orig_stash
    return events


def main() -> None:
    print("=== guard OFF (陽性対照: 断片化が再現するはず) ===")
    off_events = run(guard=False)
    for t, cc in off_events:
        print(f"  [OFF] t={t:.3f} chain_count(断片)={cc}")
    print(f"OFF 断片化イベント数: {len(off_events)}")

    print("\n=== guard ON (断片化が解消するはず) ===")
    on_events = run(guard=True)
    for t, cc in on_events:
        print(f"  [ON]  t={t:.3f} chain_count(断片)={cc}")
    print(f"ON  断片化イベント数: {len(on_events)}")

    out_path = OUT_DIR / "result.txt"
    lines = [
        f"OFF events: {off_events}",
        f"ON events: {on_events}",
        f"OFF count={len(off_events)} ON count={len(on_events)}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[結果] -> {out_path}")


if __name__ == "__main__":
    main()
