"""STEP A-2: 全区間での drain フリッカー集計。

video_124_4min.mp4 全体 (約240s) を SAMPLE_EVERY=3 で処理し、
TSUMO_FALL→STABLE 遷移回数 vs tsumo_count 増分累積 を集計する。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402
init_console()

import cv2  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO_PATH = Path("data/frames/video_124_4min.mp4")
END_SEC = 0.0   # 0 = 全体
DEFAULT_FPS = 30.0
SAMPLE_EVERY = 3


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] cannot open {VIDEO_PATH}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if END_SEC > 0:
        n_frames = min(total_frames, int(END_SEC * fps))
    else:
        n_frames = total_frames
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_t = n_frames / fps
    print(f"[input] {VIDEO_PATH} {src_w}x{src_h} fps={fps:.1f} n_frames={n_frames} ({max_t:.0f}s)")

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    pipeline.set_video_id("video_124_4min")

    ojama_tracker = OjamaAccountingTracker()
    ojama_tracker.reset()

    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU
    prev_tsumo_p1 = 0
    prev_tsumo_p2 = 0

    tsumo_fall_to_stable_p1 = 0
    tsumo_fall_to_stable_p2 = 0
    tsumo_count_drain_p1 = 0
    tsumo_count_drain_p2 = 0

    snap = ojama_tracker.get_snapshot(0.0)

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t_sec = fi / fps
        if fi % SAMPLE_EVERY != 0:
            continue

        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

        result = pipeline.update(fi, t_sec, frame)
        p1 = result.p1
        p2 = result.p2

        # 旧 drain 判定
        old_drain_p1 = (prev_state_p1 == BoardState.TSUMO_FALL and p1.state == BoardState.STABLE)
        old_drain_p2 = (prev_state_p2 == BoardState.TSUMO_FALL and p2.state == BoardState.STABLE)

        ojama_tracker.on_state_transition("p1", prev_state_p1, p1.state, p1.score, t_sec)
        ojama_tracker.on_state_transition("p2", prev_state_p2, p2.state, p2.score, t_sec)

        if old_drain_p1:
            ojama_tracker.on_tsumo_settled("p1", t_sec)
            tsumo_fall_to_stable_p1 += 1
        if old_drain_p2:
            ojama_tracker.on_tsumo_settled("p2", t_sec)
            tsumo_fall_to_stable_p2 += 1

        snap = ojama_tracker.get_snapshot(t_sec)

        cur_tc_p1 = pipeline.tsumo_count("1P")
        cur_tc_p2 = pipeline.tsumo_count("2P")
        delta_p1 = max(0, cur_tc_p1 - prev_tsumo_p1)
        delta_p2 = max(0, cur_tc_p2 - prev_tsumo_p2)
        if delta_p1 > 0:
            tsumo_count_drain_p1 += delta_p1
        if delta_p2 > 0:
            tsumo_count_drain_p2 += delta_p2

        prev_state_p1 = p1.state
        prev_state_p2 = p2.state
        prev_tsumo_p1 = cur_tc_p1
        prev_tsumo_p2 = cur_tc_p2

        if fi % (300 * SAMPLE_EVERY) == 0:
            print(
                f"  t={t_sec:.0f}s  tf_p1={tsumo_fall_to_stable_p1} tf_p2={tsumo_fall_to_stable_p2}"
                f"  tc_p1={tsumo_count_drain_p1} tc_p2={tsumo_count_drain_p2}",
                flush=True,
            )

    cap.release()

    print(f"\n=== 集計 ({n_frames/fps:.0f}s) ===")
    print(f"  旧drain(TSUMO_FALL→STABLE) 1P={tsumo_fall_to_stable_p1}  2P={tsumo_fall_to_stable_p2}")
    print(f"  tsumo_count増分累積         1P={tsumo_count_drain_p1}  2P={tsumo_count_drain_p2}")
    if tsumo_count_drain_p1 > 0:
        ratio = tsumo_fall_to_stable_p1 / tsumo_count_drain_p1
        print(f"  1P倍率: {tsumo_fall_to_stable_p1}/{tsumo_count_drain_p1} = {ratio:.2f}x")
    if tsumo_count_drain_p2 > 0:
        ratio = tsumo_fall_to_stable_p2 / tsumo_count_drain_p2
        print(f"  2P倍率: {tsumo_fall_to_stable_p2}/{tsumo_count_drain_p2} = {ratio:.2f}x")
    print(f"最終 forecast: 1P={snap.forecast_p1} 2P={snap.forecast_p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
