"""STEP A-3: t=150〜200s の区間でforecast_p1が121になる過程を追う。"""
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
START_SEC = 150.0
END_SEC = 242.0
DEFAULT_FPS = 30.0
SAMPLE_EVERY = 3


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR]", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total_frames, int(END_SEC * fps))
    print(f"[range] t={START_SEC}-{END_SEC}s SAMPLE_EVERY={SAMPLE_EVERY}")

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

    snap = ojama_tracker.get_snapshot(0.0)
    prev_fc_p1 = -1
    prev_fc_p2 = -1

    # ---- 先頭 START_SEC まで読み飛ばし ----
    # ただし pipeline.update は呼び続けないと state machine が正しく動かないため
    # START_SEC まで全フレーム update する (read-skip は不可)
    print(f"[skip] 全フレームupdate必須のため0〜{START_SEC}sを処理中...")
    start_frame = int(START_SEC * fps)

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

        old_drain_p1 = (prev_state_p1 == BoardState.TSUMO_FALL and p1.state == BoardState.STABLE)
        old_drain_p2 = (prev_state_p2 == BoardState.TSUMO_FALL and p2.state == BoardState.STABLE)
        ojama_tracker.on_state_transition("p1", prev_state_p1, p1.state, p1.score, t_sec)
        ojama_tracker.on_state_transition("p2", prev_state_p2, p2.state, p2.score, t_sec)
        if old_drain_p1:
            ojama_tracker.on_tsumo_settled("p1", t_sec)
        if old_drain_p2:
            ojama_tracker.on_tsumo_settled("p2", t_sec)
        snap = ojama_tracker.get_snapshot(t_sec)

        prev_state_p1 = p1.state
        prev_state_p2 = p2.state

        # START_SEC 以降だけ詳細出力
        if t_sec < START_SEC:
            continue

        if fi == start_frame:
            print(f"\n[at {START_SEC}s] fc_p1={snap.forecast_p1} fc_p2={snap.forecast_p2}")
            print(f"\n{'t_sec':>8} {'1P_state':>14} {'2P_state':>14} "
                  f"{'tc1P':>5} {'tc2P':>5} "
                  f"{'dr1P':>5} {'dr2P':>5} "
                  f"{'fc1P':>5} {'fc2P':>5}")
            print("-" * 90)

        fc_p1 = snap.forecast_p1
        fc_p2 = snap.forecast_p2

        if (
            fc_p1 != prev_fc_p1 or fc_p2 != prev_fc_p2
            or old_drain_p1 or old_drain_p2
            or p1.state != prev_state_p1 or p2.state != prev_state_p2
        ):
            print(
                f"{t_sec:8.2f} {p1.state.value:>14} {p2.state.value:>14} "
                f"{pipeline.tsumo_count('1P'):5d} {pipeline.tsumo_count('2P'):5d} "
                f"{'Y' if old_drain_p1 else '-':>5} {'Y' if old_drain_p2 else '-':>5} "
                f"{fc_p1:5d} {fc_p2:5d}"
            )
        prev_fc_p1 = fc_p1
        prev_fc_p2 = fc_p2

    cap.release()
    print(f"\n最終: fc_p1={snap.forecast_p1} fc_p2={snap.forecast_p2}")  # type: ignore[possibly-undefined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
