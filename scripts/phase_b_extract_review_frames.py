"""Phase B レビュー用代表 frame 抽出 (B-16 補助).

オーバーレイ動画から各 state (STABLE / TSUMO_FALL / CHAIN / OJAMA_FALL)
の代表 frame を 1 枚ずつ抽出して PNG 出力する。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_b_extract_review_frames \
        --videos 1,7,13 --duration 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402

from src.board_state_machine import BoardState  # noqa: E402
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.phase_b_render_review_video import (  # noqa: E402
    draw_board_overlay, draw_top_hud, get_match1, select_cnn_model,
)

# どの state を抽出するか (1P でその state になった最初の frame を採用)
_TARGET_STATES = [
    BoardState.STABLE,
    BoardState.TSUMO_FALL,
    BoardState.CHAIN,
    BoardState.OJAMA_FALL,
]


def render_frame_with_overlay(
    frame, result,
):
    """1 frame に HUD + 盤面 dot を描画."""
    canvas = frame.copy()
    draw_board_overlay(canvas, result.p1.confirmed_board, DEFAULT_P1_REGION)
    draw_board_overlay(canvas, result.p2.confirmed_board, DEFAULT_P2_REGION)
    draw_top_hud(
        canvas, result.time_sec, result.frame_idx,
        result.p1.state, result.p2.state,
        result.p1.score, result.p2.score,
        result.p1.drift.mismatch_count,
        result.p2.drift.mismatch_count,
    )
    return canvas


def extract_video(
    video_id: int, start_sec: float, end_sec: float,
    fps_sample: float, stable_n: int, smoothing_n: int,
    cnn_model: Path | None, out_dir: Path,
) -> int:
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        return 0
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=cnn_model,
        temporal_smoothing=smoothing_n,
        force_in_match=True,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[BoardState, str] = {}
    high_drift_captured: bool = False

    interval = 1.0 / fps_sample
    t = start_sec
    frame_idx = 0
    while t < end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        result = pipe.update(frame_idx, t, frame)

        # state ごとに最初の出現を保存
        for state in _TARGET_STATES:
            if state in captured:
                continue
            if result.p1.state == state or result.p2.state == state:
                canvas = render_frame_with_overlay(frame, result)
                fname = (
                    f"v{video_id:02d}_t{t:07.2f}_{state.value}.png"
                )
                fp = out_dir / fname
                cv2.imwrite(str(fp), canvas)
                captured[state] = str(fp)
                print(f"[done] {fname}")

        # drift > 5 の最初の 1 frame を保存
        if not high_drift_captured:
            if result.p1.drift.mismatch_count >= 5 \
                    or result.p2.drift.mismatch_count >= 5:
                canvas = render_frame_with_overlay(frame, result)
                fname = f"v{video_id:02d}_t{t:07.2f}_drift_high.png"
                fp = out_dir / fname
                cv2.imwrite(str(fp), canvas)
                print(f"[done] {fname}")
                high_drift_captured = True

        frame_idx += 1
        t += interval
    cap.release()
    return len(captured) + (1 if high_drift_captured else 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=str, default="1,7,13")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stable-n", type=int, default=6)
    parser.add_argument("--smoothing-n", type=int, default=3)
    parser.add_argument("--per-video-model", action="store_true", default=True)
    parser.add_argument("--cnn-model", type=Path, default=None)
    parser.add_argument(
        "--out-dir", type=Path,
        default=_ROOT / "data" / "review_videos" / "frames",
    )
    args = parser.parse_args()

    target_ids = [int(s) for s in args.videos.split(",") if s.strip()]
    total = 0
    for vid in target_ids:
        m = get_match1(vid)
        if m is None:
            continue
        start = m[0]
        end = min(m[1], start + args.duration)
        cnn_model = select_cnn_model(
            vid, args.per_video_model, args.cnn_model,
        )
        # per-video smoothing も PV2 から取得
        if args.per_video_model:
            from src.per_video_model_selector import select_phase_b_smoothing
            smoothing_n = select_phase_b_smoothing(vid)
        else:
            smoothing_n = args.smoothing_n
        n = extract_video(
            video_id=vid, start_sec=start, end_sec=end,
            fps_sample=args.fps, stable_n=args.stable_n,
            smoothing_n=smoothing_n,
            cnn_model=cnn_model, out_dir=args.out_dir,
        )
        total += n
    print(f"\n[summary] {total} review frames -> "
          f"{to_windows_path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
