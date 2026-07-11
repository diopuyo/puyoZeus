"""STEP0: STABLE フレームにおける next_pair / dnext_pair 取得率を計測する。

使い方:
    PYTHONPATH=. python scripts/_step0_next_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = Path("data/frames/video_124_4min.mp4")
TARGET_W, TARGET_H = 1920, 1080
MAX_SEC = 120.0  # 最初の2分だけ計測


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {VIDEO}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(MAX_SEC * fps))

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )

    stable_count = {"1P": 0, "2P": 0}
    next_ok = {"1P": 0, "2P": 0}
    dnext_ok = {"1P": 0, "2P": 0}
    both_ok = {"1P": 0, "2P": 0}

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)

        for side_res, lbl in [(result.p1, "1P"), (result.p2, "2P")]:
            if side_res.state != BoardState.STABLE:
                continue
            stable_count[lbl] += 1
            if side_res.next_pair is not None:
                next_ok[lbl] += 1
            if side_res.dnext_pair is not None:
                dnext_ok[lbl] += 1
            if side_res.next_pair is not None and side_res.dnext_pair is not None:
                both_ok[lbl] += 1

    cap.release()

    print("=== STEP0: next/dnext 取得率 (STABLE フレーム中) ===")
    for lbl in ("1P", "2P"):
        s = stable_count[lbl]
        n = next_ok[lbl]
        d = dnext_ok[lbl]
        b = both_ok[lbl]
        print(
            f"{lbl}: stable={s}  "
            f"next_ok={n} ({n/max(1,s)*100:.1f}%)  "
            f"dnext_ok={d} ({d/max(1,s)*100:.1f}%)  "
            f"both_ok={b} ({b/max(1,s)*100:.1f}%)"
        )


if __name__ == "__main__":
    main()
