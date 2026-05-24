"""pipeline 内 next_queue が更新されているか確認 (Phase C-1).

RecognitionPipeline.update() を 30 frame 走らせ、毎 frame 内部の
state_machine.context.next_queue と signals.next_pair をログる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402

init_console()

import cv2  # noqa: E402

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

_COLOR_NAME = {
    0: "EM", 1: "RD", 2: "BL", 3: "GR",
    4: "YE", 5: "PU", 9: "OJ", 10: "??",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--n-frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument(
        "--cnn-model", type=Path,
        default=_ROOT / "models" / "cnn_phase_b_v1.pt",
    )
    args = parser.parse_args()

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=2,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=args.cnn_model,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    print(
        f"[init] next_detector loaded: {pipe._next_detector is not None}"  # type: ignore[attr-defined]
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: video open failed: {args.video}")
        return 1

    interval = 1.0 / args.fps
    t = args.start_sec
    for i in range(args.n_frames):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        result = pipe.update(i, t, frame)
        ctx_1p = pipe._sm_1p.context  # type: ignore[attr-defined]
        ctx_2p = pipe._sm_2p.context  # type: ignore[attr-defined]
        q1 = "/".join(
            f"{_COLOR_NAME.get(a, '?')}{_COLOR_NAME.get(b, '?')}"
            for a, b in ctx_1p.next_queue[-3:]
        )
        q2 = "/".join(
            f"{_COLOR_NAME.get(a, '?')}{_COLOR_NAME.get(b, '?')}"
            for a, b in ctx_2p.next_queue[-3:]
        )
        print(
            f"[frame {i:3d}] t={t:7.2f}  "
            f"1P={result.p1.state.value:<10} q={q1:<14}  "
            f"2P={result.p2.state.value:<10} q={q2}"
        )
        t += interval
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
