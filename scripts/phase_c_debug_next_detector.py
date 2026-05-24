"""NextDetector 動作確認 (Phase C-1).

レビュー動画で next 表示が出なかった原因を特定する。
1 frame ずつ next_detector.detect_both() を呼んで結果を log 出力。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_c_debug_next_detector \
        --video data/frames/video_02.mp4 --start-sec 220 --n-frames 30
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

from src.next_detector import NextDetector  # noqa: E402
from src.patch_classifier import CnnPatchClassifier  # noqa: E402

_COLOR_NAME = {
    0: "EM", 1: "RD", 2: "BL", 3: "GR",
    4: "YE", 5: "PU", 9: "OJ", 10: "??",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--n-frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--cnn-path", type=Path,
        default=_ROOT / "models" / "cnn_global_best.pt",
    )
    args = parser.parse_args()

    print(f"[init] cnn={args.cnn_path}, exists={args.cnn_path.exists()}")
    cnn = CnnPatchClassifier()
    if args.cnn_path.exists():
        import torch
        state = torch.load(
            str(args.cnn_path), map_location="cpu", weights_only=True,
        )
        cnn._model.load_state_dict(state)
    try:
        if torch_available := __import__("torch").cuda.is_available():
            cnn.to_device("cuda")
            print("[init] cnn -> cuda")
    except Exception:
        pass

    next_det = NextDetector(classifier=cnn)
    print("[init] next_detector ready")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: video open failed: {args.video}")
        return 1

    interval = 1.0 / args.fps
    t = args.start_sec
    last_pair_1p: tuple[int, int] | None = None
    last_pair_2p: tuple[int, int] | None = None
    n_changes_1p = 0
    n_changes_2p = 0
    for i in range(args.n_frames):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        try:
            both = next_det.detect_both(frame)
            p1 = both.p1.next_pair
            p2 = both.p2.next_pair
            d1 = both.p1.dnext_pair
            d2 = both.p2.dnext_pair
            agree = both.colors_agree
        except Exception as e:
            print(f"[frame {i:3d}] t={t:.2f}  ERROR: {e}")
            t += interval
            continue

        p1_str = f"{_COLOR_NAME.get(p1[0], '?')}{_COLOR_NAME.get(p1[1], '?')}"
        p2_str = f"{_COLOR_NAME.get(p2[0], '?')}{_COLOR_NAME.get(p2[1], '?')}"
        d1_str = f"{_COLOR_NAME.get(d1[0], '?')}{_COLOR_NAME.get(d1[1], '?')}"
        d2_str = f"{_COLOR_NAME.get(d2[0], '?')}{_COLOR_NAME.get(d2[1], '?')}"
        change_1p = p1 != last_pair_1p and last_pair_1p is not None
        change_2p = p2 != last_pair_2p and last_pair_2p is not None
        if change_1p:
            n_changes_1p += 1
        if change_2p:
            n_changes_2p += 1
        marker_1p = " *" if change_1p else "  "
        marker_2p = " *" if change_2p else "  "
        print(
            f"[frame {i:3d}] t={t:7.2f}  "
            f"1P next={p1_str} dnext={d1_str}{marker_1p}  "
            f"2P next={p2_str} dnext={d2_str}{marker_2p}  "
            f"agree={agree}"
        )
        last_pair_1p = p1
        last_pair_2p = p2
        t += interval
    cap.release()

    print()
    print(
        f"[summary] {args.n_frames} frames, 1P changes={n_changes_1p}, "
        f"2P changes={n_changes_2p}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
