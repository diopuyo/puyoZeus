"""ネクスト ROI と認識精度の視覚レビュー素材生成 (Phase C-1 補足).

各 frame で:
    - 元 frame に NextDetector の ROI を赤枠で表示
    - 認識結果を文字でオーバーレイ (1P/2P の next + dnext)
    - ROI 部分の拡大画像を画面右下にサムネイル表示

ユーザーが「画面上の実際のネクスト色 vs 認識色」を目視確認できる。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_c_review_next_detection \
        --video data/frames/video_02.mp4 --start-sec 220 --n-frames 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.next_detector import (  # noqa: E402
    NextDetector,
    ROI_1P_NEXT_TOP, ROI_1P_NEXT_BOT,
    ROI_1P_DNEXT_TOP, ROI_1P_DNEXT_BOT,
    ROI_2P_NEXT_TOP, ROI_2P_NEXT_BOT,
    ROI_2P_DNEXT_TOP, ROI_2P_DNEXT_BOT,
)
from src.patch_classifier import CnnPatchClassifier  # noqa: E402

_COLOR_NAME = {
    0: "EM", 1: "RD", 2: "BL", 3: "GR",
    4: "YE", 5: "PU", 9: "OJ", 10: "??",
}
_COLOR_BGR = {
    0: (60, 60, 60), 1: (40, 40, 220), 2: (220, 80, 40),
    3: (40, 200, 40), 4: (40, 220, 240), 5: (200, 40, 200),
    9: (170, 170, 170), 10: (100, 100, 120),
}

_ROIS_LABELED: list[tuple[str, tuple[int, int, int, int]]] = [
    ("1P_N_T", ROI_1P_NEXT_TOP), ("1P_N_B", ROI_1P_NEXT_BOT),
    ("1P_D_T", ROI_1P_DNEXT_TOP), ("1P_D_B", ROI_1P_DNEXT_BOT),
    ("2P_N_T", ROI_2P_NEXT_TOP), ("2P_N_B", ROI_2P_NEXT_BOT),
    ("2P_D_T", ROI_2P_DNEXT_TOP), ("2P_D_B", ROI_2P_DNEXT_BOT),
]


def draw_overlay(
    canvas: np.ndarray, both, time_sec: float, frame_idx: int,
) -> None:
    # ROI 赤枠 + ラベル
    for label, (y1, y2, x1, x2) in _ROIS_LABELED:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 220), 2)
        cv2.putText(
            canvas, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 220), 1, cv2.LINE_AA,
        )

    # 結果テキスト
    p1 = both.p1
    p2 = both.p2

    def fmt(c: int) -> str:
        return _COLOR_NAME.get(int(c), "?")

    cv2.rectangle(canvas, (0, 0), (1920, 90), (10, 10, 10), -1)
    cv2.putText(
        canvas, f"t={time_sec:.2f}s  f={frame_idx:03d}",
        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (240, 240, 240), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"1P next=({fmt(p1.next_top)},{fmt(p1.next_bot)}) "
        f"dnext=({fmt(p1.dnext_top)},{fmt(p1.dnext_bot)})",
        (260, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (240, 240, 240), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"2P next=({fmt(p2.next_top)},{fmt(p2.next_bot)}) "
        f"dnext=({fmt(p2.dnext_top)},{fmt(p2.dnext_bot)})",
        (260, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (240, 240, 240), 1, cv2.LINE_AA,
    )
    agree = "AGREE" if (
        (p1.next_top, p1.next_bot) == (p2.next_top, p2.next_bot)
        and (p1.dnext_top, p1.dnext_bot) == (p2.dnext_top, p2.dnext_bot)
    ) else "DIFFER"
    color = (40, 200, 40) if agree == "AGREE" else (40, 40, 220)
    cv2.putText(
        canvas, f"1P vs 2P: {agree}", (1500, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
    )

    # 結果色 dot を画面下部にまとめ表示
    base_x = 30
    base_y = 1020
    cv2.putText(
        canvas, "Recognized:",
        (base_x, base_y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (200, 200, 200), 1, cv2.LINE_AA,
    )
    side_layouts = [
        ("1P", p1, base_x),
        ("2P", p2, base_x + 480),
    ]
    for side_name, res, x_off in side_layouts:
        for i, (label, color) in enumerate([
            ("nT", res.next_top), ("nB", res.next_bot),
            ("dT", res.dnext_top), ("dB", res.dnext_bot),
        ]):
            cx = x_off + 60 + i * 90
            cy = base_y
            cv2.circle(canvas, (cx, cy), 22, _COLOR_BGR.get(int(color), (100, 100, 100)), -1)
            cv2.circle(canvas, (cx, cy), 23, (0, 0, 0), 2)
            cv2.putText(
                canvas, fmt(color),
                (cx - 16, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, label,
                (cx - 12, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA,
            )
        cv2.putText(
            canvas, side_name,
            (x_off, base_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (240, 240, 240), 2, cv2.LINE_AA,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--n-frames", type=int, default=10)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument(
        "--cnn-path", type=Path,
        default=_ROOT / "models" / "cnn_global_best.pt",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_ROOT / "data" / "review_videos" / "next_review",
    )
    args = parser.parse_args()

    cnn = CnnPatchClassifier()
    if args.cnn_path.exists():
        import torch
        state = torch.load(
            str(args.cnn_path), map_location="cpu", weights_only=True,
        )
        cnn._model.load_state_dict(state)
    try:
        import torch
        if torch.cuda.is_available():
            cnn.to_device("cuda")
    except Exception:
        pass

    next_det = NextDetector(classifier=cnn)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
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
        both = next_det.detect_both(frame)
        canvas = frame.copy()
        draw_overlay(canvas, both, t, i)
        out_path = args.out_dir / (
            f"{args.video.stem}_t{t:07.2f}_f{i:02d}.png"
        )
        cv2.imwrite(str(out_path), canvas)
        print(f"[done] {out_path.name}")
        t += interval
    cap.release()
    print(f"\n[saved] {to_windows_path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
