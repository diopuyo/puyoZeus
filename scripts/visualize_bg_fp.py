"""bg_fp (背景指紋) を画像化してユーザー目視できるようにする (cycle 32 ポイント A).

各 cell の (H, S, V) を BGR に逆変換して 6x12 の board grid として描画。
1P と 2P を横並びで 1 枚に。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.visualize_bg_fp \
        --video data/frames/video_29.mp4 \
        --output data/seed_review/v29_bg_fp.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.background_fingerprint import BackgroundFingerprint
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.board import VISIBLE_ROWS


CELL_PX = 64


def fp_to_image(bg_fp: BackgroundFingerprint) -> np.ndarray:
    """bg_fp の各 cell HSV を BGR の正方形 patch に展開して board grid 化."""
    cols = 6
    rows = 12  # visible rows
    img = np.zeros((rows * CELL_PX, cols * CELL_PX, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            cell = bg_fp.cells[r][c]
            hsv = np.array(
                [[[int(cell.h), int(cell.s), int(cell.v)]]], dtype=np.uint8,
            )
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            img[
                r * CELL_PX:(r + 1) * CELL_PX,
                c * CELL_PX:(c + 1) * CELL_PX,
            ] = bgr
            # grid border
            cv2.rectangle(
                img,
                (c * CELL_PX, r * CELL_PX),
                ((c + 1) * CELL_PX - 1, (r + 1) * CELL_PX - 1),
                (40, 40, 40), 1,
            )
            # HSV テキスト (小さく)
            text = f"{int(cell.h)},{int(cell.s)},{int(cell.v)}"
            cv2.putText(
                img, text,
                (c * CELL_PX + 2, r * CELL_PX + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (255, 255, 255), 1,
            )
    return img


def find_match_start_frame(
    cap: cv2.VideoCapture, max_search: int = 600, skip_initial: int = 90,
) -> np.ndarray | None:
    """動画から「ほぼ空盤面」 frame を探す.

    開始 N frame skip (= タイトル / 入場画面) 後、 ImageReader で puyo_count
    が最小の frame を採用 (= 真の空盤面が見つからない場合の fallback)。
    """
    from src.image_reader import ImageReader
    reader = ImageReader()
    best_frame = None
    best_count = 9999
    fi = 0
    while fi < max_search:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080))
        if fi < skip_initial:
            fi += 1
            continue
        try:
            b1, b2 = reader.read_both_boards(frame)
            total = b1.count_puyos() + b2.count_puyos()
            if total < best_count:
                best_count = total
                best_frame = frame.copy()
            if total == 0:
                print(f"[match-start] frame={fi} (truly empty)")
                return frame
        except Exception:
            pass
        fi += 1
    print(
        f"[warn] no truly-empty frame, using best with {best_count} puyos",
    )
    return best_frame


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-search", type=int, default=1800)
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[error] cannot open {args.video}")
        return 1
    frame = find_match_start_frame(cap, max_search=args.max_search)
    cap.release()
    if frame is None:
        print("[error] no frame available")
        return 1

    bg_fp_p1 = BackgroundFingerprint.capture(
        frame,
        DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
        DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height,
    )
    bg_fp_p2 = BackgroundFingerprint.capture(
        frame,
        DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
        DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height,
    )
    img_p1 = fp_to_image(bg_fp_p1)
    img_p2 = fp_to_image(bg_fp_p2)
    sep = np.full((img_p1.shape[0], 32, 3), 80, dtype=np.uint8)
    combined = np.hstack([img_p1, sep, img_p2])
    label = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        label, f"bg_fp from {args.video.name} (left=1P / right=2P, HSV text)",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    final = np.vstack([label, combined])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), final)
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
