"""ぷよ認識遅延の視覚デバッグ (B-18).

ユーザー指摘「ぷよ画像が読み取れていない、しばらくして認識される」現象の
原因特定用。指定 time から N frame 連続抽出、各 frame で:

    - cnn_board (ImageReader 生出力 = HSV/CNN 結果)
    - inferred_board (state machine 確定 + 推論)
    - 両者の差分 cell (drift)

を並べた比較画像を出力する。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_b_debug_lag_frames \
        --video data/frames/video_07.mp4 --start-sec 290 --n-frames 20

出力: data/review_videos/debug_lag/v07_t290_f00.png .. v07_t290_f19.png
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

from src.board import (  # noqa: E402
    BOARD_COLS, COLOR_EMPTY, HIDDEN_ROWS, VISIBLE_ROWS,
)
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.phase_b_render_review_video import (  # noqa: E402
    COLOR_BGR, STATE_LABEL, draw_top_hud,
)


def draw_board_with_diff(
    canvas: np.ndarray, cnn_board, inf_board,
    region: BoardRegion, label: str,
) -> None:
    """cnn と inf を一緒に描画 (cnn=三角、inf=丸、差分=黄枠)."""
    cv2.putText(
        canvas, label, (region.x, region.y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA,
    )
    for r in range(HIDDEN_ROWS, HIDDEN_ROWS + VISIBLE_ROWS):
        for c in range(BOARD_COLS):
            cx, cy = region.cell_center(r, c)
            cnn_v = int(cnn_board.get(r, c)) if cnn_board else 0
            inf_v = int(inf_board.get(r, c)) if inf_board else 0
            # cnn = 三角、inf = 丸 で描画
            if cnn_v != COLOR_EMPTY:
                col = COLOR_BGR.get(cnn_v, (200, 200, 200))
                pts = np.array([
                    [cx - 8, cy + 5], [cx + 8, cy + 5], [cx, cy - 8],
                ], np.int32)
                cv2.fillPoly(canvas, [pts], col)
                cv2.polylines(canvas, [pts], True, (0, 0, 0), 1)
            if inf_v != COLOR_EMPTY:
                col = COLOR_BGR.get(inf_v, (200, 200, 200))
                cv2.circle(canvas, (cx, cy), 5, col, -1)
                cv2.circle(canvas, (cx, cy), 6, (0, 0, 0), 1)
            if cnn_v != inf_v:
                # 差分 cell に黄枠
                hw = max(1, int(region.cell_width / 2))
                hh = max(1, int(region.cell_height / 2))
                cv2.rectangle(
                    canvas, (cx - hw, cy - hh), (cx + hw, cy + hh),
                    (0, 220, 220), 2,
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stable-n", type=int, default=6)
    parser.add_argument("--cnn-model", type=Path, default=None)
    parser.add_argument(
        "--out-dir", type=Path,
        default=_ROOT / "data" / "review_videos" / "debug_lag",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=args.stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=args.cnn_model,
        force_in_match=True,
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

        canvas = frame.copy()
        # cnn = triangle、inf = circle、diff = yellow box
        draw_board_with_diff(
            canvas, result.p1.cnn_board, result.p1.inferred_board,
            DEFAULT_P1_REGION,
            f"1P {STATE_LABEL.get(result.p1.state, '?')} drift={result.p1.drift.mismatch_count}",
        )
        draw_board_with_diff(
            canvas, result.p2.cnn_board, result.p2.inferred_board,
            DEFAULT_P2_REGION,
            f"2P {STATE_LABEL.get(result.p2.state, '?')} drift={result.p2.drift.mismatch_count}",
        )
        # legend
        cv2.putText(
            canvas, "tri=cnn raw, circle=inferred, yellow=diff",
            (10, 1070), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (240, 240, 100), 1, cv2.LINE_AA,
        )
        # 上部に simple time/frame
        cv2.rectangle(canvas, (0, 0), (700, 50), (10, 10, 10), -1)
        cv2.putText(
            canvas, f"t={t:.2f}s  f={i:03d}", (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA,
        )

        out_path = args.out_dir / f"{args.video.stem}_t{args.start_sec:.0f}_f{i:02d}.png"
        cv2.imwrite(str(out_path), canvas)
        t += interval
    cap.release()
    print(f"[done] {args.n_frames} debug frames -> {to_windows_path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
