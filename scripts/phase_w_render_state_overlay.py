"""W: 動画下部に GameState (盤面/ネクスト/お邪魔/スコア) を表示した評価用動画。

入力: 動画 + 開始/終了秒
出力: 1920 x (1080 + パネル高) の mp4。下部パネルに各フレームの認識結果を表示。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_render_state_overlay \
        --video data/frames/video_01.mp4 --start 186 --end 256 \
        --out data/verify/state_overlay_v01_m1.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
)
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.state_pipeline import StatePipeline


# 下部パネル高さ
PANEL_H: int = 280
# パネル背景色 (BGR)
PANEL_BG: tuple[int, int, int] = (30, 30, 30)
# テキスト色
TEXT_COLOR: tuple[int, int, int] = (240, 240, 240)
TEXT_LABEL: tuple[int, int, int] = (180, 180, 180)

COLOR_LABEL: dict[int, str] = {
    COLOR_EMPTY: ".", COLOR_RED: "R", COLOR_BLUE: "B",
    COLOR_GREEN: "G", COLOR_YELLOW: "Y", COLOR_PURPLE: "P",
    COLOR_OJAMA: "O", COLOR_UNKNOWN: "?",
}
COLOR_BGR: dict[int, tuple[int, int, int]] = {
    COLOR_EMPTY: (50, 50, 50),
    COLOR_RED: (40, 40, 220),
    COLOR_BLUE: (220, 80, 40),
    COLOR_GREEN: (40, 200, 40),
    COLOR_YELLOW: (40, 220, 240),
    COLOR_PURPLE: (200, 40, 200),
    COLOR_OJAMA: (180, 180, 180),
    COLOR_UNKNOWN: (90, 90, 130),
}


def overlay_field_on_main(
    frame: np.ndarray, board, region,
) -> None:
    """上半分の動画にフィールド認識オーバーレイ (in-place)。"""
    for vrow in range(VISIBLE_ROWS):
        for col in range(BOARD_COLS):
            row = vrow + HIDDEN_ROWS
            color = int(board.get(row, col))
            if color == COLOR_EMPTY:
                continue
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            bgr = COLOR_BGR.get(color, (60, 60, 60))
            sub = frame[y1:y2, x1:x2].astype(np.float32)
            tint = np.full_like(sub, bgr, dtype=np.float32)
            frame[y1:y2, x1:x2] = (sub * 0.6 + tint * 0.4).astype(np.uint8)
            label = COLOR_LABEL.get(color, "?")
            cx = (x1 + x2) // 2 - 8
            cy = (y1 + y2) // 2 + 6
            cv2.putText(
                frame, label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                2, cv2.LINE_AA,
            )


def render_board_mini(
    panel: np.ndarray, board, x: int, y: int,
    cell: int = 14,
) -> None:
    """パネル上に盤面ミニビュー (12x6) を描画。"""
    for vrow in range(VISIBLE_ROWS):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            color = int(board.get(row, col))
            cx1 = x + col * cell
            cy1 = y + vrow * cell
            cx2 = cx1 + cell - 1
            cy2 = cy1 + cell - 1
            bgr = COLOR_BGR.get(color, (50, 50, 50))
            cv2.rectangle(panel, (cx1, cy1), (cx2, cy2), bgr, -1)
            cv2.rectangle(panel, (cx1, cy1), (cx2, cy2), (20, 20, 20), 1)


def render_pair(
    panel: np.ndarray, pair, x: int, y: int, cell: int = 24,
) -> None:
    """パネル上に next/dnext ペア (top, bot 縦) を描画。"""
    for i, color in enumerate([pair[0] if pair else 0, pair[1] if pair else 0]):
        cx1 = x
        cy1 = y + i * cell
        cx2 = x + cell - 1
        cy2 = y + (i + 1) * cell - 1
        bgr = COLOR_BGR.get(int(color), (50, 50, 50))
        cv2.rectangle(panel, (cx1, cy1), (cx2, cy2), bgr, -1)
        cv2.rectangle(panel, (cx1, cy1), (cx2, cy2), (200, 200, 200), 1)


def put_text(
    img: np.ndarray, text: str, x: int, y: int,
    color: tuple[int, int, int] = TEXT_COLOR,
    scale: float = 0.55, thickness: int = 1,
) -> None:
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def make_panel(state, t_sec: float) -> np.ndarray:
    """state を可視化したパネル画像 (1920 x PANEL_H)。"""
    panel = np.full((PANEL_H, 1920, 3), PANEL_BG, dtype=np.uint8)

    # 列 1 (左): 1P 盤面 + ネクスト
    put_text(panel, f"t={t_sec:.1f}s", 12, 22, TEXT_LABEL, 0.55)
    put_text(panel, "1P field", 12, 50, TEXT_LABEL, 0.5)
    render_board_mini(panel, state.board_p1, x=12, y=58, cell=14)
    # ネクスト/ダブルネクスト
    put_text(panel, "next", 110, 50, TEXT_LABEL, 0.5)
    render_pair(panel, state.next_p1 or (0, 0), x=110, y=60, cell=24)
    put_text(panel, "dnext", 145, 50, TEXT_LABEL, 0.5)
    render_pair(panel, state.dnext_p1 or (0, 0), x=145, y=60, cell=24)

    # 列 2 (中央): スコア / お邪魔 / 状態
    cx = 240
    put_text(panel, "1P", cx, 22, TEXT_LABEL, 0.55)
    s1 = "?" if state.score_p1 is None else f"{state.score_p1:,}"
    put_text(panel, f"score: {s1}", cx, 50, TEXT_COLOR, 0.6)
    put_text(panel, f"  conf: {state.score_confidence_p1:.2f}",
             cx, 70, TEXT_LABEL, 0.45)
    put_text(panel, f"pending ojama: {state.pending_ojama_p1}",
             cx, 100, TEXT_COLOR, 0.6)

    cx2 = 460
    put_text(panel, "2P", cx2, 22, TEXT_LABEL, 0.55)
    s2 = "?" if state.score_p2 is None else f"{state.score_p2:,}"
    put_text(panel, f"score: {s2}", cx2, 50, TEXT_COLOR, 0.6)
    put_text(panel, f"  conf: {state.score_confidence_p2:.2f}",
             cx2, 70, TEXT_LABEL, 0.45)
    put_text(panel, f"pending ojama: {state.pending_ojama_p2}",
             cx2, 100, TEXT_COLOR, 0.6)

    # 状態フラグ
    put_text(panel, f"telop: {state.is_telop_visible}",
             cx, 140, TEXT_LABEL, 0.5)
    put_text(panel, f"match-end-locked: {state.is_match_end_locked}",
             cx, 160, TEXT_LABEL, 0.5)

    # 列 3 (右側): 2P 盤面 + ネクスト
    rx = 1700
    put_text(panel, "2P field", rx, 50, TEXT_LABEL, 0.5)
    render_board_mini(panel, state.board_p2, x=rx, y=58, cell=14)
    put_text(panel, "next", rx + 95, 50, TEXT_LABEL, 0.5)
    render_pair(panel, state.next_p2 or (0, 0), x=rx + 95, y=60, cell=24)
    put_text(panel, "dnext", rx + 130, 50, TEXT_LABEL, 0.5)
    render_pair(panel, state.dnext_p2 or (0, 0), x=rx + 130, y=60, cell=24)

    # 区切り線
    cv2.line(panel, (0, 0), (1920, 0), (60, 60, 60), 2)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--detect-interval", type=float, default=0.4,
        help="StatePipeline.extract を呼ぶ間隔 (秒)。残りは前回の結果を再表示",
    )
    parser.add_argument(
        "--bg-fp-time", type=float, default=-1.0,
        help="試合開始秒。指定すると BG FP を取得 (背景打消し)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"video open failed: {args.video}")
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    pipeline = StatePipeline()
    # BG FP 設定 (背景打消し)
    if args.bg_fp_time >= 0:
        ok = pipeline.set_background_fingerprints_from_video(
            cap, args.bg_fp_time,
        )
        if ok:
            print(f"BG FP set at t={args.bg_fp_time}")
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    pipeline.reset(match_start_sec=args.start)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path), fourcc, src_fps, (1920, 1080 + PANEL_H),
    )
    if not writer.isOpened():
        print("output writer failed")
        return 1

    detect_interval_frames = max(1, int(src_fps * args.detect_interval))
    n_frames = int((args.end - args.start) * src_fps)
    frame_idx = 0
    state = None
    panel = np.full((PANEL_H, 1920, 3), PANEL_BG, dtype=np.uint8)
    print(f"render: {n_frames} frames, detect every {detect_interval_frames} frames")
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )

        t_sec = args.start + frame_idx / src_fps
        if frame_idx % detect_interval_frames == 0:
            try:
                state = pipeline.extract(frame, t_sec=t_sec)
                panel = make_panel(state, t_sec)
            except Exception:
                pass

        # 上部: 元動画 + フィールド overlay
        if state is not None:
            overlay_field_on_main(frame, state.board_p1, DEFAULT_P1_REGION)
            overlay_field_on_main(frame, state.board_p2, DEFAULT_P2_REGION)

        # 結合
        out = np.vstack([frame, panel])
        writer.write(out)
        if frame_idx % 600 == 0:
            print(f"  progress {frame_idx}/{n_frames}")
        frame_idx += 1

    writer.release()
    cap.release()
    print(f"\n[OK] {to_windows_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
