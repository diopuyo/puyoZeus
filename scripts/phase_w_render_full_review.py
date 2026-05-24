"""W5: 1 試合動画の下部に state + 評価値 + 16 指標を表示する総合レビュー動画。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_render_full_review \
        --video data/frames/video_05.mp4 --start 320 --end 390 \
        --winner 1P --out data/verify/full_review_v05_m1.mp4
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
    BOARD_COLS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW,
    HIDDEN_ROWS, VISIBLE_ROWS,
)
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.indicators import (
    ALL_INDICATOR_NAMES,
    EXTRA_INDICATOR_NAMES,
    IndicatorCalculator,
)
from src.state_features import encode_state
from src.state_pipeline import StatePipeline
from src.win_predictor import WinPredictorMLP


# レイアウト
PANEL_H: int = 540
PANEL_BG: tuple[int, int, int] = (30, 30, 30)
TEXT_COLOR: tuple[int, int, int] = (240, 240, 240)
TEXT_LABEL: tuple[int, int, int] = (180, 180, 180)
COLOR_BGR_MAP: dict[int, tuple[int, int, int]] = {
    COLOR_EMPTY: (50, 50, 50),
    COLOR_RED: (40, 40, 220),
    COLOR_BLUE: (220, 80, 40),
    COLOR_GREEN: (40, 200, 40),
    COLOR_YELLOW: (40, 220, 240),
    COLOR_PURPLE: (200, 40, 200),
    COLOR_OJAMA: (180, 180, 180),
    COLOR_UNKNOWN: (90, 90, 130),
}
COLOR_LABEL_MAP: dict[int, str] = {
    COLOR_EMPTY: ".", COLOR_RED: "R", COLOR_BLUE: "B",
    COLOR_GREEN: "G", COLOR_YELLOW: "Y", COLOR_PURPLE: "P",
    COLOR_OJAMA: "O", COLOR_UNKNOWN: "?",
}


def put(img, text, x, y, color=TEXT_COLOR, scale=0.5, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def render_board_mini(panel, board, x, y, cell=12):
    for vrow in range(VISIBLE_ROWS):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            color = int(board.get(row, col))
            cx1 = x + col * cell
            cy1 = y + vrow * cell
            cx2 = cx1 + cell - 1
            cy2 = cy1 + cell - 1
            cv2.rectangle(panel, (cx1, cy1), (cx2, cy2),
                          COLOR_BGR_MAP.get(color, (50, 50, 50)), -1)
            cv2.rectangle(panel, (cx1, cy1), (cx2, cy2), (20, 20, 20), 1)


def render_pair(panel, pair, x, y, cell=20):
    for i, color in enumerate([
        pair[0] if pair else 0,
        pair[1] if pair else 0,
    ]):
        cx1, cy1 = x, y + i * cell
        cx2, cy2 = x + cell - 1, y + (i + 1) * cell - 1
        cv2.rectangle(panel, (cx1, cy1), (cx2, cy2),
                      COLOR_BGR_MAP.get(int(color), (50, 50, 50)), -1)
        cv2.rectangle(panel, (cx1, cy1), (cx2, cy2), (200, 200, 200), 1)


def render_winrate_bar(panel, prob, x, y, w=600, h=24):
    """評価値 P(1P win) を bar で。"""
    cv2.rectangle(panel, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (200, 200, 200), 1)
    fill = int(w * float(prob))
    color = (220, 80, 40) if prob >= 0.5 else (40, 40, 220)  # 1P=blue, 2P=red
    cv2.rectangle(panel, (x, y), (x + fill, y + h), color, -1)
    # 中央線
    cv2.line(panel, (x + w // 2, y), (x + w // 2, y + h),
             (255, 255, 255), 1)
    label = f"P(1P win) = {prob:.3f}"
    put(panel, label, x + w + 10, y + 18, TEXT_COLOR, 0.55, 1)


# 16 指標 (8 + 8)
ALL_16: tuple[str, ...] = ALL_INDICATOR_NAMES + tuple(
    n for n in EXTRA_INDICATOR_NAMES if n not in ALL_INDICATOR_NAMES
)


def render_indicators(panel, p1_set, p2_set, x, y):
    """16 指標を 1P/2P 縦並びで表示。"""
    line_h = 18
    put(panel, "indicator", x, y, TEXT_LABEL, 0.45, 1)
    put(panel, "1P", x + 240, y, TEXT_LABEL, 0.45, 1)
    put(panel, "2P", x + 290, y, TEXT_LABEL, 0.45, 1)
    for i, name in enumerate(ALL_16):
        yy = y + 12 + (i + 1) * line_h
        try:
            v1 = p1_set.score_of(name) if name in p1_set.results else getattr(p1_set, name, 0.0)
        except Exception:
            v1 = 0.0
        try:
            v2 = p2_set.score_of(name) if name in p2_set.results else getattr(p2_set, name, 0.0)
        except Exception:
            v2 = 0.0
        put(panel, name, x, yy, TEXT_COLOR, 0.42, 1)
        put(panel, f"{v1:.2f}", x + 240, yy, TEXT_COLOR, 0.42, 1)
        put(panel, f"{v2:.2f}", x + 290, yy, TEXT_COLOR, 0.42, 1)


def make_panel(state, p1_set, p2_set, prob_1p, t_sec, winner_actual=None):
    panel = np.full((PANEL_H, 1920, 3), PANEL_BG, dtype=np.uint8)

    # row 1: time + winrate bar
    put(panel, f"t={t_sec:.1f}s", 12, 22, TEXT_LABEL, 0.55)
    if winner_actual:
        put(panel, f"actual winner: {winner_actual}", 110, 22, TEXT_LABEL, 0.5)
    render_winrate_bar(panel, prob_1p, x=600, y=8, w=600, h=24)

    # 1P 盤面 + ネクスト
    put(panel, "1P field", 12, 56, TEXT_LABEL, 0.5)
    render_board_mini(panel, state.board_p1, x=12, y=64, cell=12)
    put(panel, "next", 95, 56, TEXT_LABEL, 0.45)
    render_pair(panel, state.next_p1 or (0, 0), x=95, y=64, cell=20)
    put(panel, "dnext", 125, 56, TEXT_LABEL, 0.45)
    render_pair(panel, state.dnext_p1 or (0, 0), x=125, y=64, cell=20)

    # 1P score / ojama
    cx = 200
    put(panel, "1P", cx, 56, TEXT_LABEL, 0.5)
    s1 = "?" if state.score_p1 is None else f"{state.score_p1:,}"
    put(panel, f"score: {s1}", cx, 80, TEXT_COLOR, 0.55)
    put(panel, f"  conf: {state.score_confidence_p1:.2f}",
        cx, 100, TEXT_LABEL, 0.45)
    put(panel, f"pending O: {state.pending_ojama_p1}",
        cx, 120, TEXT_COLOR, 0.55)

    # 2P
    cx2 = 400
    put(panel, "2P", cx2, 56, TEXT_LABEL, 0.5)
    s2 = "?" if state.score_p2 is None else f"{state.score_p2:,}"
    put(panel, f"score: {s2}", cx2, 80, TEXT_COLOR, 0.55)
    put(panel, f"  conf: {state.score_confidence_p2:.2f}",
        cx2, 100, TEXT_LABEL, 0.45)
    put(panel, f"pending O: {state.pending_ojama_p2}",
        cx2, 120, TEXT_COLOR, 0.55)

    # 2P 盤面 + ネクスト (右側)
    rx = 1700
    put(panel, "2P field", rx, 56, TEXT_LABEL, 0.5)
    render_board_mini(panel, state.board_p2, x=rx, y=64, cell=12)
    put(panel, "next", rx + 80, 56, TEXT_LABEL, 0.45)
    render_pair(panel, state.next_p2 or (0, 0), x=rx + 80, y=64, cell=20)
    put(panel, "dnext", rx + 110, 56, TEXT_LABEL, 0.45)
    render_pair(panel, state.dnext_p2 or (0, 0), x=rx + 110, y=64, cell=20)

    # 状態フラグ
    put(panel, f"telop: {state.is_telop_visible}", cx, 145, TEXT_LABEL, 0.45)
    put(panel, f"locked: {state.is_match_end_locked}", cx2, 145, TEXT_LABEL, 0.45)

    # 16 指標 (中央〜右)
    render_indicators(panel, p1_set, p2_set, x=600, y=160)

    return panel


def overlay_field(frame, board, region):
    for vrow in range(VISIBLE_ROWS):
        for col in range(BOARD_COLS):
            row = vrow + HIDDEN_ROWS
            color = int(board.get(row, col))
            if color == COLOR_EMPTY:
                continue
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            bgr = COLOR_BGR_MAP.get(color, (60, 60, 60))
            sub = frame[y1:y2, x1:x2].astype(np.float32)
            tint = np.full_like(sub, bgr, dtype=np.float32)
            frame[y1:y2, x1:x2] = (sub * 0.6 + tint * 0.4).astype(np.uint8)
            label = COLOR_LABEL_MAP.get(color, "?")
            cv2.putText(
                frame, label, ((x1 + x2) // 2 - 6, (y1 + y2) // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--winner", default=None,
                        help="1P/2P (実勝者、ラベル表示用)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--detect-interval", type=float, default=0.5)
    parser.add_argument(
        "--model", default="models/win_predictor_v3.pt",
        help="存在しなければ v2 にフォールバック",
    )
    parser.add_argument(
        "--bg-fp-time", type=float, default=-1.0,
        help="試合開始秒。指定すると BG FP を取得 (背景打消し)",
    )
    parser.add_argument(
        "--cnn-model",
        default="models/cnn_phase_u_v7.pt",
        help="盤面認識用 CNN モデル",
    )
    parser.add_argument(
        "--use-calib", action="store_true",
        help="W11-C per-video 色キャリブレーション ON",
    )
    parser.add_argument(
        "--use-temporal", action="store_true",
        help="W11-D temporal voting (3 フレーム多数決) ON",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"video open failed: {args.video}")
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    pipeline = StatePipeline(
        cnn_model_path=args.cnn_model,
        use_per_video_calibrator=args.use_calib,
        use_temporal_voting=args.use_temporal,
    )
    if args.bg_fp_time >= 0:
        ok = pipeline.set_background_fingerprints_from_video(
            cap, args.bg_fp_time,
        )
        if ok:
            print(f"BG FP set at t={args.bg_fp_time}")
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    pipeline.reset(match_start_sec=args.start)

    # MLP モデル
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path("models/win_predictor_v2_mixed.pt")
        print(f"falling back to: {model_path}")
    model = WinPredictorMLP()
    model.load(model_path)

    # 16 指標 計算器
    ind = IndicatorCalculator()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path), fourcc, src_fps, (1920, 1080 + PANEL_H),
    )
    if not writer.isOpened():
        print("writer failed")
        return 1

    detect_interval_frames = max(1, int(src_fps * args.detect_interval))
    n_frames = int((args.end - args.start) * src_fps)
    state = None
    panel = np.full((PANEL_H, 1920, 3), PANEL_BG, dtype=np.uint8)
    p1_set = None
    p2_set = None
    prob_1p = 0.5
    print(f"render: {n_frames} frames, detect every {detect_interval_frames}")

    frame_idx = 0
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        t_sec = args.start + frame_idx / src_fps
        if frame_idx % detect_interval_frames == 0 or state is None:
            try:
                state = pipeline.extract(frame, t_sec=t_sec)
                # 16 指標
                p1_set = ind.compute_all(
                    state.board_p1,
                    next_pair=state.next_p1,
                    dnext_pair=state.dnext_p1,
                    incoming_ojama=state.pending_ojama_p1,
                    opponent_board=state.board_p2,
                )
                p2_set = ind.compute_all(
                    state.board_p2,
                    next_pair=state.next_p2,
                    dnext_pair=state.dnext_p2,
                    incoming_ojama=state.pending_ojama_p2,
                    opponent_board=state.board_p1,
                )
                features = encode_state(state)
                prob_1p = model.predict(features)
                panel = make_panel(
                    state, p1_set, p2_set, prob_1p, t_sec,
                    winner_actual=args.winner,
                )
            except Exception as e:
                print(f"  err t={t_sec}: {e}")

        # 上部 overlay
        if state is not None:
            overlay_field(frame, state.board_p1, DEFAULT_P1_REGION)
            overlay_field(frame, state.board_p2, DEFAULT_P2_REGION)
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
