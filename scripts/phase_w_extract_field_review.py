"""W14-E: Field 全体レビューツール。

cell パッチではなく**盤面全体 (フィールド) を grid+ラベルoverlay で表示**して
ユーザーがコンテキストを見ながら誤検出を発見できるレビュー画像を生成。

各フレームに 1P/2P の board crop + grid + CNN label を重ね合わせ、
複数フレームを 1 枚の sheet に並べる。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_extract_field_review \
        --video data/frames/video_18.mp4 \
        --start 251 --end 320 \
        --bg-fp-time 251 \
        --n-frames 12 \
        --out-dir data/verify/phase_w_review/v18_m03_field
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

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN,
    COLOR_OJAMA, COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN,
    COLOR_YELLOW, HIDDEN_ROWS,
)
from src.image_reader import (
    DEFAULT_P1_REGION, DEFAULT_P2_REGION,
)
from src.state_pipeline import StatePipeline


COLOR_LABEL: dict[int, str] = {
    COLOR_EMPTY: "", COLOR_RED: "R", COLOR_BLUE: "B",
    COLOR_GREEN: "G", COLOR_YELLOW: "Y", COLOR_PURPLE: "P",
    COLOR_OJAMA: "O", COLOR_UNKNOWN: "?",
}
COLOR_BORDER: dict[int, tuple[int, int, int]] = {
    COLOR_EMPTY: (60, 60, 60),
    COLOR_RED: (60, 60, 240),
    COLOR_BLUE: (240, 100, 60),
    COLOR_GREEN: (60, 220, 60),
    COLOR_YELLOW: (60, 240, 240),
    COLOR_PURPLE: (220, 60, 220),
    COLOR_OJAMA: (190, 190, 190),
    COLOR_UNKNOWN: (120, 120, 200),
}


def crop_field(
    frame: np.ndarray, region, pad: int = 4,
) -> tuple[np.ndarray, tuple[int, int]]:
    """region で示される盤面を crop。返り値: (cropped, (x_off, y_off))。"""
    h, w = frame.shape[:2]
    # visible area = rows HIDDEN_ROWS .. HIDDEN_ROWS+11
    x1, y1, _, _ = region.cell_sample_rect(HIDDEN_ROWS, 0)
    _, _, x2, y2 = region.cell_sample_rect(HIDDEN_ROWS + 11, BOARD_COLS - 1)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame[y1:y2, x1:x2].copy(), (x1, y1)


def annotate_field(
    field: np.ndarray, region, board_pred,
    side: str, frame_xy: tuple[int, int],
) -> np.ndarray:
    """field 画像に grid + CNN label overlay。"""
    out = field.copy()
    x_off, y_off = frame_xy
    for vrow in range(12):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            color = int(board_pred.get(row, col))
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1, x2 = x1 - x_off, x2 - x_off
            y1, y2 = y1 - y_off, y2 - y_off
            border = COLOR_BORDER.get(color, (60, 60, 60))
            label = COLOR_LABEL.get(color, "?")
            # 半透明の枠 (色付きラベルセルだけ目立たせる)
            if color != COLOR_EMPTY:
                cv2.rectangle(
                    out, (x1, y1), (x2, y2), border, 2,
                )
                if label:
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1,
                    )
                    tx = x1 + 2
                    ty = y1 + th + 2
                    cv2.rectangle(
                        out, (tx - 1, ty - th - 1),
                        (tx + tw + 1, ty + 2),
                        (0, 0, 0), -1,
                    )
                    cv2.putText(
                        out, label, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6,
                        border, 1, cv2.LINE_AA,
                    )
            else:
                # EM は薄いグレー枠だけ
                cv2.rectangle(
                    out, (x1, y1), (x2, y2),
                    (40, 40, 40), 1,
                )
    # side ラベル
    cv2.putText(
        out, side, (4, 18),
        cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA,
    )
    return out


def build_sheet(
    rows: list[list[np.ndarray]], pad: int = 8,
) -> np.ndarray:
    """rows[i] = [(P1_field, P2_field, info), ...] を grid 配置。"""
    n_rows = len(rows)
    if n_rows == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    n_cols = len(rows[0])
    # 各 cell の最大 size
    cell_w = 0
    cell_h = 0
    for row_imgs in rows:
        for img in row_imgs:
            cell_w = max(cell_w, img.shape[1])
            cell_h = max(cell_h, img.shape[0])
    cell_w_pad = cell_w + pad * 2
    cell_h_pad = cell_h + pad * 2 + 20  # +info bar
    sheet_w = cell_w_pad * n_cols
    sheet_h = cell_h_pad * n_rows
    sheet = np.full(
        (sheet_h, sheet_w, 3), 24, dtype=np.uint8,
    )
    for r, row_imgs in enumerate(rows):
        for c, img in enumerate(row_imgs):
            y0 = r * cell_h_pad + pad
            x0 = c * cell_w_pad + pad
            ih, iw = img.shape[:2]
            sheet[y0:y0 + ih, x0:x0 + iw] = img
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--bg-fp-time", type=float, default=-1.0)
    parser.add_argument("--n-frames", type=int, default=2)
    parser.add_argument(
        "--cnn-model", default="models/cnn_phase_u_v15.pt",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--all-refiners-on", action="store_true",
        help="W11/W12/W13 全 refiner ON で出力",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"video open failed: {args.video}")
        return 1

    pipeline = StatePipeline(
        cnn_model_path=args.cnn_model,
        use_per_video_calibrator=args.all_refiners_on,
        use_temporal_voting=args.all_refiners_on,
        use_score_eraser=args.all_refiners_on,
        use_pair_landing_check=args.all_refiners_on,
    )
    if args.bg_fp_time >= 0:
        pipeline.set_background_fingerprints_from_video(
            cap, args.bg_fp_time,
        )
    pipeline.reset(match_start_sec=args.start)

    duration = args.end - args.start
    times = [
        args.start + duration * (i + 0.5) / args.n_frames
        for i in range(args.n_frames)
    ]

    # warmup は temporal voting の history を EM で埋めて frame の色を潰すバグの
    # 原因となるので削除。スパース時間サンプリングでは個別フレーム評価が正しい。

    rows_imgs: list[list[np.ndarray]] = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "labels.csv"
    csv_rows: list[list[str]] = [
        ["id", "time", "side", "row", "col", "recognized", "your_answer"],
    ]
    sample_id = 0

    for t in times:
        # スパース時間: temporal state を carry over しない (frame ごと独立評価)
        pipeline.reset(match_start_sec=t)
        if args.bg_fp_time >= 0:
            pipeline.set_background_fingerprints_from_video(
                cap, args.bg_fp_time,
            )
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        state = pipeline.extract(frame, t)
        # crop and annotate
        p1_field, p1_off = crop_field(frame, DEFAULT_P1_REGION)
        p2_field, p2_off = crop_field(frame, DEFAULT_P2_REGION)
        p1_ann = annotate_field(
            p1_field, DEFAULT_P1_REGION, state.board_p1, "1P", p1_off,
        )
        p2_ann = annotate_field(
            p2_field, DEFAULT_P2_REGION, state.board_p2, "2P", p2_off,
        )
        # info banner
        info = np.full((26, p1_ann.shape[1] + p2_ann.shape[1] + 12, 3),
                       40, dtype=np.uint8)
        text = (
            f"t={t:.1f}s  next1P={state.next_p1}  next2P={state.next_p2}  "
            f"score={state.score_p1}/{state.score_p2}"
        )
        cv2.putText(
            info, text, (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA,
        )
        rows_imgs.append([p1_ann, p2_ann])

        # CSV (cells = 12 visible × 6 col × 2 sides = 144)
        for side, board in (("1P", state.board_p1), ("2P", state.board_p2)):
            for vrow in range(12):
                row = vrow + HIDDEN_ROWS
                for col in range(BOARD_COLS):
                    color = int(board.get(row, col))
                    label = COLOR_LABEL.get(color, "?") or "EM"
                    if label == "":
                        label = "EM"
                    sample_id += 1
                    csv_rows.append([
                        str(sample_id), f"{t:.1f}", side,
                        str(vrow), str(col),
                        label_full(color),
                        "",
                    ])
    cap.release()

    # 4 列 × N 行 にする (1 行に P1+P2 1 timestep)
    sheet = build_sheet(rows_imgs)
    sheet_path = out_dir / "field_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"saved: {to_windows_path(sheet_path)}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"saved: {to_windows_path(csv_path)} ({sample_id} cells)")
    return 0


def label_full(code: int) -> str:
    return {
        0: "EM", 1: "RED", 2: "BLUE", 3: "GRN",
        4: "YEL", 5: "PUR", 9: "OJM", 10: "UN",
    }.get(code, "?")


if __name__ == "__main__":
    sys.exit(main())
