"""W9-B: 1 試合のフレームから full-sheet review データを抽出。

弱点動画 (v07/v11/v13/v17/v18/v19) で full-sheet review シートを生成、
user による全セル目視ラベル付けを依頼するためのデータ。

各試合の中盤を中心に N フレームをサンプリング、各フレームから両側 12×6 = 72 セル
を patch として切り出し、CNN v7 (HybridClassifier) の自動ラベルとともに sheet と
labels.csv を出力。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_extract_full_sheet \
        --video data/frames/video_19.mp4 \
        --start 1500 --end 1580 \
        --bg-fp-time 1500 \
        --out-dir data/verify/phase_w_review/v19_m_full
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
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, ImageReader,
)


COLOR_LABEL: dict[int, str] = {
    COLOR_EMPTY: "EM", COLOR_RED: "RED", COLOR_BLUE: "BLUE",
    COLOR_GREEN: "GRN", COLOR_YELLOW: "YEL", COLOR_PURPLE: "PUR",
    COLOR_OJAMA: "OJM", COLOR_UNKNOWN: "??",
}
COLOR_BG: dict[int, tuple[int, int, int]] = {
    COLOR_EMPTY: (40, 40, 40),
    COLOR_RED: (40, 40, 200),
    COLOR_BLUE: (200, 80, 40),
    COLOR_GREEN: (40, 180, 40),
    COLOR_YELLOW: (40, 200, 220),
    COLOR_PURPLE: (180, 40, 180),
    COLOR_OJAMA: (170, 170, 170),
    COLOR_UNKNOWN: (80, 80, 120),
}

SAMPLE_PX = 110
SHEET_COLS = 20
SHEET_PAD = 10
LABEL_LINE1_HEIGHT = 22
LABEL_LINE2_HEIGHT = 50


def build_sheet(samples: list[dict]) -> np.ndarray:
    n = len(samples)
    cols = SHEET_COLS
    rows = (n + cols - 1) // cols
    cell_w = SAMPLE_PX + SHEET_PAD * 2
    cell_h = SAMPLE_PX + LABEL_LINE1_HEIGHT + LABEL_LINE2_HEIGHT + SHEET_PAD * 2
    sheet = np.full(
        (rows * cell_h, cols * cell_w, 3), 18, dtype=np.uint8,
    )
    for k, s in enumerate(samples):
        gr = k // cols
        gc = k % cols
        x0 = gc * cell_w + SHEET_PAD
        y0 = gr * cell_h + SHEET_PAD
        patch = cv2.resize(
            s["patch"], (SAMPLE_PX, SAMPLE_PX),
            interpolation=cv2.INTER_CUBIC,
        )
        sheet[y0:y0 + SAMPLE_PX, x0:x0 + SAMPLE_PX] = patch
        line1_y = y0 + SAMPLE_PX + LABEL_LINE1_HEIGHT - 4
        info1 = (
            f"#{s['id']} {s['side']} r{s['row']}c{s['col']} "
            f"t={s['time']:.1f}"
        )
        cv2.putText(
            sheet, info1, (x0 + 4, line1_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1, cv2.LINE_AA,
        )
        bg_y0 = y0 + SAMPLE_PX + LABEL_LINE1_HEIGHT
        bg_y1 = bg_y0 + LABEL_LINE2_HEIGHT
        bg_color = COLOR_BG.get(s["color"], (60, 60, 60))
        sheet[bg_y0:bg_y1, x0:x0 + SAMPLE_PX] = bg_color
        brightness = sum(bg_color) / 3
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        label = COLOR_LABEL.get(s["color"], "?")
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2,
        )
        tx = x0 + (SAMPLE_PX - tw) // 2
        ty = bg_y0 + (LABEL_LINE2_HEIGHT + th) // 2 - 4
        cv2.putText(
            sheet, label, (tx, ty),
            cv2.FONT_HERSHEY_DUPLEX, 1.2, text_color, 2, cv2.LINE_AA,
        )
    return sheet


def setup_reader(bg_fp_time: float, cap) -> ImageReader:
    """ImageReader を構築 (CNN v7 + BG FP)。"""
    from src.hybrid_classifier import HybridClassifier
    from src.patch_classifier import CnnPatchClassifier
    import torch
    cnn = CnnPatchClassifier()
    state = torch.load(
        "models/cnn_phase_u_v7.pt", map_location="cpu", weights_only=True,
    )
    cnn._model.load_state_dict(state)
    cnn._model.eval()
    classifier = HybridClassifier(cnn_classifier=cnn)
    reader = ImageReader(
        classifier=classifier,
        use_match_state=True,
        use_ui_mask=True,
        use_telop_mask=True,
    )
    if bg_fp_time >= 0:
        from src.background_fingerprint import capture_pair_robust
        bg_frames = []
        for offset in (-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5):
            t = max(0.0, bg_fp_time + offset)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fb = cap.read()
            if not ok or fb is None:
                continue
            if fb.shape[:2] != (1080, 1920):
                fb = cv2.resize(
                    fb, (1920, 1080), interpolation=cv2.INTER_AREA,
                )
            bg_frames.append(fb)
        if bg_frames:
            p1_t = (
                DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
                DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height,
            )
            p2_t = (
                DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
                DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height,
            )
            fp1, fp2 = capture_pair_robust(
                bg_frames, p1_t, p2_t,
            )
            reader.set_background_fingerprints(fp1, fp2)
            print(f"BG FP set at t={bg_fp_time}")
    return reader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--bg-fp-time", type=float, default=-1.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-frames", type=int, default=2,
                        help="サンプリングフレーム数 (各 60-180s)")
    parser.add_argument("--max-cells", type=int, default=300,
                        help="出力する最大セル数")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"video open failed: {args.video}")
        return 1

    reader = setup_reader(args.bg_fp_time, cap)

    # サンプリング時刻を均等分割
    duration = args.end - args.start
    if args.n_frames < 1:
        return 1
    times = [
        args.start + duration * (i + 0.5) / args.n_frames
        for i in range(args.n_frames)
    ]

    all_samples: list[dict] = []
    sample_id = 0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        b1, b2 = reader.read_both_boards(frame)
        for side, region, board in (
            ("1P", DEFAULT_P1_REGION, b1),
            ("2P", DEFAULT_P2_REGION, b2),
        ):
            for vrow in range(12):  # visible rows 0-11
                for col in range(BOARD_COLS):
                    row = vrow + HIDDEN_ROWS
                    color = int(board.get(row, col))
                    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(y1 + 1, min(y2, h))
                    patch = frame[y1:y2, x1:x2].copy()
                    if patch.size == 0:
                        continue
                    sample_id += 1
                    all_samples.append({
                        "id": sample_id, "time": t, "side": side,
                        "row": vrow, "col": col, "color": color,
                        "patch": patch,
                    })
    cap.release()

    if len(all_samples) > args.max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_samples), args.max_cells, replace=False)
        all_samples = [all_samples[i] for i in sorted(idx.tolist())]
        # ID を再付与
        for k, s in enumerate(all_samples):
            s["id"] = k + 1

    print(f"total cells: {len(all_samples)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet = build_sheet(all_samples)
    sheet_path = out_dir / "sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"saved: {to_windows_path(sheet_path)}")

    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "time", "side", "row", "col", "recognized", "your_answer",
        ])
        for s in all_samples:
            w.writerow([
                s["id"], f"{s['time']:.1f}", s["side"], s["row"], s["col"],
                COLOR_LABEL.get(s["color"], "?"), "",
            ])
    print(f"saved: {to_windows_path(csv_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
