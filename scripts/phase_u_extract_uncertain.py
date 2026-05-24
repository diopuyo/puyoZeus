"""怪しいセル (CNN 確信度低) のみ抽出する重点レビュー用シート生成。

phase_u_extract_samples.py の派生。CNN v5 の predict_proba で
max prob < threshold のセルだけシート化、ユーザのレビュー量を最小化。

利用例:
    python scripts/phase_u_extract_uncertain.py \\
        data/frames/video_01.mp4 \\
        --times 200,220,300,400,500 \\
        --out-dir data/verify/phase_u_uncertain_v01 \\
        --cnn-model models/cnn_phase_u_v5.pt \\
        --threshold 0.80 --max-samples 30
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
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
from src.background_fingerprint import capture_pair_robust
from src.hybrid_classifier import HybridClassifier
from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    ImageReader,
)
from src.patch_classifier import CLASS_INDEX_TO_COLOR, CnnPatchClassifier

SAMPLE_PX = 130
SHEET_COLS = 20
SHEET_PAD = 12
LABEL_LINE1_HEIGHT = 22
LABEL_LINE2_HEIGHT = 56
LABEL_HEIGHT = LABEL_LINE1_HEIGHT + LABEL_LINE2_HEIGHT

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


@dataclass
class Cell:
    sample_id: int
    time: float
    side: str
    row: int
    col: int
    color: int
    confidence: float
    patch: np.ndarray


def _build_sheet(cands: list[Cell]) -> np.ndarray:
    n = len(cands)
    cols = SHEET_COLS
    rows = (n + cols - 1) // cols
    cell_w = SAMPLE_PX + SHEET_PAD * 2
    cell_h = SAMPLE_PX + LABEL_HEIGHT + SHEET_PAD * 2
    sheet = np.full(
        (rows * cell_h, cols * cell_w, 3), 18, dtype=np.uint8,
    )
    for idx, c in enumerate(cands):
        gr = idx // cols
        gc = idx % cols
        x0 = gc * cell_w + SHEET_PAD
        y0 = gr * cell_h + SHEET_PAD
        patch = cv2.resize(
            c.patch, (SAMPLE_PX, SAMPLE_PX), interpolation=cv2.INTER_CUBIC,
        )
        sheet[y0:y0 + SAMPLE_PX, x0:x0 + SAMPLE_PX] = patch
        line1_y = y0 + SAMPLE_PX + LABEL_LINE1_HEIGHT - 4
        info = (
            f"#{c.sample_id} {c.side} r{c.row}c{c.col} "
            f"t={c.time:.1f} p={c.confidence:.2f}"
        )
        cv2.putText(
            sheet, info, (x0 + 4, line1_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210),
            1, cv2.LINE_AA,
        )
        bg_y0 = y0 + SAMPLE_PX + LABEL_LINE1_HEIGHT
        bg_y1 = bg_y0 + LABEL_LINE2_HEIGHT
        bg_color = COLOR_BG.get(c.color, (60, 60, 60))
        sheet[bg_y0:bg_y1, x0:x0 + SAMPLE_PX] = bg_color
        brightness = sum(bg_color) / 3
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        label = COLOR_LABEL.get(c.color, "?")
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3,
        )
        tx = x0 + (SAMPLE_PX - tw) // 2
        ty = bg_y0 + (LABEL_LINE2_HEIGHT + th) // 2 - 4
        cv2.putText(
            sheet, label, (tx, ty),
            cv2.FONT_HERSHEY_DUPLEX, 1.4, text_color, 3, cv2.LINE_AA,
        )
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("--times", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cnn-model", default="models/cnn_phase_u_v5.pt")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="CNN max prob がこの値未満なら不確か")
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--side", choices=["1P", "2P", "both"], default="both")
    parser.add_argument("--bg-fp-time", type=float, default=-1.0,
                        help="この時刻 ±0.5s から BG FP 取得 (試合開始時刻、"
                             "BG 距離小セルを uncertain から除外)")
    parser.add_argument("--bg-empty-threshold", type=float, default=28.0,
                        help="BG 距離がこの未満なら EMPTY 確定で除外")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    times = [float(t) for t in args.times.split(",") if t.strip()]

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"video open failed: {args.input_video}")
        return 1

    # CNN ロード
    import torch
    cnn = CnnPatchClassifier()
    state = torch.load(
        args.cnn_model, map_location="cpu", weights_only=True,
    )
    cnn._model.load_state_dict(state)
    cnn._model.eval()
    print(f"loaded CNN: {args.cnn_model}")

    # 試合開始時刻指定 → 周辺 8 フレームから robust BG FP 取得
    fp1 = fp2 = None
    if args.bg_fp_time >= 0:
        bg_frames = []
        for offset in (-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5):
            t = max(0.0, args.bg_fp_time + offset)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fb = cap.read()
            if not ok or fb is None:
                continue
            if fb.shape[:2] != (1080, 1920):
                fb = cv2.resize(fb, (1920, 1080), interpolation=cv2.INTER_AREA)
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
            fp1, fp2 = capture_pair_robust(bg_frames, p1_t, p2_t)
            print(f"BG FP captured from {len(bg_frames)} frames at t={args.bg_fp_time}")

    all_cells: list[Cell] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok or fr is None:
            print(f"frame fetch failed: t={t}")
            continue
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
        hsv_full = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV) if (fp1 or fp2) else None
        sides = []
        if args.side in ("1P", "both"):
            sides.append(("1P", DEFAULT_P1_REGION, fp1))
        if args.side in ("2P", "both"):
            sides.append(("2P", DEFAULT_P2_REGION, fp2))
        for side, region, bg_fp in sides:
            for vrow in range(VISIBLE_ROWS):
                for col in range(BOARD_COLS):
                    row = vrow + HIDDEN_ROWS
                    h, w = fr.shape[:2]
                    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(y1 + 1, min(y2, h))
                    patch = fr[y1:y2, x1:x2]
                    if patch.size == 0:
                        continue
                    # BG FP 距離チェック (近ければ EMPTY 確定 → uncertain 除外)
                    if bg_fp is not None and hsv_full is not None:
                        from src.background_fingerprint import CellFingerprint
                        hp = hsv_full[y1:y2, x1:x2]
                        cur_fp = CellFingerprint(
                            int(np.median(hp[:, :, 0])),
                            int(np.median(hp[:, :, 1])),
                            int(np.median(hp[:, :, 2])),
                        )
                        bg_cell = bg_fp.cell_at(vrow, col)
                        if cur_fp.distance_to(bg_cell) < args.bg_empty_threshold:
                            continue
                    probs = cnn.predict_proba(patch)
                    best_idx = int(np.argmax(probs))
                    color = CLASS_INDEX_TO_COLOR[best_idx]
                    conf = float(probs[best_idx])
                    if conf >= args.threshold:
                        continue
                    all_cells.append(Cell(
                        sample_id=0, time=t, side=side,
                        row=vrow, col=col, color=color,
                        confidence=conf, patch=patch.copy(),
                    ))

    # 確信度の低い順で max_samples
    all_cells.sort(key=lambda c: c.confidence)
    all_cells = all_cells[: args.max_samples]
    for i, c in enumerate(all_cells, start=1):
        c.sample_id = i

    if not all_cells:
        print("no uncertain cells found (all confident)")
        return 0

    sheet = _build_sheet(all_cells)
    sheet_path = out_dir / "sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"sheet: {to_windows_path(sheet_path)} ({len(all_cells)} uncertain cells)")

    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "time", "side", "row", "col",
            "recognized", "confidence", "your_answer",
        ])
        for c in all_cells:
            label = COLOR_LABEL.get(c.color, "?")
            w.writerow([
                c.sample_id, f"{c.time:.1f}", c.side, c.row, c.col,
                label, f"{c.confidence:.3f}", label,
            ])
    print(f"csv  : {to_windows_path(csv_path)}")
    print("(your_answer は recognized でプリセット、違う行のみ書き換え)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
