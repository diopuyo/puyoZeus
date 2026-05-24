"""W8: CNN 誤認の最も確実なケース = 物理矛盾 (4+ 同色連結) のセルとその周辺を抽出。

CNN v7 は誤認しても自信満々 (high confidence) で uncertain 抽出に引っかからない。
代わりに「4+ 同色連結が残っている = 物理的にあり得ない」状態のセルとその周辺
8 セルを目視レビュー対象として抽出する。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_extract_violations \
        data/frames/video_05.mp4 --start 315 --end 385 \
        --out-dir data/verify/phase_w_review/violations/v05
"""
from __future__ import annotations

import argparse
import csv
import os
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
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    Board,
)
from src.chain import ChainSimulator, MIN_ERASE_COUNT
from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    ImageReader,
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

SAMPLE_PX = 130
SHEET_COLS = 20
SHEET_PAD = 12
LABEL_LINE1_HEIGHT = 22
LABEL_LINE2_HEIGHT = 56


def detect_violations(
    board: Board, simulator: ChainSimulator,
) -> list[tuple[int, int, int]]:
    """4+ 同色連結に含まれる全セルと、その隣接セル (合わせて目視対象) を返す。

    Returns:
        [(row, col, color), ...] - 重複なし
    """
    out: dict[tuple[int, int], int] = {}
    groups = simulator.find_groups(board)
    for g in groups:
        if g.color == COLOR_OJAMA or g.color == COLOR_EMPTY:
            continue
        if g.size < MIN_ERASE_COUNT:
            continue
        for r, c in g.cells:
            out[(r, c)] = g.color
            # 隣接セルも疑い対象 (4+ 連結を構成しているセルが誤認の可能性)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (HIDDEN_ROWS <= nr < BOARD_ROWS
                        and 0 <= nc < BOARD_COLS):
                    nc_color = int(board.get(nr, nc))
                    if (nr, nc) not in out:
                        out[(nr, nc)] = nc_color
    return [(r, c, color) for (r, c), color in out.items()]


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
            s["patch"], (SAMPLE_PX, SAMPLE_PX), interpolation=cv2.INTER_CUBIC,
        )
        sheet[y0:y0 + SAMPLE_PX, x0:x0 + SAMPLE_PX] = patch
        line1_y = y0 + SAMPLE_PX + LABEL_LINE1_HEIGHT - 4
        info1 = (
            f"#{s['id']} {s['side']} r{s['row']}c{s['col']} "
            f"t={s['time']:.1f}"
        )
        cv2.putText(
            sheet, info1, (x0 + 4, line1_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA,
        )
        bg_y0 = y0 + SAMPLE_PX + LABEL_LINE1_HEIGHT
        bg_y1 = bg_y0 + LABEL_LINE2_HEIGHT
        bg_color = COLOR_BG.get(s["color"], (60, 60, 60))
        sheet[bg_y0:bg_y1, x0:x0 + SAMPLE_PX] = bg_color
        brightness = sum(bg_color) / 3
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        label = COLOR_LABEL.get(s["color"], "?")
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
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--bg-fp-time", type=float, default=-1.0,
        help="試合開始秒。指定すると BG FP を設定 (背景打消し)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"video open failed: {args.input_video}")
        return 1

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
    )
    # BG FP 設定 (背景打消し)
    if args.bg_fp_time >= 0:
        from src.background_fingerprint import capture_pair_robust
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
            reader.set_background_fingerprints(fp1, fp2)
            print(f"BG FP set from {len(bg_frames)} frames at t={args.bg_fp_time}")
    simulator = ChainSimulator()

    samples: list[dict] = []
    sample_id = 1
    seen: set[tuple[float, str, int, int]] = set()

    t = args.start
    while t <= args.end and len(samples) < args.max_samples:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok or fr is None:
            t += args.interval
            continue
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
        b1, b2 = reader.read_both_boards(fr)
        for side, board, region in [
            ("1P", b1, DEFAULT_P1_REGION),
            ("2P", b2, DEFAULT_P2_REGION),
        ]:
            violations = detect_violations(board, simulator)
            for row, col, color in violations:
                key = (round(t, 1), side, row, col)
                if key in seen:
                    continue
                seen.add(key)
                vrow = row - HIDDEN_ROWS
                if vrow < 0:
                    continue
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                h, w = fr.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))
                patch = fr[y1:y2, x1:x2].copy()
                if patch.size == 0:
                    continue
                samples.append({
                    "id": sample_id, "time": t, "side": side,
                    "row": vrow, "col": col,
                    "color": color, "patch": patch,
                })
                sample_id += 1
                if len(samples) >= args.max_samples:
                    break
            if len(samples) >= args.max_samples:
                break
        t += args.interval
    cap.release()

    if not samples:
        print("no violations found")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = build_sheet(samples)
    sheet_path = out_dir / "sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "time", "side", "row", "col", "recognized", "your_answer",
        ])
        for s in samples:
            label = COLOR_LABEL.get(s["color"], "?")
            w.writerow([
                s["id"], f"{s['time']:.1f}", s["side"],
                s["row"], s["col"], label, label,
            ])
    print(f"sheet: {to_windows_path(sheet_path)} ({len(samples)} cells)")
    print(f"csv  : {to_windows_path(csv_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
