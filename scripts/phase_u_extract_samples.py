"""Phase U: ラベル付け補助シート生成 (認識色を大文字表示)。

Phase U の素朴な ImageReader (HSV ベース + 物理ルール) で動画フレームを認識し、
誤認識候補セルを 1 枚の grid 画像と CSV に出力する。

旧 extract_review_samples.py に対する改善:
    - 認識色 (EM/RD/BL/...) を **大きい文字 + 色付き背景** で目立たせる
    - id, 座標, presence は小さい字で副情報として表示
    - 全セル抽出 / 不確かセルのみ抽出 を選択可能

利用例:
    python scripts/phase_u_extract_samples.py \\
        data/verify/review_videos/clip_v02_m1.mp4 \\
        --times 5,15,30,60 \\
        --out-dir data/verify/phase_u_samples_v02 \\
        --max-samples 60 \\
        --side both
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

from src.background_fingerprint import (
    BackgroundFingerprint,
    capture_pair_robust,
)
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
from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    ImageReader,
)

# === レイアウト定数 ===
SAMPLE_PX = 130              # 拡大セル画像の 1 辺
SHEET_COLS = 20               # 1 行あたりの列数
SHEET_PAD = 12               # 余白
LABEL_LINE1_HEIGHT = 22      # 1 行目 (id, 座標) の高さ
LABEL_LINE2_HEIGHT = 56      # 2 行目 (認識色 大文字) の高さ
LABEL_HEIGHT = LABEL_LINE1_HEIGHT + LABEL_LINE2_HEIGHT

COLOR_LABEL: dict[int, str] = {
    COLOR_EMPTY: "EM", COLOR_RED: "RED", COLOR_BLUE: "BLUE",
    COLOR_GREEN: "GRN", COLOR_YELLOW: "YEL", COLOR_PURPLE: "PUR",
    COLOR_OJAMA: "OJM", COLOR_UNKNOWN: "??",
}

# ラベル背景色 (BGR)
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
    patch: np.ndarray


def _classify_frame(
    frame: np.ndarray, reader: ImageReader, side_filter: str,
) -> list[Cell]:
    """1 フレームから全セル or 1 サイドのみを取得。"""
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    board1, board2 = reader.read_both_boards(frame)
    cells: list[Cell] = []
    sides = []
    if side_filter in ("1P", "both"):
        sides.append(("1P", board1, DEFAULT_P1_REGION))
    if side_filter in ("2P", "both"):
        sides.append(("2P", board2, DEFAULT_P2_REGION))
    for side, board, region in sides:
        for vrow in range(VISIBLE_ROWS):
            for col in range(BOARD_COLS):
                row = vrow + HIDDEN_ROWS
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                cells.append(Cell(
                    sample_id=0, time=0.0, side=side,
                    row=vrow, col=col, color=int(board.get(row, col)),
                    patch=patch.copy(),
                ))
    return cells


def _build_sheet(cands: list[Cell]) -> np.ndarray:
    """大文字認識色 + 色付き背景で見やすいシートを生成。"""
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
        # パッチ拡大
        patch = cv2.resize(
            c.patch, (SAMPLE_PX, SAMPLE_PX), interpolation=cv2.INTER_CUBIC,
        )
        sheet[y0:y0 + SAMPLE_PX, x0:x0 + SAMPLE_PX] = patch
        # 1 行目: id + side + 座標 (小さい字)
        line1_y = y0 + SAMPLE_PX + LABEL_LINE1_HEIGHT - 4
        info1 = f"#{c.sample_id} {c.side} r{c.row}c{c.col} t={c.time:.1f}"
        cv2.putText(
            sheet, info1, (x0 + 4, line1_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210),
            1, cv2.LINE_AA,
        )
        # 2 行目: 認識色を大文字 + 色付き背景
        bg_y0 = y0 + SAMPLE_PX + LABEL_LINE1_HEIGHT
        bg_y1 = bg_y0 + LABEL_LINE2_HEIGHT
        bg_color = COLOR_BG.get(c.color, (60, 60, 60))
        sheet[bg_y0:bg_y1, x0:x0 + SAMPLE_PX] = bg_color
        # コントラスト計算: 背景輝度から白/黒を選ぶ
        brightness = sum(bg_color) / 3
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        label = COLOR_LABEL.get(c.color, "?")
        # text を中央配置
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
    parser.add_argument("--times", required=True,
                        help="comma-separated seconds, e.g. '5,15,30,60'")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bg-fp-time", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--side", choices=["1P", "2P", "both"], default="both")
    parser.add_argument("--all-cells", action="store_true",
                        help="全セル含める (False: 認識色 EMPTY 以外のみ)")
    parser.add_argument("--skip-anim", action="store_true",
                        help="連鎖中フレームを自動回避 (前 0.2s と差分が大きい "
                             "時刻はスキップして次秒シフト)")
    parser.add_argument("--anim-diff-threshold", type=float, default=18.0,
                        help="連鎖中判定の盤面差分しきい値")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = [float(t) for t in args.times.split(",") if t.strip()]

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"video open failed: {args.input_video}")
        return 1

    # bg fp 取得 (試合開始時の盤面)
    bg_frames = []
    for i in range(8):
        t = max(0.0, args.bg_fp_time + (i - 4) * 0.1)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fb = cap.read()
        if not ok or fb is None:
            continue
        if fb.shape[:2] != (1080, 1920):
            fb = cv2.resize(fb, (1920, 1080), interpolation=cv2.INTER_AREA)
        bg_frames.append(fb)
    p1_t = (
        DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
        DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height,
    )
    p2_t = (
        DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
        DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height,
    )
    if bg_frames:
        fp1, fp2 = capture_pair_robust(bg_frames, p1_t, p2_t)
    else:
        fp1 = fp2 = None

    reader = ImageReader(
        bg_fingerprint_p1=fp1, bg_fingerprint_p2=fp2,
        use_ui_mask=False,  # ID シフト回避のためレビュー時は UI Mask off
    )

    def _is_animation_frame(t: float) -> bool:
        """前 0.2s フレームと盤面領域の差分が大 → 連鎖中と判定。"""
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - 0.2) * 1000.0)
        ok_p, fp = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok_c, fc = cap.read()
        if not (ok_p and ok_c) or fp is None or fc is None:
            return False
        if fp.shape[:2] != (1080, 1920):
            fp = cv2.resize(fp, (1920, 1080), interpolation=cv2.INTER_AREA)
        if fc.shape[:2] != (1080, 1920):
            fc = cv2.resize(fc, (1920, 1080), interpolation=cv2.INTER_AREA)
        # 1P/2P 盤面領域のみで差分計算
        for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
            x, y, w, h = region.x, region.y, region.width, region.height
            a = fp[y:y + h, x:x + w].astype(np.int16)
            b = fc[y:y + h, x:x + w].astype(np.int16)
            diff = float(np.mean(np.abs(a - b)))
            if diff >= args.anim_diff_threshold:
                return True
        return False

    all_cells: list[Cell] = []
    for t in times:
        if args.skip_anim:
            # 最大 5 秒分シフトして安定フレームを探す
            shifted_t = t
            for _ in range(5):
                if not _is_animation_frame(shifted_t):
                    break
                shifted_t += 1.0
            t = shifted_t
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        if not ok or fr is None:
            print(f"frame fetch failed: t={t}")
            continue
        cells = _classify_frame(fr, reader, args.side)
        for c in cells:
            c.time = t
        all_cells.extend(cells)

    # フィルタ: --all-cells で全部、それ以外は認識色 EMPTY 以外のみ
    if not args.all_cells:
        all_cells = [c for c in all_cells if c.color != COLOR_EMPTY]

    # max-samples で打ち切り (時刻順 → 位置順)
    all_cells = all_cells[: args.max_samples]
    for i, c in enumerate(all_cells, start=1):
        c.sample_id = i

    if not all_cells:
        print("no cells matched")
        return 1

    sheet = _build_sheet(all_cells)
    sheet_path = (out_dir / "sheet.png").resolve()
    cv2.imwrite(str(sheet_path), sheet)
    # Windows フルパス (Ctrl+click 用)
    print(f"sheet: {to_windows_path(sheet_path)} ({len(all_cells)} cells)")

    csv_path = (out_dir / "labels.csv").resolve()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "time", "side", "row", "col", "recognized", "your_answer",
        ])
        for c in all_cells:
            label = COLOR_LABEL.get(c.color, "?")
            # your_answer を recognized でプリセット (差分のみ修正)
            w.writerow([
                c.sample_id, f"{c.time:.1f}", c.side, c.row, c.col,
                label, label,
            ])
    print(f"csv  : {to_windows_path(csv_path)}")
    print(
        "labels: EM=empty RED=red BLUE=blue GRN=green "
        "YEL=yellow PUR=purple OJM=ojama"
    )
    print(
        "your_answer is preset to recognized -> "
        "edit only rows that differ from the actual color"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
