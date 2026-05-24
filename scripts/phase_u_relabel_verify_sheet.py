"""Phase U V1.3' 検証: 再ラベル結果から各色 N 件をサンプルしてシート画像化。

`data/training_phase_u/parallel_relabeled/*.npz` から各色 (RED/BLUE/GRN/
YEL/PUR/OJM/EM) を均等にランダム抽出し、認識色を大文字 + 色付き背景で
表示するレビューシート画像 + CSV を生成する。

ユーザーは色が明らかにズレている件があれば閾値再調整を判断できる。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_relabel_verify_sheet \
        --per-color 10
"""
from __future__ import annotations

import argparse
import csv
import os
import random
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
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
)

SAMPLE_PX = 130
SHEET_COLS = 5
SHEET_PAD = 12
LABEL_LINE1_HEIGHT = 22
LABEL_LINE2_HEIGHT = 56
LABEL_HEIGHT = LABEL_LINE1_HEIGHT + LABEL_LINE2_HEIGHT

COLOR_LABEL: dict[int, str] = {
    COLOR_EMPTY: "EM", COLOR_RED: "RED", COLOR_BLUE: "BLUE",
    COLOR_GREEN: "GRN", COLOR_YELLOW: "YEL", COLOR_PURPLE: "PUR",
    COLOR_OJAMA: "OJM",
}
COLOR_BG: dict[int, tuple[int, int, int]] = {
    COLOR_EMPTY: (40, 40, 40),
    COLOR_RED: (40, 40, 200),
    COLOR_BLUE: (200, 80, 40),
    COLOR_GREEN: (40, 180, 40),
    COLOR_YELLOW: (40, 200, 220),
    COLOR_PURPLE: (180, 40, 180),
    COLOR_OJAMA: (170, 170, 170),
}
COLOR_ORDER = [
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
]


def collect_samples(
    in_dir: Path, per_color: int, seed: int,
) -> dict[int, list[tuple[np.ndarray, str, int]]]:
    """各色を均等に per_color 件サンプル。(patch, source_npz, idx) のリスト。"""
    rng = random.Random(seed)
    npz_files = sorted(in_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"no npz files in {in_dir}")

    # 全 npz を線形スキャンしながら reservoir sampling
    reservoirs: dict[int, list[tuple[np.ndarray, str, int]]] = {
        c: [] for c in COLOR_ORDER
    }
    counts: dict[int, int] = {c: 0 for c in COLOR_ORDER}

    for npz_path in npz_files:
        d = np.load(npz_path)
        patches = d["patches"]
        labels = d["labels"]
        for i in range(patches.shape[0]):
            color = int(labels[i])
            if color not in reservoirs:
                continue
            counts[color] += 1
            res = reservoirs[color]
            if len(res) < per_color:
                res.append((patches[i].copy(), npz_path.name, i))
            else:
                # reservoir sampling
                j = rng.randint(0, counts[color] - 1)
                if j < per_color:
                    res[j] = (patches[i].copy(), npz_path.name, i)
    return reservoirs


def build_sheet(
    reservoirs: dict[int, list[tuple[np.ndarray, str, int]]],
) -> tuple[np.ndarray, list[dict]]:
    """grid シートを作成して csv 行も返す。"""
    flat: list[tuple[int, np.ndarray, str, int]] = []
    for color in COLOR_ORDER:
        for patch, src, idx in reservoirs[color]:
            flat.append((color, patch, src, idx))

    n = len(flat)
    cols = SHEET_COLS
    rows = (n + cols - 1) // cols
    cell_w = SAMPLE_PX + SHEET_PAD * 2
    cell_h = SAMPLE_PX + LABEL_HEIGHT + SHEET_PAD * 2
    sheet = np.full(
        (rows * cell_h, cols * cell_w, 3), 18, dtype=np.uint8,
    )

    csv_rows: list[dict] = []

    for k, (color, patch, src, idx) in enumerate(flat):
        gr = k // cols
        gc = k % cols
        x0 = gc * cell_w + SHEET_PAD
        y0 = gr * cell_h + SHEET_PAD

        resized = cv2.resize(
            patch, (SAMPLE_PX, SAMPLE_PX), interpolation=cv2.INTER_CUBIC,
        )
        sheet[y0:y0 + SAMPLE_PX, x0:x0 + SAMPLE_PX] = resized

        line1_y = y0 + SAMPLE_PX + LABEL_LINE1_HEIGHT - 4
        info1 = f"#{k + 1} {src} idx={idx}"
        cv2.putText(
            sheet, info1, (x0 + 4, line1_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210),
            1, cv2.LINE_AA,
        )

        bg_y0 = y0 + SAMPLE_PX + LABEL_LINE1_HEIGHT
        bg_y1 = bg_y0 + LABEL_LINE2_HEIGHT
        bg_color = COLOR_BG.get(color, (60, 60, 60))
        sheet[bg_y0:bg_y1, x0:x0 + SAMPLE_PX] = bg_color
        brightness = sum(bg_color) / 3
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        label = COLOR_LABEL.get(color, "?")
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3,
        )
        tx = x0 + (SAMPLE_PX - tw) // 2
        ty = bg_y0 + (LABEL_LINE2_HEIGHT + th) // 2 - 4
        cv2.putText(
            sheet, label, (tx, ty),
            cv2.FONT_HERSHEY_DUPLEX, 1.4, text_color, 3, cv2.LINE_AA,
        )

        csv_rows.append({
            "id": k + 1,
            "source": src,
            "idx": idx,
            "predicted_label": label,
            "your_answer": label,  # 差分のみ修正
        })

    return sheet, csv_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="data/training_phase_u/parallel_relabeled",
    )
    parser.add_argument(
        "--out-dir",
        default="data/verify/phase_u_relabel_check",
    )
    parser.add_argument("--per-color", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"sampling from: {in_dir}")
    print(f"per color:     {args.per_color}")
    reservoirs = collect_samples(in_dir, args.per_color, args.seed)

    for color in COLOR_ORDER:
        print(f"  {COLOR_LABEL[color]}: {len(reservoirs[color])} samples")

    sheet, rows = build_sheet(reservoirs)

    sheet_path = (out_dir / "sheet.png").resolve()
    cv2.imwrite(str(sheet_path), sheet)
    print(f"sheet: {to_windows_path(sheet_path)}")

    csv_path = (out_dir / "labels.csv").resolve()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "id", "source", "idx", "predicted_label", "your_answer",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"csv:   {to_windows_path(csv_path)}")
    print(
        "labels: EM/RED/BLUE/GRN/YEL/PUR/OJM. "
        "your_answer プリセット = predicted。差分のみ修正。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
