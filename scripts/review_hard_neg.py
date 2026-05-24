"""hard_neg_v02 のパッチを大きく一覧表示（人手レビュー用）。"""
from __future__ import annotations
import os, sys, csv
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

INDIV = Path("data/verify/hard_neg_v02/individual")
TSV = Path("data/verify/hard_neg_v02/report.tsv")
OUT = Path("data/verify/hard_neg_v02/review_grid.png")

TILE = 200             # パッチ表示サイズ
LABEL_H = 50           # ラベル帯
COLS = 5               # 5 列にして大きく見せる


def main() -> int:
    if not TSV.exists():
        print(f"TSV なし: {TSV}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with TSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)

    files = sorted(INDIV.glob("*.png"))
    # idx と file を対応付け（ファイル名先頭の番号で）
    files_by_idx = {int(p.name.split("_")[0]): p for p in files}

    n = len(rows)
    rows_grid = (n + COLS - 1) // COLS
    grid = np.full((rows_grid * (TILE + LABEL_H), COLS * TILE, 3), 16, dtype=np.uint8)

    for i, row in enumerate(rows):
        idx = int(row["idx"])
        img_path = files_by_idx.get(idx)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        resized = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
        gr = i // COLS
        gc = i % COLS
        y0 = gr * (TILE + LABEL_H)
        x0 = gc * TILE
        grid[y0:y0 + TILE, x0:x0 + TILE] = resized
        # ラベル帯
        label_y0 = y0 + TILE
        cv2.rectangle(grid, (x0, label_y0), (x0 + TILE, label_y0 + LABEL_H),
                      (50, 50, 50), -1)
        line1 = f"#{idx}  m{row['match']}  {row['side']} r{row['row']}c{row['col']}"
        line2 = f"CNN={row['cnn_pred']}  R={row['red_ratio']}"
        cv2.putText(grid, line1, (x0 + 6, label_y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(grid, line2, (x0 + 6, label_y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(OUT), grid)
    print(f"レビュー用グリッド: {OUT}")
    print(f"  {n} 件 × 5 列 × {TILE}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
