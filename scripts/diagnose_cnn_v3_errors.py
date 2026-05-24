"""v3 で CNN が誤判定したセルを目視可能な grid 画像にして診断する。

各誤判定セルについて:
    - 元 cell 画像 (拡大)
    - 中央 36×36 patch (CNN 入力)
    - truth ラベル
    - CNN 予測 + confidence

を並べて出力。背景・cell 位置・アニメ余韻などの問題を目視で特定。
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.ojama_cnn import load_cnn
from src.ojama_warning import (
    CELL_WIDTH,
    ICON_SAMPLE_HALF,
    P1_BOARD_X,
    P2_BOARD_X,
    WARNING_BOTTOM_Y,
    WARNING_HEIGHT,
    WARNING_TOP_Y,
)

LABEL_PATH = Path("data/verify/ojama_labels_v3.tsv")
INDEX_PATH = Path("data/verify/ojama_label_index_v3.tsv")
OUT = Path("data/verify/cnn_v3_errors_grid.png")
ORDER_OUT = Path("data/verify/cnn_v3_errors_order.tsv")

LABEL_TO_CLASS = {
    "empty": "empty", "small": "small", "large": "line", "rock": "rock",
    "star": "big_crown", "moon": "moon", "crown": "crown",
}


def get_frame(video_path: Path, t_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def main() -> int:
    cnn = load_cnn()
    if cnn is None:
        print("CNN モデルなし")
        return 1

    labels: dict = {}
    with open(LABEL_PATH) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            labels[(int(r["frame_idx"]), r["side"], int(r["cell_idx"]))] = r["label"]
    idx: dict = {}
    with open(INDEX_PATH) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            idx[(int(r["frame_idx"]), r["side"], int(r["cell_idx"]))] = (
                float(r["t_sec"]), r["video"]
            )

    frame_cache: dict = {}
    error_panels: list[np.ndarray] = []
    order_rows: list[dict] = []
    for key, lbl in labels.items():
        info = idx.get(key)
        if not info:
            continue
        t, vid = info
        if (vid, t) not in frame_cache:
            frame_cache[(vid, t)] = get_frame(Path(f"data/frames/{vid}.mp4"), t)
        frame = frame_cache[(vid, t)]
        if frame is None:
            continue
        fi, side, ci = key
        base_x = P1_BOARD_X if side == "1P" else P2_BOARD_X
        # 大きい cell (画像表示用)
        cell_x1 = base_x + ci * CELL_WIDTH
        cell = frame[
            WARNING_TOP_Y:WARNING_BOTTOM_Y,
            cell_x1:cell_x1 + CELL_WIDTH,
        ]
        # 中央 36x36 patch (CNN 入力)
        cx = cell_x1 + CELL_WIDTH // 2
        cy = WARNING_TOP_Y + WARNING_HEIGHT // 2
        h = ICON_SAMPLE_HALF
        patch = frame[cy - h: cy + h, cx - h: cx + h]
        if patch.shape[:2] != (36, 36):
            continue
        pred, conf = cnn.predict_class(patch)
        truth = LABEL_TO_CLASS.get(lbl, lbl)
        if pred == truth:
            continue
        # 誤判定セル: パネル作成
        cell_big = cv2.resize(cell, (cell.shape[1] * 4, cell.shape[0] * 4),
                              interpolation=cv2.INTER_NEAREST)
        patch_big = cv2.resize(patch, (patch.shape[1] * 4, patch.shape[0] * 4),
                                interpolation=cv2.INTER_NEAREST)
        # 横並び (cell_big + sep + patch_big)
        sep = np.full((cell_big.shape[0], 8, 3), 60, dtype=np.uint8)
        # patch_big の高さに合わせて cell_big を crop or pad
        target_h = max(cell_big.shape[0], patch_big.shape[0])
        if cell_big.shape[0] < target_h:
            pad = np.zeros((target_h - cell_big.shape[0],
                            cell_big.shape[1], 3), dtype=np.uint8)
            cell_big = np.vstack([cell_big, pad])
        if patch_big.shape[0] < target_h:
            pad = np.zeros((target_h - patch_big.shape[0],
                            patch_big.shape[1], 3), dtype=np.uint8)
            patch_big = np.vstack([patch_big, pad])
        body = np.hstack([cell_big, sep, patch_big])
        bar = np.zeros((48, body.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, f"F{fi} {side} S{ci} | {vid} t={t:.1f}s",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 200, 100), 1)
        cv2.putText(bar, f"truth={lbl}({truth}) pred={pred} conf={conf:.2f}",
                    (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1)
        error_panels.append(np.vstack([bar, body]))
        order_rows.append({
            "order_idx": len(order_rows),
            "frame_idx": fi,
            "side": side,
            "cell_idx": ci,
            "video": vid,
            "t_sec": round(t, 2),
            "old_truth": lbl,
            "old_truth_impl": truth,
            "cnn_pred": pred,
            "cnn_conf": round(conf, 3),
        })

    # 順序情報を TSV 出力
    with open(ORDER_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, delimiter="\t",
            fieldnames=list(order_rows[0].keys()) if order_rows else [],
        )
        w.writeheader()
        w.writerows(order_rows)
    print(f"errors: {len(error_panels)}")
    print(f"order: {ORDER_OUT}")
    if not error_panels:
        return 0

    # 縦結合
    sep = np.full((6, error_panels[0].shape[1], 3), 30, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for p in error_panels:
        parts.append(p)
        parts.append(sep)
    grid = np.vstack(parts[:-1])
    cv2.imwrite(str(OUT), grid)
    print(f"出力: {OUT} (shape={grid.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
