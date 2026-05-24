"""CNN 単体の精度 (テンプレ等のフォールバック排除) を v1+v2 全 288 セルで測定。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from src.ojama_cnn import load_cnn
from src.ojama_warning import (
    CELL_WIDTH,
    ICON_SAMPLE_HALF,
    P1_BOARD_X,
    P2_BOARD_X,
    WARNING_HEIGHT,
    WARNING_TOP_Y,
)

LABEL_TO_CLASS = {
    "empty": "empty", "small": "small", "large": "line", "rock": "rock",
    "star": "big_crown", "moon": "moon", "crown": "crown",
}
LABEL_SETS = [
    ("v1", "data/verify/ojama_labels.tsv", "data/verify/ojama_label_index.tsv"),
    ("v2", "data/verify/ojama_labels_v2.tsv", "data/verify/ojama_label_index_v2.tsv"),
    ("v3", "data/verify/ojama_labels_v3.tsv", "data/verify/ojama_label_index_v3.tsv"),
    ("v4", "data/verify/ojama_labels_v4.tsv", "data/verify/ojama_label_index_v4.tsv"),
    ("v5", "data/verify/ojama_labels_v5.tsv", "data/verify/ojama_label_index_v5.tsv"),
]


def evaluate(label_path: str, index_path: str) -> tuple[int, int, list]:
    cnn = load_cnn()
    if cnn is None:
        return 0, 0, []
    labels: dict = {}
    with open(label_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            labels[(int(r["frame_idx"]), r["side"], int(r["cell_idx"]))] = r["label"]
    idx: dict = {}
    with open(index_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            idx[(int(r["frame_idx"]), r["side"], int(r["cell_idx"]))] = (
                float(r["t_sec"]), r["video"]
            )

    frame_cache: dict = {}
    n_total = 0
    n_correct = 0
    errors: list = []
    confs: list[float] = []
    for key, lbl in labels.items():
        info = idx.get(key)
        if not info:
            continue
        t, vid = info
        if (vid, t) not in frame_cache:
            cap = cv2.VideoCapture(f"data/frames/{vid}.mp4")
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            cap.release()
            if ok and f is not None and f.shape[:2] != (1080, 1920):
                f = cv2.resize(f, (1920, 1080))
            frame_cache[(vid, t)] = f
        frame = frame_cache[(vid, t)]
        if frame is None:
            continue
        fi, side, ci = key
        base_x = P1_BOARD_X if side == "1P" else P2_BOARD_X
        cx = base_x + int((ci + 0.5) * CELL_WIDTH)
        cy = WARNING_TOP_Y + WARNING_HEIGHT // 2
        h = ICON_SAMPLE_HALF
        patch = frame[cy - h: cy + h, cx - h: cx + h]
        if patch.shape[:2] != (36, 36):
            continue
        pred, conf = cnn.predict_class(patch)
        confs.append(conf)
        truth = LABEL_TO_CLASS.get(lbl, lbl)
        n_total += 1
        if pred == truth:
            n_correct += 1
        else:
            errors.append((key, lbl, truth, pred, conf))
    return n_correct, n_total, errors


for tag, lp, ip in LABEL_SETS:
    correct, total, errors = evaluate(lp, ip)
    if total == 0:
        continue
    print(f"\n=== {tag} CNN raw ===")
    print(f"  {correct}/{total} = {correct/total:.3f}")
    print(f"  errors: {len(errors)}")
    for e in errors[:8]:
        print(f"    F{e[0][0]} {e[0][1]} S{e[0][2]} truth={e[1]} pred={e[3]} conf={e[4]:.3f}")
