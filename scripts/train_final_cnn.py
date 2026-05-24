"""
最終 bulk データで CNN 学習 + 可視化。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.calibration import CalibratedConfig
from src.patch_extraction import PatchDataset
from src.patch_classifier import (
    CnnPatchClassifier, GatedCnnClassifier, PatchSample,
)
from src.image_reader import ImageReader
from src.board import (
    BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS,
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA, COLOR_UNKNOWN,
)

NAMES_JA = {
    COLOR_EMPTY: "空", COLOR_RED: "赤", COLOR_BLUE: "青", COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫", COLOR_OJAMA: "お邪魔",
}


def train_cnn(balanced: PatchDataset) -> tuple[CnnPatchClassifier, dict]:
    N = len(balanced.labels)
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    to_s = lambda ii: [
        PatchSample(patch=balanced.patches[i], color=int(balanced.labels[i]))
        for i in ii
    ]
    train = to_s(perm[:int(N*0.8)])
    val = to_s(perm[int(N*0.8):int(N*0.9)])
    test = to_s(perm[int(N*0.9):])

    print(f"train={len(train)} val={len(val)} test={len(test)}")
    cnn = CnnPatchClassifier()
    start = time.time()
    losses = cnn.fit(train, epochs=30, lr=0.005, batch_size=128)
    print(f"学習: {time.time()-start:.1f}秒 loss={losses[0]:.3f}→{losses[-1]:.3f}")
    print(f"val={cnn.accuracy(val):.4f} test={cnn.accuracy(test):.4f}")

    y_t = np.array([s.color for s in test])
    y_p = np.array([cnn.classify(s.patch) for s in test])
    class_acc = {}
    for code in sorted(NAMES_JA.keys()):
        m = y_t == code
        if m.sum() == 0:
            continue
        acc = (y_p[m] == code).mean()
        class_acc[NAMES_JA[code]] = (int(m.sum()), float(acc))
        print(f"  {NAMES_JA[code]} (n={m.sum()}): {acc:.4f}")
    return cnn, class_acc


def visualize(cnn: CnnPatchClassifier, config: CalibratedConfig, tag: str) -> None:
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(
        classifier=gated,
        p1_region=config.p1_region,
        p2_region=config.p2_region,
    )
    COLOR_VIS = {
        0: (80, 80, 80), 1: (60, 60, 255), 2: (255, 120, 60),
        3: (80, 220, 80), 4: (60, 220, 240), 5: (200, 80, 220),
        9: (210, 210, 210), 10: (128, 0, 128),
    }
    LBL = {0: "_", 1: "R", 2: "B", 3: "G", 4: "Y", 5: "P", 9: "O", 10: "?"}
    out_dir = Path(f"data/verify/accuracy_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def render(frame, board, region, label):
        m = 20
        x1, y1 = max(0, region.x - m), max(0, region.y - m)
        x2 = min(frame.shape[1], region.x + region.width + m)
        y2 = min(frame.shape[0], region.y + region.height + m)
        raw = frame[y1:y2, x1:x2].copy()
        ovl = raw.copy()
        rx, ry = region.x - x1, region.y - y1
        cw, ch = region.cell_width, region.cell_height
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            vr = row - HIDDEN_ROWS
            for col in range(BOARD_COLS):
                color = board.get(row, col)
                cx = int(rx + (col + 0.5) * cw)
                cy = int(ry + (vr + 0.5) * ch)
                hw, hh = int(cw / 2) - 2, int(ch / 2) - 2
                tmp = ovl.copy()
                cv2.rectangle(
                    tmp, (cx - hw, cy - hh), (cx + hw, cy + hh),
                    COLOR_VIS[color], -1,
                )
                cv2.addWeighted(tmp, 0.35, ovl, 0.65, 0, ovl)
                cv2.rectangle(
                    ovl, (cx - hw, cy - hh), (cx + hw, cy + hh),
                    COLOR_VIS[color], 2,
                )
                lc = (255, 255, 255) if sum(COLOR_VIS[color]) < 400 else (0, 0, 0)
                cv2.putText(
                    ovl, LBL[color], (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 2,
                )
        th = 40
        rt = np.vstack([
            np.full((th, raw.shape[1], 3), 30, np.uint8), raw,
        ])
        ot = np.vstack([
            np.full((th, ovl.shape[1], 3), 30, np.uint8), ovl,
        ])
        cv2.putText(
            rt, f"{label} Original", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            ot, f"{label} {tag}", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        return np.hstack([rt, ot])

    for sec in [600, 900, 1500, 2100, 2700, 3200]:
        frame = cv2.imread(f"data/frames/sample/frame_{sec:04d}s.png")
        b1, b2 = reader.read_both_boards(frame)
        p1 = render(frame, b1, config.p1_region, "1P")
        p2 = render(frame, b2, config.p2_region, "2P")
        if p1.shape[1] != p2.shape[1]:
            w = max(p1.shape[1], p2.shape[1])
            if p1.shape[1] < w:
                p1 = np.hstack([
                    p1, np.full((p1.shape[0], w - p1.shape[1], 3), 30, np.uint8),
                ])
            else:
                p2 = np.hstack([
                    p2, np.full((p2.shape[0], w - p2.shape[1], 3), 30, np.uint8),
                ])
        gap = np.full((30, p1.shape[1], 3), 20, np.uint8)
        cv2.imwrite(
            str(out_dir / f"compare_{sec:04d}s.png"),
            np.vstack([p1, gap, p2]),
        )
    print(f"可視化 → {out_dir}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/training/bulk_patches_balanced_through_v41.npz"
    )
    balanced = PatchDataset.load(path)
    print(f"データ: {path.name} {balanced.stats.patches_total} パッチ")

    cnn, class_acc = train_cnn(balanced)

    tag = "v17"
    model_path = Path(f"models/cnn_bulk_{tag}.pt")
    cnn.save(model_path)
    print(f"保存: {model_path}")

    config = CalibratedConfig.load("models/calibration_video01.json")
    visualize(cnn, config, tag)


if __name__ == "__main__":
    main()
