"""
data/training/parallel/*.npz (動画毎) + 既存 bulk データを統合
+ 目フィルタ + balance + CNN学習 (CPU強制) + 可視化
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

# torch を CPU 強制 (GPU 転送オーバーヘッドが計算より大きい小型モデル)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.calibration import CalibratedConfig
from src.patch_extraction import PatchDataset, balance_dataset
from src.patch_classifier import (
    CnnPatchClassifier, GatedCnnClassifier, PatchSample,
)
from src.image_reader import ImageReader
from src.board import (
    BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS,
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA, COLOR_UNKNOWN,
)

NAMES = {
    COLOR_EMPTY: "空", COLOR_RED: "赤", COLOR_BLUE: "青", COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫", COLOR_OJAMA: "お邪魔",
}


def has_eyes(p: np.ndarray) -> bool:
    h, w = p.shape[:2]
    mh, mw = int(h*0.15), int(w*0.15)
    c = p[mh:h-mh, mw:w-mw]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    d = (g < 70).astype(np.uint8) * 255
    n, _, s, _ = cv2.connectedComponentsWithStats(d, connectivity=4)
    ta = c.shape[0] * c.shape[1]
    return sum(
        1 for i in range(1, n)
        if 2 <= s[i, cv2.CC_STAT_AREA] <= ta*0.12
    ) >= 2


def main() -> None:
    # parallel ディレクトリの動画毎パッチを読み込む
    all_patches: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    parallel_dir = Path("data/training/parallel")
    for f in sorted(parallel_dir.glob("*.npz")):
        data = np.load(f)
        if "patches" not in data.files:
            continue
        all_patches.append(data["patches"])
        all_labels.append(data["labels"])
        print(f"  読込: {f.name} ({len(data['labels'])})")

    # 既存の bulk データ (video 01-08 のバランス済み) も統合
    bulk_prev = Path("data/training/bulk_patches_balanced_through_v08.npz")
    if bulk_prev.exists():
        ds_bulk = PatchDataset.load(bulk_prev)
        all_patches.append(ds_bulk.patches)
        all_labels.append(ds_bulk.labels)
        print(f"  継承: bulk_v08 ({len(ds_bulk.labels)})")

    # 既存の multi3 (1st run で作成) も追加
    prev = Path("data/training/multi3_patches_balanced.npz")
    if prev.exists():
        ds = PatchDataset.load(prev)
        all_patches.append(ds.patches)
        all_labels.append(ds.labels)
        print(f"  継承: multi3 ({len(ds.labels)})")

    if not all_patches:
        print("データなし")
        return

    patches = np.concatenate(all_patches)
    labels = np.concatenate(all_labels)
    print(f"\n統合前: {len(labels)}")

    # 目フィルタ
    keep = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        e = has_eyes(patches[i])
        keep[i] = (not e) if labels[i] == COLOR_EMPTY else e
    patches = patches[keep]; labels = labels[keep]
    print(f"目フィルタ後: {len(labels)}")

    ds = PatchDataset(patches=patches, labels=labels)
    ds.stats.patches_total = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    ds.stats.per_class_count = {int(k): int(v) for k, v in zip(unique, counts)}

    balanced = balance_dataset(ds, empty_ratio_cap=0.35)
    balanced.save(Path("data/training/final_merged_patches.npz"))
    print(f"\nバランス後: {balanced.stats.patches_total}")
    for k in sorted(NAMES.keys()):
        print(f"  {NAMES[k]}: {balanced.stats.per_class_count.get(k, 0)}")

    # CNN学習
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

    cnn = CnnPatchClassifier()
    start = time.time()
    losses = cnn.fit(train, epochs=30, lr=0.005, batch_size=256)
    print(f"\n学習: {time.time()-start:.1f}s loss={losses[0]:.3f}→{losses[-1]:.3f}")
    print(f"val={cnn.accuracy(val):.4f} test={cnn.accuracy(test):.4f}")

    y_t = np.array([s.color for s in test])
    y_p = np.array([cnn.classify(s.patch) for s in test])
    for code in sorted(NAMES.keys()):
        m = y_t == code
        if m.sum() == 0:
            continue
        print(f"  {NAMES[code]} (n={m.sum()}): {(y_p[m] == code).mean():.4f}")

    cnn.save(Path("models/cnn_final_v19.pt"))
    print("\n保存: models/cnn_final_v19.pt")

    # 可視化
    config = CalibratedConfig.load("models/calibration_video01.json")
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(
        classifier=gated,
        p1_region=config.p1_region, p2_region=config.p2_region,
    )
    out_dir = Path("data/verify/accuracy_v19")
    out_dir.mkdir(parents=True, exist_ok=True)
    COLOR_VIS = {0:(80,80,80), 1:(60,60,255), 2:(255,120,60), 3:(80,220,80),
                 4:(60,220,240), 5:(200,80,220), 9:(210,210,210), 10:(128,0,128)}
    LBL = {0:"_", 1:"R", 2:"B", 3:"G", 4:"Y", 5:"P", 9:"O", 10:"?"}

    for sec in [600, 900, 1500, 2100, 2700, 3200]:
        frame = cv2.imread(f"data/frames/sample/frame_{sec:04d}s.png")
        if frame is None: continue
        b1, b2 = reader.read_both_boards(frame)
        pages = []
        for player_label, board, region in [("1P", b1, config.p1_region), ("2P", b2, config.p2_region)]:
            m = 20
            x1, y1 = max(0, region.x-m), max(0, region.y-m)
            x2 = min(frame.shape[1], region.x+region.width+m)
            y2 = min(frame.shape[0], region.y+region.height+m)
            raw = frame[y1:y2, x1:x2].copy(); ovl = raw.copy()
            rx, ry = region.x-x1, region.y-y1
            cw, ch = region.cell_width, region.cell_height
            for row in range(HIDDEN_ROWS, BOARD_ROWS):
                vr = row - HIDDEN_ROWS
                for col in range(BOARD_COLS):
                    color = board.get(row, col)
                    cx, cy = int(rx+(col+0.5)*cw), int(ry+(vr+0.5)*ch)
                    hw, hh = int(cw/2)-2, int(ch/2)-2
                    tmp = ovl.copy()
                    cv2.rectangle(tmp, (cx-hw, cy-hh), (cx+hw, cy+hh), COLOR_VIS[color], -1)
                    cv2.addWeighted(tmp, 0.35, ovl, 0.65, 0, ovl)
                    cv2.rectangle(ovl, (cx-hw, cy-hh), (cx+hw, cy+hh), COLOR_VIS[color], 2)
                    lc = (255,255,255) if sum(COLOR_VIS[color]) < 400 else (0,0,0)
                    cv2.putText(ovl, LBL[color], (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 2)
            th = 40
            rt = np.vstack([np.full((th, raw.shape[1], 3), 30, np.uint8), raw])
            ot = np.vstack([np.full((th, ovl.shape[1], 3), 30, np.uint8), ovl])
            cv2.putText(rt, f"{player_label} Original", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(ot, f"{player_label} v19 final", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            pages.append(np.hstack([rt, ot]))
        if pages[0].shape[1] != pages[1].shape[1]:
            w = max(p.shape[1] for p in pages)
            for i in range(len(pages)):
                if pages[i].shape[1] < w:
                    pad = np.full((pages[i].shape[0], w-pages[i].shape[1], 3), 30, np.uint8)
                    pages[i] = np.hstack([pages[i], pad])
        gap = np.full((30, pages[0].shape[1], 3), 20, np.uint8)
        full = np.vstack([pages[0], gap, pages[1]])
        cv2.imwrite(str(out_dir / f"compare_{sec:04d}s.png"), full)
    print(f"可視化 → {out_dir}")


if __name__ == "__main__":
    main()
