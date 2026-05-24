"""
アジャイル評価サイクルスクリプト

1. parallel/*.npz を統合 + 既存データとマージ
2. 目フィルタ + バランス調整
3. CNN学習 (CPU強制)
4. クラス別精度レポート
5. 実フレームで盤面読み取り → 8指標算出 → 可視化
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

# torch を CPU 強制
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.board import (
    BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS,
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)
from src.calibration import CalibratedConfig
from src.chain import ChainSimulator
from src.image_reader import ImageReader
from src.indicators import IndicatorCalculator, IndicatorSet
from src.patch_classifier import (
    CnnPatchClassifier, GatedCnnClassifier, PatchSample,
    COLOR_TO_CLASS_INDEX, NUM_CLASSES,
)
from src.patch_extraction import PatchDataset, balance_dataset
from src.scorer import Scorer

NAMES = {
    COLOR_EMPTY: "空", COLOR_RED: "赤", COLOR_BLUE: "青", COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫", COLOR_OJAMA: "お邪魔",
}
COLOR_VIS = {
    0: (80, 80, 80), 1: (60, 60, 255), 2: (255, 120, 60),
    3: (80, 220, 80), 4: (60, 220, 240), 5: (200, 80, 220),
    9: (210, 210, 210),
}
LBL = {0: "_", 1: "R", 2: "B", 3: "G", 4: "Y", 5: "P", 9: "O"}


def has_eyes(p: np.ndarray) -> bool:
    """目検出フィルタ。"""
    h, w = p.shape[:2]
    mh, mw = int(h * 0.15), int(w * 0.15)
    c = p[mh:h - mh, mw:w - mw]
    if c.size == 0:
        return False
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    d = (g < 70).astype(np.uint8) * 255
    n, _, s, _ = cv2.connectedComponentsWithStats(d, connectivity=4)
    ta = c.shape[0] * c.shape[1]
    return sum(
        1 for i in range(1, n)
        if 2 <= s[i, cv2.CC_STAT_AREA] <= ta * 0.12
    ) >= 2


def load_and_merge() -> tuple[np.ndarray, np.ndarray]:
    """parallel/*.npz + 既存データを統合。"""
    all_p, all_l = [], []

    # parallel (新規)
    pdir = Path("data/training/parallel")
    for f in sorted(pdir.glob("*.npz")):
        data = np.load(f)
        if "patches" in data:
            all_p.append(data["patches"])
            all_l.append(data["labels"])
            print(f"  {f.name}: {len(data['labels'])}")

    # 既存 multi3
    prev = Path("data/training/multi3_patches_balanced.npz")
    if prev.exists():
        ds = PatchDataset.load(prev)
        all_p.append(ds.patches)
        all_l.append(ds.labels)
        print(f"  (既存) multi3: {len(ds.labels)}")

    patches = np.concatenate(all_p)
    labels = np.concatenate(all_l)
    return patches, labels


def apply_eye_filter(patches: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """目フィルタ: 空セルは目なし、色セルは目あり。"""
    keep = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        e = has_eyes(patches[i])
        keep[i] = (not e) if labels[i] == COLOR_EMPTY else e
    return patches[keep], labels[keep]


def train_cnn(patches: np.ndarray, labels: np.ndarray) -> tuple[CnnPatchClassifier, dict]:
    """CNN学習 + 評価。"""
    ds = PatchDataset(patches=patches, labels=labels)
    balanced = balance_dataset(ds, empty_ratio_cap=0.35)
    print(f"\nバランス後: {balanced.stats.patches_total}")
    for k in sorted(NAMES.keys()):
        print(f"  {NAMES[k]}: {balanced.stats.per_class_count.get(k, 0)}")

    N = len(balanced.labels)
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)

    def to_samples(indices):
        return [
            PatchSample(patch=balanced.patches[i], color=int(balanced.labels[i]))
            for i in indices
        ]

    train = to_samples(perm[:int(N * 0.8)])
    val = to_samples(perm[int(N * 0.8):int(N * 0.9)])
    test = to_samples(perm[int(N * 0.9):])

    cnn = CnnPatchClassifier()
    t0 = time.time()
    losses = cnn.fit(train, epochs=30, lr=0.005, batch_size=256)
    elapsed = time.time() - t0
    val_acc = cnn.accuracy(val)
    test_acc = cnn.accuracy(test)

    print(f"\n学習: {elapsed:.1f}s  loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    print(f"val: {val_acc:.4f}  test: {test_acc:.4f}")

    # クラス別精度
    report = {}
    y_true = np.array([s.color for s in test])
    y_pred = np.array([cnn.classify(s.patch) for s in test])
    for code in sorted(NAMES.keys()):
        mask = y_true == code
        if mask.sum() == 0:
            continue
        acc = float((y_pred[mask] == code).mean())
        report[NAMES[code]] = {"n": int(mask.sum()), "acc": acc}
        print(f"  {NAMES[code]} (n={mask.sum()}): {acc:.4f}")

    # 混同行列サマリ (間違いが多いペアを表示)
    print("\n主な誤分類:")
    for code_t in sorted(NAMES.keys()):
        mask_t = y_true == code_t
        if mask_t.sum() == 0:
            continue
        wrong = y_pred[mask_t] != code_t
        if wrong.sum() == 0:
            continue
        wrong_preds = y_pred[mask_t][wrong]
        uniq, counts = np.unique(wrong_preds, return_counts=True)
        for u, c in sorted(zip(uniq, counts), key=lambda x: -x[1])[:2]:
            if c >= 3:
                print(f"  {NAMES[code_t]} → {NAMES.get(int(u), '?')}: {c}件")

    return cnn, report


def evaluate_indicators(cnn: CnnPatchClassifier, config: CalibratedConfig) -> None:
    """実フレームで指標を算出し、妥当性を確認。"""
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(
        classifier=gated,
        p1_region=config.p1_region, p2_region=config.p2_region,
    )
    calc = IndicatorCalculator()
    scorer = Scorer()

    sample_dir = Path("data/frames/sample")
    if not sample_dir.exists():
        print("\nサンプルフレームなし — 指標評価スキップ")
        return

    print("\n=== 指標評価 ===")
    for fp in sorted(sample_dir.glob("frame_*.png"))[:8]:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        b1, b2 = reader.read_both_boards(frame)

        iset_1p = calc.compute_all(b1)
        iset_2p = calc.compute_all(b2)
        result = scorer.score(iset_1p, iset_2p)

        print(f"\n--- {fp.name} ---")
        print(f"  スコア: {result.total_score:+.1f} ({result.advantage_side()})")
        for name in iset_1p.results:
            v1 = iset_1p.score_of(name)
            v2 = iset_2p.score_of(name)
            print(f"  {name:30s}  1P={v1:.3f}  2P={v2:.3f}  diff={v1-v2:+.3f}")


def visualize(cnn: CnnPatchClassifier, config: CalibratedConfig) -> None:
    """読み取り結果の可視化画像を生成。"""
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(
        classifier=gated,
        p1_region=config.p1_region, p2_region=config.p2_region,
    )

    out_dir = Path("data/verify/eval_cycle")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = Path("data/frames/sample")
    if not sample_dir.exists():
        print("サンプルフレームなし — 可視化スキップ")
        return

    for fp in sorted(sample_dir.glob("frame_*.png"))[:6]:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        b1, b2 = reader.read_both_boards(frame)
        pages = []
        for label, board, region in [("1P", b1, config.p1_region), ("2P", b2, config.p2_region)]:
            m = 20
            x1 = max(0, region.x - m)
            y1 = max(0, region.y - m)
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
                    cv2.rectangle(tmp, (cx - hw, cy - hh), (cx + hw, cy + hh),
                                  COLOR_VIS.get(color, (128, 128, 128)), -1)
                    cv2.addWeighted(tmp, 0.35, ovl, 0.65, 0, ovl)
                    cv2.rectangle(ovl, (cx - hw, cy - hh), (cx + hw, cy + hh),
                                  COLOR_VIS.get(color, (128, 128, 128)), 2)
                    lc = (255, 255, 255) if sum(COLOR_VIS.get(color, (128,128,128))) < 400 else (0, 0, 0)
                    cv2.putText(ovl, LBL.get(color, "?"), (cx - 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 2)
            th = 40
            rt = np.vstack([np.full((th, raw.shape[1], 3), 30, np.uint8), raw])
            ot = np.vstack([np.full((th, ovl.shape[1], 3), 30, np.uint8), ovl])
            cv2.putText(rt, f"{label} Original", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(ot, f"{label} eval_cycle", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            pages.append(np.hstack([rt, ot]))
        if pages[0].shape[1] != pages[1].shape[1]:
            w = max(p.shape[1] for p in pages)
            for i in range(len(pages)):
                if pages[i].shape[1] < w:
                    pad = np.full((pages[i].shape[0], w - pages[i].shape[1], 3), 30, np.uint8)
                    pages[i] = np.hstack([pages[i], pad])
        gap = np.full((30, pages[0].shape[1], 3), 20, np.uint8)
        full = np.vstack([pages[0], gap, pages[1]])
        out_path = out_dir / f"eval_{fp.stem}.png"
        cv2.imwrite(str(out_path), full)

    print(f"可視化 → {out_dir}")


def main() -> None:
    print("=== データ統合 ===")
    patches, labels = load_and_merge()
    print(f"統合: {len(labels)} patches")

    print("\n=== 目フィルタ ===")
    patches, labels = apply_eye_filter(patches, labels)
    print(f"フィルタ後: {len(labels)}")

    print("\n=== CNN学習 ===")
    cnn, report = train_cnn(patches, labels)

    # モデル保存
    model_path = Path("models/cnn_eval_cycle.pt")
    cnn.save(model_path)
    print(f"\nモデル保存: {model_path}")

    # 指標評価
    config = CalibratedConfig.load("models/calibration_video01.json")
    evaluate_indicators(cnn, config)

    # 可視化
    print("\n=== 可視化 ===")
    visualize(cnn, config)

    print("\n=== 評価完了 ===")


if __name__ == "__main__":
    main()
