"""ラベル付き 288 セルから OjamaCnn を訓練する。

データ拡張:
    - ランダム色ジッタ (brightness ±15%, contrast ±10%)
    - 水平反転 50%
    - わずかな回転 (-5°〜+5°)
    - わずかな平行移動 (-2px〜+2px)

訓練:
    - epoch=80
    - lr=1e-3, Adam
    - 80/20 train/val split (各クラス均等)
    - val accuracy で早期停止 (patience=15)

出力:
    models/ojama_cnn.pt
    data/verify/ojama_cnn_train_log.txt
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.ojama_cnn import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    N_CLASSES,
    OJAMA_CLASSES,
    OjamaCnn,
    save_cnn,
)
from src.ojama_warning import (
    CELL_WIDTH,
    ICON_SAMPLE_HALF,
    P1_BOARD_X,
    P2_BOARD_X,
    WARNING_HEIGHT,
    WARNING_TOP_Y,
)

LABEL_SETS: list[tuple[Path, Path]] = [
    (Path("data/verify/ojama_labels.tsv"),
     Path("data/verify/ojama_label_index.tsv")),
    (Path("data/verify/ojama_labels_v2.tsv"),
     Path("data/verify/ojama_label_index_v2.tsv")),
    (Path("data/verify/ojama_labels_v3.tsv"),
     Path("data/verify/ojama_label_index_v3.tsv")),
    (Path("data/verify/ojama_labels_v4.tsv"),
     Path("data/verify/ojama_label_index_v4.tsv")),
    (Path("data/verify/ojama_labels_v5.tsv"),
     Path("data/verify/ojama_label_index_v5.tsv")),
]

OUTPUT_MODEL_PATH = Path("models/ojama_cnn.pt")
LOG_PATH = Path("data/verify/ojama_cnn_train_log.txt")
EXPECTED_FRAME_SHAPE: tuple[int, int] = (1080, 1920)

# ユーザラベル → OJAMA_CLASSES の名前変換
LABEL_TO_CLASS: dict[str, str] = {
    "empty": "empty",
    "small": "small",
    "large": "line",
    "rock": "rock",
    "star": "big_crown",
    "moon": "moon",
    "crown": "crown",
}

# 訓練ハイパラ (強化データ拡張に合わせて epoch 増、patience 増)
EPOCHS: int = 120
BATCH_SIZE: int = 32
LR: float = 1e-3
PATIENCE: int = 25
SEED: int = 42
# label smoothing (過学習抑制)
LABEL_SMOOTHING: float = 0.05


def get_frame(video_path: Path, t_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != EXPECTED_FRAME_SHAPE:
        frame = cv2.resize(
            frame,
            (EXPECTED_FRAME_SHAPE[1], EXPECTED_FRAME_SHAPE[0]),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def extract_center_patch(
    frame: np.ndarray, side: str, cell_idx: int
) -> np.ndarray:
    base_x = P1_BOARD_X if side == "1P" else P2_BOARD_X
    cell_x1 = base_x + cell_idx * CELL_WIDTH
    cell_cx = cell_x1 + CELL_WIDTH // 2
    cell_cy = WARNING_TOP_Y + WARNING_HEIGHT // 2
    half = ICON_SAMPLE_HALF
    return frame[
        cell_cy - half: cell_cy + half,
        cell_cx - half: cell_cx + half,
    ].copy()


def collect_dataset() -> tuple[list[np.ndarray], list[int], list[tuple[int, int]]]:
    """全ラベルセットからパッチとクラスインデックスを集める。
    Returns: (patches, labels, frame_keys (set_id, frame_idx))
    """
    class_to_idx = {c: i for i, c in enumerate(OJAMA_CLASSES)}
    patches: list[np.ndarray] = []
    labels: list[int] = []
    frame_keys: list[tuple[int, int]] = []
    frame_cache: dict[tuple[str, float], np.ndarray | None] = {}

    for set_id, (labels_path, index_path) in enumerate(LABEL_SETS):
        if not labels_path.is_file() or not index_path.is_file():
            continue
        idx_map: dict[tuple[int, str, int], tuple[float, str]] = {}
        with open(index_path, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                key = (int(r["frame_idx"]), r["side"], int(r["cell_idx"]))
                idx_map[key] = (float(r["t_sec"]), r["video"])
        with open(labels_path, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                key = (int(r["frame_idx"]), r["side"], int(r["cell_idx"]))
                info = idx_map.get(key)
                if info is None:
                    continue
                raw_label = r["label"]
                cls = LABEL_TO_CLASS.get(raw_label)
                if cls is None or cls not in class_to_idx:
                    continue
                t_sec, vid = info
                cache_key = (vid, t_sec)
                if cache_key not in frame_cache:
                    frame_cache[cache_key] = get_frame(
                        Path(f"data/frames/{vid}.mp4"), t_sec,
                    )
                frame = frame_cache[cache_key]
                if frame is None:
                    continue
                patch = extract_center_patch(frame, r["side"],
                                              int(r["cell_idx"]))
                if patch.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
                    continue
                patches.append(patch)
                labels.append(class_to_idx[cls])
                frame_keys.append((set_id, int(r["frame_idx"])))
    return patches, labels, frame_keys


def augment_patch(
    patch: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """データ拡張 (強化版 2026-04-27): 水平反転、ColorJitter、回転、スケール、
    平行移動、RandomErasing。"""
    out = patch.copy()
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    alpha = float(rng.uniform(0.75, 1.25))
    beta = float(rng.uniform(-25, 25))
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        sat_shift = int(rng.uniform(-20, 20))
        hsv[..., 1] = np.clip(hsv[..., 1] + sat_shift, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    angle = float(rng.uniform(-10.0, 10.0))
    scale = float(rng.uniform(0.9, 1.1))
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += float(rng.uniform(-3.0, 3.0))
    M[1, 2] += float(rng.uniform(-3.0, 3.0))
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.3:
        eh = int(rng.uniform(4, h // 3))
        ew = int(rng.uniform(4, w // 3))
        ey = int(rng.uniform(0, h - eh))
        ex = int(rng.uniform(0, w - ew))
        mean_color = out.mean(axis=(0, 1)).astype(np.uint8)
        out[ey:ey + eh, ex:ex + ew] = mean_color
    return out


class OjamaDataset(Dataset):
    def __init__(
        self,
        patches: list[np.ndarray],
        labels: list[int],
        augment: bool = False,
        seed: int = SEED,
    ) -> None:
        self._patches = patches
        self._labels = labels
        self._augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        patch = self._patches[idx]
        if self._augment:
            patch = augment_patch(patch, self._rng)
        x = torch.from_numpy(patch).float() / 255.0
        x = x.permute(2, 0, 1)  # (3, H, W)
        return x, self._labels[idx]


def train_val_split_by_frame(
    frame_keys: list[tuple[int, int]],
    val_ratio: float = 0.20,
    seed: int = SEED,
) -> tuple[list[int], list[int]]:
    """フレーム単位で 80/20 分割 (同フレームの全セルが同じ split に入る)。

    これにより同フレームの異なるセルが train と val 両方に入る過大評価を防ぐ。
    """
    unique_frames = sorted(set(frame_keys))
    rng = random.Random(seed)
    rng.shuffle(unique_frames)
    n_val = max(1, int(len(unique_frames) * val_ratio))
    val_frame_set = set(unique_frames[:n_val])
    train_idx: list[int] = []
    val_idx: list[int] = []
    for i, k in enumerate(frame_keys):
        if k in val_frame_set:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    log("=== ojama CNN training ===")
    log("loading dataset...")
    patches, labels, frame_keys = collect_dataset()
    log(f"  total: {len(patches)} patches")
    cls_counts = defaultdict(int)
    for c in labels:
        cls_counts[OJAMA_CLASSES[c]] += 1
    for c in OJAMA_CLASSES:
        log(f"  {c:10s} {cls_counts.get(c, 0)}")

    train_idx, val_idx = train_val_split_by_frame(frame_keys)
    log(f"train={len(train_idx)} val={len(val_idx)} (frame-level split)")

    train_set = OjamaDataset(
        [patches[i] for i in train_idx],
        [labels[i] for i in train_idx],
        augment=True,
    )
    val_set = OjamaDataset(
        [patches[i] for i in val_idx],
        [labels[i] for i in val_idx],
        augment=False,
    )

    # クラス不均衡対策: WeightedRandomSampler
    class_weights = np.zeros(N_CLASSES, dtype=np.float32)
    for c in labels:
        class_weights[c] += 1
    class_weights = 1.0 / np.maximum(class_weights, 1.0)
    sample_weights = [class_weights[labels[i]] for i in train_idx]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_idx) * 8, replacement=True,
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, sampler=sampler,
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = OjamaCnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val_acc: float = 0.0
    best_epoch: int = -1
    no_improve: int = 0

    for epoch in range(EPOCHS):
        # train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)

        # val
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        log(f"epoch {epoch:3d} train_loss={train_loss/max(1,train_total):.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            save_cnn(model, OUTPUT_MODEL_PATH)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log(f"early stop at epoch {epoch}")
                break

    log(f"\nbest val_acc={best_val_acc:.3f} at epoch {best_epoch}")
    log(f"model saved: {OUTPUT_MODEL_PATH}")

    LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
