"""
自動反復改善スクリプト — test精度99%を目指して自動で回し続ける。

各ラウンド:
  1. 全データ統合 + 目フィルタ + バランス
  2. CNN学習 (ハード例オーバーサンプリング + Focal Loss + 色拡張)
  3. 精度評価 + 混同行列
  4. 改善あり → モデル保存 → 次ラウンド
  5. 改善なし or 99%到達 → 終了
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# CPU 強制
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.optim as optim

from src.board import (
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)
from src.calibration import CalibratedConfig
from src.patch_classifier import (
    CnnPatchClassifier, GatedCnnClassifier,
    COLOR_TO_CLASS_INDEX, CLASS_INDEX_TO_COLOR, NUM_CLASSES,
    PATCH_RESIZE_H, PATCH_RESIZE_W,
)
from src.patch_extraction import PatchDataset, balance_dataset
from src.image_reader import ImageReader
from src.old.indicators import IndicatorCalculator
from src.old.scorer import Scorer

NAMES = {
    COLOR_EMPTY: "空", COLOR_RED: "赤", COLOR_BLUE: "青", COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫", COLOR_OJAMA: "お邪魔",
}

TARGET_ACC = 0.99
MAX_ROUNDS = 20
PATIENCE = 3  # 改善なしが続いたら終了

MODEL_DIR = Path("models")
LOG_PATH = Path("data/iterative_improve_log.txt")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ============================
# データ読み込み
# ============================

def has_eyes(p: np.ndarray) -> bool:
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


def load_all_data() -> tuple[np.ndarray, np.ndarray]:
    """全パッチデータを統合して読み込む。"""
    all_p, all_l = [], []
    pdir = Path("data/training/parallel")
    for f in sorted(pdir.glob("*.npz")):
        data = np.load(f)
        if "patches" in data:
            all_p.append(data["patches"])
            all_l.append(data["labels"])

    prev = Path("data/training/multi3_patches_balanced.npz")
    if prev.exists():
        ds = PatchDataset.load(prev)
        all_p.append(ds.patches)
        all_l.append(ds.labels)

    patches = np.concatenate(all_p)
    labels = np.concatenate(all_l)
    log(f"統合データ: {len(labels)}")

    # 目フィルタ
    keep = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        e = has_eyes(patches[i])
        keep[i] = (not e) if labels[i] == COLOR_EMPTY else e
    patches, labels = patches[keep], labels[keep]
    log(f"目フィルタ後: {len(labels)}")
    return patches, labels


# ============================
# 色拡張 (学習時のデータ拡張)
# ============================

def augment_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """色空間でランダムに拡張する (Hue±8, Saturation±15, Value±15)。"""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.integers(-8, 9)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + rng.integers(-15, 16), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + rng.integers(-15, 16), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ============================
# ハード例マイニング
# ============================

def mine_hard_examples(
    cnn: CnnPatchClassifier,
    patches: np.ndarray,
    labels: np.ndarray,
    max_per_class: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """CNNが間違えるパッチを抽出してオーバーサンプリング用に返す。"""
    hard_p, hard_l = [], []
    for i in range(len(labels)):
        pred = cnn.classify(patches[i])
        if pred != labels[i]:
            hard_p.append(patches[i])
            hard_l.append(labels[i])

    if not hard_p:
        return np.zeros((0,) + patches.shape[1:], dtype=np.uint8), np.zeros(0, dtype=np.int64)

    hard_p = np.array(hard_p)
    hard_l = np.array(hard_l)
    log(f"ハード例: {len(hard_l)} ({len(hard_l)/len(labels)*100:.1f}%)")

    # クラス別内訳
    for code in sorted(NAMES.keys()):
        mask = hard_l == code
        if mask.sum() > 0:
            log(f"  {NAMES[code]}: {mask.sum()}")

    return hard_p, hard_l


# ============================
# Focal Loss
# ============================

class FocalLoss(nn.Module):
    """クラス重み付き Focal Loss。ハード例に損失を集中させる。"""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ============================
# 学習 (拡張版)
# ============================

def train_round(
    patches: np.ndarray,
    labels: np.ndarray,
    hard_patches: np.ndarray | None = None,
    hard_labels: np.ndarray | None = None,
    epochs: int = 40,
    lr: float = 0.003,
    batch_size: int = 256,
    hard_oversample: int = 3,
    use_focal: bool = True,
    seed: int = 42,
) -> tuple[CnnPatchClassifier, dict]:
    """1ラウンドの学習。ハード例オーバーサンプリング + Focal Loss。"""
    # バランス調整
    ds = PatchDataset(patches=patches, labels=labels)
    ds.stats.patches_total = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    ds.stats.per_class_count = {int(k): int(v) for k, v in zip(unique, counts)}
    balanced = balance_dataset(ds, empty_ratio_cap=0.35)

    train_patches = balanced.patches
    train_labels = balanced.labels

    # ハード例オーバーサンプリング
    if hard_patches is not None and len(hard_patches) > 0:
        rng_aug = np.random.default_rng(seed + 1)
        augmented_p, augmented_l = [], []
        for _ in range(hard_oversample):
            for hp, hl in zip(hard_patches, hard_labels):
                augmented_p.append(augment_patch(hp, rng_aug))
                augmented_l.append(hl)
        augmented_p = np.array(augmented_p)
        augmented_l = np.array(augmented_l)
        train_patches = np.concatenate([train_patches, augmented_p])
        train_labels = np.concatenate([train_labels, augmented_l])
        log(f"ハード例追加後: {len(train_labels)} (+{len(augmented_l)})")

    log(f"学習データ: {len(train_labels)}")
    for k in sorted(NAMES.keys()):
        c = (train_labels == k).sum()
        if c > 0:
            log(f"  {NAMES[k]}: {c}")

    # Train/Val/Test 分割
    N = len(train_labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    split_train = int(N * 0.8)
    split_val = int(N * 0.9)

    idx_train = perm[:split_train]
    idx_val = perm[split_train:split_val]
    idx_test = perm[split_val:]

    # Tensor 変換
    cnn = CnnPatchClassifier(seed=seed)

    def patches_to_tensors(indices):
        X = torch.stack([cnn._patch_to_tensor(train_patches[i])[0] for i in indices])
        y = torch.tensor([COLOR_TO_CLASS_INDEX[int(train_labels[i])] for i in indices], dtype=torch.long)
        return X, y

    X_train, y_train = patches_to_tensors(idx_train)
    X_val, y_val = patches_to_tensors(idx_val)
    X_test, y_test = patches_to_tensors(idx_test)

    # 損失関数
    if use_focal:
        counts = torch.bincount(y_train, minlength=NUM_CLASSES).float().clamp(min=1.0)
        alpha = (1.0 / counts)
        alpha = alpha / alpha.sum() * NUM_CLASSES
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
    else:
        counts = torch.bincount(y_train, minlength=NUM_CLASSES).float().clamp(min=1.0)
        weight = (1.0 / counts)
        weight = weight / weight.sum() * NUM_CLASSES
        criterion = nn.CrossEntropyLoss(weight=weight)

    model = cnn._model
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        perm_e = torch.randperm(X_train.size(0))
        total_loss = 0.0
        for start in range(0, X_train.size(0), batch_size):
            end = min(start + batch_size, X_train.size(0))
            idx = perm_e[start:end]
            optimizer.zero_grad()
            logits = model(X_train[idx])
            loss = criterion(logits, y_train[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (end - start)
        scheduler.step()

        # Val 評価 (5エポックごと)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_pred = torch.argmax(val_logits, dim=1)
                val_acc = (val_pred == y_val).float().mean().item()
            model.train()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            avg_loss = total_loss / X_train.size(0)
            cur_lr = scheduler.get_last_lr()[0]
            log(f"  epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} val={val_acc:.4f} lr={cur_lr:.5f}")

    elapsed = time.time() - t0
    log(f"学習完了: {elapsed:.1f}s")

    # ベストモデルを復元
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Test 評価
    with torch.no_grad():
        test_logits = model(X_test)
        test_pred = torch.argmax(test_logits, dim=1)
        test_acc = (test_pred == y_test).float().mean().item()

    log(f"val={best_val_acc:.4f} test={test_acc:.4f}")

    # クラス別精度
    report = {}
    y_true_np = y_test.numpy()
    y_pred_np = test_pred.numpy()
    for code in sorted(NAMES.keys()):
        ci = COLOR_TO_CLASS_INDEX.get(code)
        if ci is None:
            continue
        mask = y_true_np == ci
        if mask.sum() == 0:
            continue
        acc = float((y_pred_np[mask] == ci).mean())
        report[NAMES[code]] = {"n": int(mask.sum()), "acc": acc}
        log(f"  {NAMES[code]} (n={mask.sum()}): {acc:.4f}")

    # 混同行列 (主な誤り)
    log("主な誤分類:")
    for code in sorted(NAMES.keys()):
        ci = COLOR_TO_CLASS_INDEX.get(code)
        if ci is None:
            continue
        mask_t = y_true_np == ci
        if mask_t.sum() == 0:
            continue
        wrong = y_pred_np[mask_t] != ci
        if wrong.sum() == 0:
            continue
        wrong_preds = y_pred_np[mask_t][wrong]
        uniq, cnts = np.unique(wrong_preds, return_counts=True)
        for u, c in sorted(zip(uniq, cnts), key=lambda x: -x[1])[:2]:
            if c >= 3:
                pred_name = NAMES.get(CLASS_INDEX_TO_COLOR[int(u)], "?")
                log(f"  {NAMES[code]} → {pred_name}: {c}件")

    return cnn, {"test_acc": test_acc, "val_acc": best_val_acc, "report": report}


# ============================
# 指標評価 (サンプルフレーム)
# ============================

def evaluate_indicators(cnn: CnnPatchClassifier) -> None:
    config = CalibratedConfig.load("models/calibration_video01.json")
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(
        classifier=gated,
        p1_region=config.p1_region, p2_region=config.p2_region,
    )
    calc = IndicatorCalculator()
    scorer = Scorer()

    sample_dir = Path("data/frames/sample")
    if not sample_dir.exists():
        return

    log("=== 指標評価 ===")
    for fp in sorted(sample_dir.glob("frame_*.png"))[:4]:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        b1, b2 = reader.read_both_boards(frame)
        iset_1p = calc.compute_all(b1)
        iset_2p = calc.compute_all(b2)
        result = scorer.score(iset_1p, iset_2p)
        log(f"  {fp.name}: スコア={result.total_score:+.1f} ({result.advantage_side()})")


# ============================
# メインループ
# ============================

def main() -> None:
    log("=" * 60)
    log("自動反復改善開始 (目標: test精度99%)")
    log("=" * 60)

    patches, labels = load_all_data()

    best_acc = 0.0
    no_improve = 0
    current_cnn = None

    for round_num in range(1, MAX_ROUNDS + 1):
        log(f"\n{'='*40}")
        log(f"ラウンド {round_num}/{MAX_ROUNDS}")
        log(f"{'='*40}")

        # ハード例マイニング (2ラウンド目以降)
        hard_p, hard_l = None, None
        if current_cnn is not None:
            # テスト分割の代わりに全データからサンプリングしてハード例を探す
            rng = np.random.default_rng(round_num)
            sample_idx = rng.choice(len(labels), size=min(50000, len(labels)), replace=False)
            hard_p, hard_l = mine_hard_examples(
                current_cnn, patches[sample_idx], labels[sample_idx],
            )

        # 学習パラメータのスケジューリング
        base_lr = 0.003 if round_num <= 3 else 0.001
        base_epochs = 40 if round_num <= 5 else 50
        hard_os = min(3 + round_num, 8)  # ラウンドが進むほどハード例を増やす
        use_focal = round_num >= 2  # 1ラウンド目は通常Loss

        cnn, result = train_round(
            patches, labels,
            hard_patches=hard_p,
            hard_labels=hard_l,
            epochs=base_epochs,
            lr=base_lr,
            batch_size=256,
            hard_oversample=hard_os,
            use_focal=use_focal,
            seed=42 + round_num,
        )

        test_acc = result["test_acc"]

        if test_acc > best_acc:
            improvement = test_acc - best_acc
            best_acc = test_acc
            no_improve = 0
            model_path = MODEL_DIR / f"cnn_iter_r{round_num:02d}.pt"
            cnn.save(model_path)
            # ベストモデルも保存
            cnn.save(MODEL_DIR / "cnn_best.pt")
            log(f"改善! {test_acc:.4f} (+{improvement:.4f}) → {model_path}")
            current_cnn = cnn

            # 指標評価
            evaluate_indicators(cnn)
        else:
            no_improve += 1
            log(f"改善なし (best={best_acc:.4f}, 今回={test_acc:.4f}, patience={no_improve}/{PATIENCE})")
            # 改善なくても現在のモデルは使う (ハード例が変わる可能性)
            current_cnn = cnn

        # 終了判定
        if best_acc >= TARGET_ACC:
            log(f"\n目標達成! test精度={best_acc:.4f} >= {TARGET_ACC}")
            break
        if no_improve >= PATIENCE:
            log(f"\n{PATIENCE}ラウンド改善なし → 終了 (best={best_acc:.4f})")
            break

    log(f"\n最終結果: best test精度={best_acc:.4f}")
    log(f"最良モデル: models/cnn_best.pt")
    log("=" * 60)


if __name__ == "__main__":
    main()
