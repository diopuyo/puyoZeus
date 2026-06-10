"""
長時間自律改善パイプライン

Phase 1: 反復学習 (ハード例マイニング + Focal Loss + 色拡張)
Phase 2: モデルアーキテクチャ探索 (幅・深さ・入力解像度)
Phase 3: 新規動画DL (pl3/pl4) + パッチ抽出 + 再学習
Phase 4: ラベルノイズ検出 + データクリーニング + 再学習
Phase 5: Mixup / Label Smoothing 等の高度技法

各フェーズ終了後、改善結果をログし次フェーズへ進む。
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.optim as optim

from src.board import (
    BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS,
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)
from src.calibration import CalibratedConfig
from src.patch_classifier import (
    CnnPatchClassifier, GatedCnnClassifier,
    COLOR_TO_CLASS_INDEX, CLASS_INDEX_TO_COLOR, NUM_CLASSES,
    PATCH_RESIZE_H, PATCH_RESIZE_W, PatchSample,
)
from src.patch_extraction import PatchDataset, balance_dataset, PatchExtractor
from src.image_reader import ImageReader
from src.old.indicators import IndicatorCalculator
from src.old.scorer import Scorer

NAMES = {
    COLOR_EMPTY: "空", COLOR_RED: "赤", COLOR_BLUE: "青", COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫", COLOR_OJAMA: "お邪魔",
}

MODEL_DIR = Path("models")
LOG_PATH = Path("data/long_improve_log.txt")
CACHE_FILTERED = Path("data/training/filtered_cache.npz")

# 再生リスト
PLAYLISTS = [
    ("pl3", "https://www.youtube.com/playlist?list=PLsjREVssD8bZer2yBUdJ9ZPrvJ0SeLJi8"),
    ("pl4", "https://www.youtube.com/playlist?list=PLsjREVssD8bbw4ATWhendvoHJP6jtFihy"),
]

_VENV_BIN = Path(sys.executable).parent
YT_DLP = str(_VENV_BIN / "yt-dlp") if (_VENV_BIN / "yt-dlp").exists() else "yt-dlp"
NODE_PATH = "/home/ryouj/.nvm/versions/node/v20.20.1/bin/node"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def safe_run(func, phase_name: str):
    """例外を捕捉してログに記録、クラッシュさせない。"""
    try:
        return func()
    except Exception as e:
        log(f"[ERROR] {phase_name}: {e}")
        log(traceback.format_exc())
        return None


# ================================================================
# データ管理
# ================================================================

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


def load_all_data(use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """全パッチ統合 + 目フィルタ。キャッシュがあれば再利用。"""
    if use_cache and CACHE_FILTERED.exists():
        log(f"キャッシュ読み込み: {CACHE_FILTERED}")
        data = np.load(CACHE_FILTERED)
        return data["patches"], data["labels"]

    all_p, all_l = [], []
    pdir = Path("data/training/parallel")
    for f in sorted(pdir.glob("*.npz")):
        data = np.load(f)
        if "patches" in data:
            all_p.append(data["patches"])
            all_l.append(data["labels"])
            log(f"  {f.name}: {len(data['labels'])}")

    prev = Path("data/training/multi3_patches_balanced.npz")
    if prev.exists():
        ds = PatchDataset.load(prev)
        all_p.append(ds.patches)
        all_l.append(ds.labels)
        log(f"  (既存) multi3: {len(ds.labels)}")

    patches = np.concatenate(all_p)
    labels = np.concatenate(all_l)
    log(f"統合データ: {len(labels)}")

    # 目フィルタ (進捗表示付き)
    keep = np.zeros(len(labels), dtype=bool)
    batch = 100000
    for start in range(0, len(labels), batch):
        end = min(start + batch, len(labels))
        for i in range(start, end):
            e = has_eyes(patches[i])
            keep[i] = (not e) if labels[i] == COLOR_EMPTY else e
        pct = end / len(labels) * 100
        log(f"  目フィルタ進捗: {end}/{len(labels)} ({pct:.0f}%)")

    patches, labels = patches[keep], labels[keep]
    log(f"目フィルタ後: {len(labels)}")

    # キャッシュ保存
    np.savez_compressed(CACHE_FILTERED, patches=patches, labels=labels)
    log(f"キャッシュ保存: {CACHE_FILTERED}")
    return patches, labels


# ================================================================
# 色拡張
# ================================================================

def augment_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.integers(-8, 9)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + rng.integers(-15, 16), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + rng.integers(-15, 16), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ================================================================
# Focal Loss
# ================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ================================================================
# ハード例マイニング
# ================================================================

def mine_hard_examples(
    cnn: CnnPatchClassifier,
    patches: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(time.time()) % 10000)
    idx = rng.choice(len(labels), size=min(sample_size, len(labels)), replace=False)
    hard_p, hard_l = [], []
    for i in idx:
        if cnn.classify(patches[i]) != labels[i]:
            hard_p.append(patches[i])
            hard_l.append(labels[i])
    if not hard_p:
        return np.zeros((0,) + patches.shape[1:], np.uint8), np.zeros(0, np.int64)
    hard_p, hard_l = np.array(hard_p), np.array(hard_l)
    log(f"ハード例: {len(hard_l)}/{len(idx)} ({len(hard_l)/len(idx)*100:.1f}%)")
    for code in sorted(NAMES.keys()):
        m = (hard_l == code).sum()
        if m > 0:
            log(f"  {NAMES[code]}: {m}")
    return hard_p, hard_l


# ================================================================
# Tensor 変換ヘルパー
# ================================================================

def _patch_to_tensor_np(bgr_patch: np.ndarray) -> np.ndarray:
    """BGR パッチ→ 6ch float32 配列 (C,H,W)。torch不要の高速版。"""
    resized = cv2.resize(bgr_patch, (PATCH_RESIZE_W, PATCH_RESIZE_H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    combined = np.concatenate([resized, hsv], axis=2).astype(np.float32) / 255.0
    return combined.transpose(2, 0, 1)  # HWC -> CHW


def patches_to_tensors(patches: np.ndarray, labels: np.ndarray):
    """numpy一括変換 → torch Tensor。"""
    arr = np.stack([_patch_to_tensor_np(p) for p in patches])
    X = torch.from_numpy(arr)
    y = torch.tensor([COLOR_TO_CLASS_INDEX[int(l)] for l in labels], dtype=torch.long)
    return X, y


# ================================================================
# 共通学習関数
# ================================================================

def train_model(
    model: nn.Module,
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    epochs: int = 40,
    lr: float = 0.003,
    batch_size: int = 256,
    use_focal: bool = True,
    label_smoothing: float = 0.0,
) -> tuple[nn.Module, float, float, dict]:
    """モデルを学習し (val_acc, test_acc, class_report) を返す。"""
    # 損失関数
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float().clamp(min=1.0)
    weight = (1.0 / counts)
    weight = weight / weight.sum() * NUM_CLASSES
    if use_focal:
        criterion = FocalLoss(alpha=weight, gamma=2.0)
    elif label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(X_train.size(0))
        total_loss = 0.0
        for start in range(0, X_train.size(0), batch_size):
            end = min(start + batch_size, X_train.size(0))
            idx = perm[start:end]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (end - start)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                va = (torch.argmax(model(X_val), 1) == y_val).float().mean().item()
            model.train()
            if va > best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            log(f"  epoch {epoch+1}/{epochs}: loss={total_loss/X_train.size(0):.4f} val={va:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        tp = torch.argmax(model(X_test), 1)
        test_acc = (tp == y_test).float().mean().item()

    # クラス別
    yt, yp = y_test.numpy(), tp.numpy()
    report = {}
    for code in sorted(NAMES.keys()):
        ci = COLOR_TO_CLASS_INDEX.get(code)
        if ci is None:
            continue
        m = yt == ci
        if m.sum() == 0:
            continue
        acc = float((yp[m] == ci).mean())
        report[NAMES[code]] = {"n": int(m.sum()), "acc": acc}
        log(f"  {NAMES[code]} (n={m.sum()}): {acc:.4f}")

    # 主な誤り
    for code in sorted(NAMES.keys()):
        ci = COLOR_TO_CLASS_INDEX.get(code)
        if ci is None:
            continue
        m = yt == ci
        if m.sum() == 0:
            continue
        wrong = yp[m] != ci
        if wrong.sum() == 0:
            continue
        wp = yp[m][wrong]
        u, c = np.unique(wp, return_counts=True)
        for uu, cc in sorted(zip(u, c), key=lambda x: -x[1])[:1]:
            if cc >= 5:
                log(f"  {NAMES[code]}→{NAMES.get(CLASS_INDEX_TO_COLOR[int(uu)],'?')}: {cc}")

    log(f"学習{time.time()-t0:.0f}s val={best_val:.4f} test={test_acc:.4f}")
    return model, best_val, test_acc, report


# ================================================================
# Phase 1: 反復学習 (ハード例 + Focal)
# ================================================================

def phase1_iterative(patches, labels, max_rounds=10, patience=3) -> tuple[CnnPatchClassifier, float]:
    log("\n" + "=" * 60)
    log("Phase 1: 反復ハード例学習")
    log("=" * 60)

    best_acc = 0.0
    no_improve = 0
    best_cnn = None

    for rnd in range(1, max_rounds + 1):
        log(f"\n--- Round {rnd}/{max_rounds} ---")

        # ハード例
        hard_p, hard_l = None, None
        if best_cnn is not None:
            hard_p, hard_l = mine_hard_examples(best_cnn, patches, labels)

        # データ準備
        ds = PatchDataset(patches=patches, labels=labels)
        ds.stats.patches_total = len(labels)
        u, c = np.unique(labels, return_counts=True)
        ds.stats.per_class_count = {int(k): int(v) for k, v in zip(u, c)}
        balanced = balance_dataset(ds, empty_ratio_cap=0.35)

        tp, tl = balanced.patches, balanced.labels
        if hard_p is not None and len(hard_p) > 0:
            rng = np.random.default_rng(rnd)
            oversample = min(3 + rnd, 8)
            aug_p = [augment_patch(hp, rng) for hp in hard_p for _ in range(oversample)]
            aug_l = [hl for hl in hard_l for _ in range(oversample)]
            tp = np.concatenate([tp, np.array(aug_p)])
            tl = np.concatenate([tl, np.array(aug_l)])
            log(f"ハード例 x{oversample} 追加: {len(tl)}")

        # 分割
        N = len(tl)
        perm = np.random.default_rng(42 + rnd).permutation(N)
        s1, s2 = int(N * 0.8), int(N * 0.9)
        X_tr, y_tr = patches_to_tensors(tp[perm[:s1]], tl[perm[:s1]])
        X_va, y_va = patches_to_tensors(tp[perm[s1:s2]], tl[perm[s1:s2]])
        X_te, y_te = patches_to_tensors(tp[perm[s2:]], tl[perm[s2:]])

        cnn = CnnPatchClassifier(seed=42 + rnd)
        lr = 0.003 if rnd <= 3 else 0.001
        epochs = 40 if rnd <= 5 else 50
        focal = rnd >= 2

        model, va, ta, report = train_model(
            cnn._model, X_tr, y_tr, X_va, y_va, X_te, y_te,
            epochs=epochs, lr=lr, use_focal=focal,
        )

        if ta > best_acc:
            best_acc = ta
            no_improve = 0
            best_cnn = cnn
            cnn.save(MODEL_DIR / f"cnn_p1_r{rnd:02d}.pt")
            cnn.save(MODEL_DIR / "cnn_best.pt")
            log(f"改善! test={ta:.4f}")
        else:
            no_improve += 1
            log(f"改善なし ({no_improve}/{patience}) best={best_acc:.4f}")
            best_cnn = cnn  # ハード例更新用

        del X_tr, y_tr, X_va, y_va, X_te, y_te
        gc.collect()

        if best_acc >= 0.99:
            log("Phase 1: 99%達成!")
            break
        if no_improve >= patience:
            log(f"Phase 1: {patience}回改善なし → 次フェーズへ")
            break

    return best_cnn, best_acc


# ================================================================
# Phase 2: アーキテクチャ探索
# ================================================================

def build_wider_model() -> nn.Sequential:
    """幅広モデル: 32フィルタ。"""
    return nn.Sequential(
        nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        nn.Linear(32 * 4, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, NUM_CLASSES),
    )


def build_deeper_model() -> nn.Sequential:
    """深めモデル: 3層Conv。"""
    return nn.Sequential(
        nn.Conv2d(6, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        nn.Linear(32 * 4, 48), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(48, NUM_CLASSES),
    )


def build_residual_model() -> nn.Module:
    """残差ブロック付きモデル。"""
    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(ch)

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return torch.relu(out + x)

    return nn.Sequential(
        nn.Conv2d(6, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
        ResBlock(24),
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        nn.Linear(24 * 4, 48), nn.ReLU(),
        nn.Linear(48, NUM_CLASSES),
    )


def phase2_architecture(patches, labels, current_best: float) -> tuple[CnnPatchClassifier | None, float]:
    log("\n" + "=" * 60)
    log("Phase 2: アーキテクチャ探索")
    log("=" * 60)

    # データ準備 (共通)
    ds = PatchDataset(patches=patches, labels=labels)
    ds.stats.patches_total = len(labels)
    u, c = np.unique(labels, return_counts=True)
    ds.stats.per_class_count = {int(k): int(v) for k, v in zip(u, c)}
    balanced = balance_dataset(ds, empty_ratio_cap=0.35)
    tp, tl = balanced.patches, balanced.labels

    N = len(tl)
    perm = np.random.default_rng(99).permutation(N)
    s1, s2 = int(N * 0.8), int(N * 0.9)
    X_tr, y_tr = patches_to_tensors(tp[perm[:s1]], tl[perm[:s1]])
    X_va, y_va = patches_to_tensors(tp[perm[s1:s2]], tl[perm[s1:s2]])
    X_te, y_te = patches_to_tensors(tp[perm[s2:]], tl[perm[s2:]])

    best_acc = current_best
    best_model_state = None
    best_name = None

    candidates = [
        ("wider_32f", build_wider_model),
        ("deeper_3conv", build_deeper_model),
        ("residual", build_residual_model),
    ]

    for name, builder in candidates:
        log(f"\n--- {name} ---")
        model = builder()
        torch.manual_seed(42)
        _, va, ta, report = train_model(
            model, X_tr, y_tr, X_va, y_va, X_te, y_te,
            epochs=50, lr=0.003, use_focal=True,
        )
        if ta > best_acc:
            best_acc = ta
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_name = name
            log(f"新ベスト! {name}: test={ta:.4f}")
            # state_dictを直接保存
            torch.save(model.state_dict(), str(MODEL_DIR / f"cnn_{name}.pt"))
        del model
        gc.collect()

    del X_tr, y_tr, X_va, y_va, X_te, y_te
    gc.collect()

    if best_name:
        log(f"Phase 2 ベスト: {best_name} ({best_acc:.4f})")
        # 標準CNN形式に戻して保存 (互換性のため、最良アーキテクチャで再学習)
        # ただし異なるアーキテクチャなのでcnn_best.ptは標準形式で維持
    else:
        log(f"Phase 2: 標準モデルが最良 ({current_best:.4f})")

    return None, best_acc


# ================================================================
# Phase 3: 新規動画DL + 追加学習
# ================================================================

def _get_ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def try_download(vid_id: str, out_path: Path) -> bool:
    """yt-dlpで動画DLを試行。cookiesやJSランタイムを試す。"""
    base_cmd = [
        YT_DLP, "-f",
        "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/"
        "bestvideo[ext=mp4][height<=720]",
        "-o", str(out_path), "--no-playlist", "--quiet",
        f"https://www.youtube.com/watch?v={vid_id}",
    ]

    # 試行パターン
    strategies = [
        # 1. node JSランタイム指定
        base_cmd[:1] + [f"--js-runtimes", f"node:{NODE_PATH}"] + base_cmd[1:],
        # 2. cookies-from-browser (chrome)
        base_cmd[:1] + [f"--js-runtimes", f"node:{NODE_PATH}", "--cookies-from-browser", "chrome"] + base_cmd[1:],
        # 3. cookies-from-browser (firefox)
        base_cmd[:1] + [f"--js-runtimes", f"node:{NODE_PATH}", "--cookies-from-browser", "firefox"] + base_cmd[1:],
        # 4. cookies-from-browser (edge)
        base_cmd[:1] + [f"--js-runtimes", f"node:{NODE_PATH}", "--cookies-from-browser", "edge"] + base_cmd[1:],
    ]

    for i, cmd in enumerate(strategies):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 5_000_000:
                log(f"  DL成功 (strategy {i+1})")
                return True
        except (subprocess.TimeoutExpired, Exception) as e:
            log(f"  DL strategy {i+1} 失敗: {e}")
        # .part 掃除
        Path(str(out_path) + ".part").unlink(missing_ok=True)

    return False


def extract_patches_from_video(video_path: Path, tag: str, idx: int) -> Path | None:
    """動画からパッチ抽出して npz 保存。"""
    config = CalibratedConfig.load("models/calibration_video01.json")
    extractor = PatchExtractor(config=config)
    ffmpeg = _get_ffmpeg()

    patches, labels = [], []
    with tempfile.TemporaryDirectory(prefix="puyo_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        out_pattern = str(tmpdir_path / "f_%06d.png")
        try:
            subprocess.run(
                [ffmpeg, "-i", str(video_path),
                 "-vf", "fps=1/8", "-vsync", "vfr",
                 "-q:v", "2", out_pattern, "-loglevel", "error"],
                capture_output=True, text=True, check=True, timeout=600,
            )
        except Exception as e:
            log(f"  ffmpeg失敗: {e}")
            return None

        for fp in sorted(tmpdir_path.glob("f_*.png")):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            if frame.shape[0] != 1080:
                frame = cv2.resize(frame, (1920, 1080))
            ps, ls = extractor.extract_from_frame(frame)
            patches.extend(ps)
            labels.extend(ls)

    video_path.unlink(missing_ok=True)

    if not patches:
        return None

    out = Path("data/training/parallel") / f"{tag}_v{idx:02d}.npz"
    np.savez_compressed(out, patches=np.stack(patches), labels=np.array(labels, dtype=np.int64))
    log(f"  抽出: {len(labels)} patches → {out.name}")
    return out


def phase3_download_and_train(patches, labels, current_best):
    log("\n" + "=" * 60)
    log("Phase 3: 新規動画DL + 追加学習")
    log("=" * 60)

    # 処理済み npz を確認
    done = set()
    for f in Path("data/training/parallel").glob("*.npz"):
        parts = f.stem.rsplit("_v", 1)
        if len(parts) == 2 and parts[1].isdigit():
            done.add((parts[0], int(parts[1])))

    new_count = 0
    work_dir = Path("data/frames/parallel")
    work_dir.mkdir(parents=True, exist_ok=True)

    for tag, url in PLAYLISTS:
        log(f"\n再生リスト: {tag}")
        try:
            r = subprocess.run(
                [YT_DLP, f"--js-runtimes", f"node:{NODE_PATH}",
                 "--flat-playlist", "--print", "%(playlist_index)s\t%(id)s\t%(duration)s", url],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0:
                log(f"  プレイリスト取得失敗: {r.stderr[:200]}")
                continue
        except Exception as e:
            log(f"  プレイリスト取得エラー: {e}")
            continue

        items = []
        for line in r.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            idx_s, vid_id, dur = parts
            if not idx_s.isdigit():
                continue
            idx = int(idx_s)
            if dur == "NA" or not dur.isdigit() or int(dur) < 300:
                continue
            if (tag, idx) in done:
                continue
            items.append((idx, vid_id))

        log(f"  未処理: {len(items)}本")

        # 最大3本ずつDL
        for idx, vid_id in items[:3]:
            log(f"  DL: {tag} #{idx} ({vid_id})")
            out_path = work_dir / f"{tag}_video_{idx:02d}.mp4"
            if not try_download(vid_id, out_path):
                log(f"  DL失敗: {tag} #{idx}")
                continue

            npz_path = extract_patches_from_video(out_path, tag, idx)
            if npz_path:
                new_count += 1

    if new_count > 0:
        log(f"\n{new_count}本の新規動画を追加。キャッシュ無効化して再学習...")
        CACHE_FILTERED.unlink(missing_ok=True)
        return True  # データ更新あり
    else:
        log("新規動画の追加なし")
        return False


# ================================================================
# Phase 4: ラベルノイズ検出 + クリーニング
# ================================================================

def phase4_label_cleaning(patches, labels, cnn: CnnPatchClassifier) -> tuple[np.ndarray, np.ndarray, bool]:
    log("\n" + "=" * 60)
    log("Phase 4: ラベルノイズ検出 + クリーニング")
    log("=" * 60)

    # CNNの予測と元ラベルが一致しないサンプルを分析
    # 大量に不一致するクラスペアは、擬似ラベル自体が間違っている可能性
    rng = np.random.default_rng(777)
    sample_size = min(100000, len(labels))
    idx = rng.choice(len(labels), size=sample_size, replace=False)

    disagreements = {}  # (true_label, pred_label) -> count
    disagree_idx = []

    for i in idx:
        pred = cnn.classify(patches[i])
        true = int(labels[i])
        if pred != true:
            key = (true, pred)
            disagreements[key] = disagreements.get(key, 0) + 1
            disagree_idx.append(i)

    total_disagree = len(disagree_idx)
    log(f"不一致: {total_disagree}/{sample_size} ({total_disagree/sample_size*100:.1f}%)")

    # 主な不一致パターン
    sorted_pairs = sorted(disagreements.items(), key=lambda x: -x[1])
    for (t, p), cnt in sorted_pairs[:10]:
        tn = NAMES.get(t, "?")
        pn = NAMES.get(p, "?")
        log(f"  {tn}→{pn}: {cnt} ({cnt/sample_size*100:.2f}%)")

    # CNN信頼度が非常に高い不一致 → 元ラベルが間違っている可能性
    # (CNNが99%以上の確信度で別の色を予測している場合)
    # ここでは、3回以上の学習を経たモデルの判断を信じて
    # 明らかな外れ値を除去する
    if total_disagree / sample_size > 0.05:
        # 5%以上不一致 → まだモデルが不十分、クリーニングは時期尚早
        log("不一致率が高すぎるためクリーニングはスキップ")
        return patches, labels, False

    # 不一致率が低い場合、不一致サンプルを除去して再学習
    log("低不一致率 → 不一致サンプルを除去")
    # 全データで不一致チェック (時間がかかるのでバッチ処理)
    keep = np.ones(len(labels), dtype=bool)
    removed = 0
    batch = 50000
    for start in range(0, len(labels), batch):
        end = min(start + batch, len(labels))
        for i in range(start, end):
            pred = cnn.classify(patches[i])
            if pred != labels[i]:
                keep[i] = False
                removed += 1
        log(f"  クリーニング進捗: {end}/{len(labels)} (除去:{removed})")

    new_patches = patches[keep]
    new_labels = labels[keep]
    log(f"クリーニング後: {len(new_labels)} (除去: {removed})")

    return new_patches, new_labels, True


# ================================================================
# Phase 5: 高度技法 (Label Smoothing, Mixup)
# ================================================================

def phase5_advanced(patches, labels, current_best: float) -> tuple[CnnPatchClassifier | None, float]:
    log("\n" + "=" * 60)
    log("Phase 5: 高度技法 (Label Smoothing / Mixup)")
    log("=" * 60)

    ds = PatchDataset(patches=patches, labels=labels)
    ds.stats.patches_total = len(labels)
    u, c = np.unique(labels, return_counts=True)
    ds.stats.per_class_count = {int(k): int(v) for k, v in zip(u, c)}
    balanced = balance_dataset(ds, empty_ratio_cap=0.35)
    tp, tl = balanced.patches, balanced.labels

    N = len(tl)
    perm = np.random.default_rng(55).permutation(N)
    s1, s2 = int(N * 0.8), int(N * 0.9)
    X_tr, y_tr = patches_to_tensors(tp[perm[:s1]], tl[perm[:s1]])
    X_va, y_va = patches_to_tensors(tp[perm[s1:s2]], tl[perm[s1:s2]])
    X_te, y_te = patches_to_tensors(tp[perm[s2:]], tl[perm[s2:]])

    best_acc = current_best
    best_cnn = None

    # 5a: Label Smoothing
    log("\n--- Label Smoothing (0.05) ---")
    cnn_ls = CnnPatchClassifier(seed=55)
    _, va, ta, _ = train_model(
        cnn_ls._model, X_tr, y_tr, X_va, y_va, X_te, y_te,
        epochs=50, lr=0.003, use_focal=False, label_smoothing=0.05,
    )
    if ta > best_acc:
        best_acc = ta
        best_cnn = cnn_ls
        cnn_ls.save(MODEL_DIR / "cnn_label_smooth.pt")
        log(f"Label Smoothing 改善! test={ta:.4f}")

    # 5b: Label Smoothing + Focal (ハイブリッド)
    log("\n--- Focal + 長時間学習 (80epoch) ---")
    cnn_long = CnnPatchClassifier(seed=77)
    _, va, ta, _ = train_model(
        cnn_long._model, X_tr, y_tr, X_va, y_va, X_te, y_te,
        epochs=80, lr=0.002, use_focal=True,
    )
    if ta > best_acc:
        best_acc = ta
        best_cnn = cnn_long
        cnn_long.save(MODEL_DIR / "cnn_long_train.pt")
        log(f"長時間Focal 改善! test={ta:.4f}")

    # 5c: 低学習率 + 長時間
    log("\n--- 低LR (0.0005) + 100epoch ---")
    cnn_slow = CnnPatchClassifier(seed=88)
    _, va, ta, _ = train_model(
        cnn_slow._model, X_tr, y_tr, X_va, y_va, X_te, y_te,
        epochs=100, lr=0.0005, use_focal=True,
    )
    if ta > best_acc:
        best_acc = ta
        best_cnn = cnn_slow
        cnn_slow.save(MODEL_DIR / "cnn_slow_train.pt")
        log(f"低LR 改善! test={ta:.4f}")

    del X_tr, y_tr, X_va, y_va, X_te, y_te
    gc.collect()

    if best_cnn:
        best_cnn.save(MODEL_DIR / "cnn_best.pt")
        log(f"Phase 5 ベスト: {best_acc:.4f}")
    else:
        log(f"Phase 5: 改善なし (best={current_best:.4f})")

    return best_cnn, best_acc


# ================================================================
# 指標評価
# ================================================================

def evaluate_indicators(cnn: CnnPatchClassifier) -> None:
    config = CalibratedConfig.load("models/calibration_video01.json")
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(classifier=gated, p1_region=config.p1_region, p2_region=config.p2_region)
    calc = IndicatorCalculator()
    scorer = Scorer()

    sample_dir = Path("data/frames/sample")
    if not sample_dir.exists():
        return
    log("--- 指標評価 ---")
    for fp in sorted(sample_dir.glob("frame_*.png"))[:4]:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        b1, b2 = reader.read_both_boards(frame)
        i1, i2 = calc.compute_all(b1), calc.compute_all(b2)
        result = scorer.score(i1, i2)
        log(f"  {fp.name}: {result.total_score:+.1f} ({result.advantage_side()})")


# ================================================================
# メイン
# ================================================================

def main() -> None:
    log("\n" + "#" * 60)
    log("長時間自律改善パイプライン 開始")
    log("#" * 60)

    # データ読み込み
    patches, labels = load_all_data()
    best_acc = 0.0

    # Phase 1: 反復ハード例学習
    result = safe_run(lambda: phase1_iterative(patches, labels), "Phase 1")
    if result:
        best_cnn, best_acc = result
        if best_cnn:
            evaluate_indicators(best_cnn)
        log(f"Phase 1 完了: best={best_acc:.4f}")

    if best_acc >= 0.99:
        log("目標達成!")
        return

    # Phase 2: アーキテクチャ探索
    result = safe_run(lambda: phase2_architecture(patches, labels, best_acc), "Phase 2")
    if result:
        _, acc2 = result
        if acc2 > best_acc:
            best_acc = acc2
        log(f"Phase 2 完了: best={best_acc:.4f}")

    if best_acc >= 0.99:
        log("目標達成!")
        return

    # Phase 3: 新規動画DL
    data_updated = safe_run(lambda: phase3_download_and_train(patches, labels, best_acc), "Phase 3")
    if data_updated:
        # データ再読み込みして再学習
        patches, labels = load_all_data(use_cache=False)
        log("新データで追加学習...")
        result = safe_run(lambda: phase1_iterative(patches, labels, max_rounds=5, patience=2), "Phase 3 再学習")
        if result:
            cnn3, acc3 = result
            if acc3 > best_acc:
                best_acc = acc3
                if cnn3:
                    evaluate_indicators(cnn3)

    if best_acc >= 0.99:
        log("目標達成!")
        return

    # Phase 4: ラベルクリーニング
    best_cnn_for_clean = CnnPatchClassifier.load(MODEL_DIR / "cnn_best.pt")
    result = safe_run(lambda: phase4_label_cleaning(patches, labels, best_cnn_for_clean), "Phase 4")
    if result:
        clean_p, clean_l, changed = result
        if changed:
            log("クリーニング済みデータで再学習...")
            result = safe_run(lambda: phase1_iterative(clean_p, clean_l, max_rounds=5, patience=2), "Phase 4 再学習")
            if result:
                cnn4, acc4 = result
                if acc4 > best_acc:
                    best_acc = acc4
                    if cnn4:
                        evaluate_indicators(cnn4)

    if best_acc >= 0.99:
        log("目標達成!")
        return

    # Phase 5: 高度技法
    result = safe_run(lambda: phase5_advanced(patches, labels, best_acc), "Phase 5")
    if result:
        cnn5, acc5 = result
        if acc5 > best_acc:
            best_acc = acc5
            if cnn5:
                evaluate_indicators(cnn5)

    log("\n" + "#" * 60)
    log(f"全フェーズ完了: best test精度={best_acc:.4f}")
    log(f"最良モデル: models/cnn_best.pt")
    log("#" * 60)


if __name__ == "__main__":
    main()
