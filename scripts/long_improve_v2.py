"""
長時間自律改善パイプライン v2 (メモリ効率版)

v1との違い:
- ファイル単位で目フィルタを適用してメモリピークを抑制
- MAX_SAMPLES でサブサンプルし tensor 化のメモリ爆発を回避
- filtered_cache.npz はサブサンプル後の小サイズで保存

Phase 1: 反復ハード例学習 (Focal Loss + 色拡張)
Phase 2: モデルアーキテクチャ探索
Phase 3: 新規動画DL + 追加学習
Phase 4: ラベルノイズ検出 + クリーニング
Phase 5: 高度技法 (Label Smoothing / 長時間学習)
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

# プロジェクトルートを sys.path に追加 (scripts/ から呼ばれるため src/ が見えない問題を回避)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
from src.cutmix import generate_cutmix_arrays
from src.image_reader import ImageReader
from src.indicators import IndicatorCalculator
from src.scorer import Scorer

NAMES = {
    COLOR_EMPTY: "空", COLOR_RED: "赤", COLOR_BLUE: "青", COLOR_GREEN: "緑",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫", COLOR_OJAMA: "お邪魔",
}

MODEL_DIR = Path("models")
LOG_PATH = Path("data/long_improve_log.txt")
MILESTONE_PATH = Path("data/milestones.jsonl")
CACHE_FILTERED = Path("data/training/filtered_cache_v2.npz")

# holdout 基準のグローバルベスト永続保持
# phase1_iterative は各サイクルで cnn_best.pt を無条件上書きするので、
# holdout が改善した瞬間に cnn_global_best.pt へコピーして保護する。
GLOBAL_BEST_MODEL = MODEL_DIR / "cnn_global_best.pt"
GLOBAL_BEST_STATE = Path("data/global_best.json")
# holdout がこの値以上改善した場合のみ global best を昇格させる
GLOBAL_BEST_EPS = 0.003

# WSL2 の割当メモリは 16GB のため Linux 空きを基準に設定
# 180K patches × 44×44×6×4byte ≈ 8.3GB tensor
MAX_SAMPLES_PER_TRAIN = 180_000

# 99% 到達で E2E 検証モードに遷移 (訓練は継続)
E2E_THRESHOLD = 0.99

# 時間ベース停止 (40時間)
MAX_HOURS = 40.0
MAX_CYCLES = 200  # 実効的には時間キャップが先に効く

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
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def milestone(kind: str, summary: str, **details) -> None:
    """大きな進捗・革命的事象を Claude Code 側で検出できる JSON Lines に書き出す。

    kind: cycle_start / cycle_complete / new_best / threshold_99 / e2e_result / early_exit / fatal /
          phase_complete / anomaly / discovery
    summary: 80字以内の簡潔な一行要約
    details: 任意の追加フィールド (acc, holdout, cycle_n, phase 等)
    """
    import json as _json
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "summary": summary, **details}
    try:
        with open(MILESTONE_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    log(f"[MILESTONE:{kind}] {summary}")


def safe_run(func, phase_name: str):
    try:
        return func()
    except Exception as e:
        log(f"[ERROR] {phase_name}: {e}")
        log(traceback.format_exc())
        return None


# ================================================================
# データ管理 (ファイル単位フィルタ版)
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


def _filter_one_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """1つのnpzを読み込み目フィルタして返す。メモリピーク抑制。"""
    data = np.load(npz_path)
    if "patches" not in data:
        return np.zeros((0, 1, 1, 3), np.uint8), np.zeros(0, np.int64)
    patches = data["patches"]
    labels = data["labels"]
    keep = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        e = has_eyes(patches[i])
        keep[i] = (not e) if labels[i] == COLOR_EMPTY else e
    fp = patches[keep].copy()
    fl = labels[keep].copy()
    del patches, labels, keep, data
    return fp, fl


def load_all_data(
    use_cache: bool = True,
    max_samples: int = MAX_SAMPLES_PER_TRAIN,
) -> tuple[np.ndarray, np.ndarray]:
    """ファイル単位で目フィルタしつつ、ファイル単位でサブサンプリングしてメモリピークを抑える。

    322万パッチ全体を一度concatenateすると ~18GB で WSL2(16GB) OOM クラッシュする。
    対策: 各ファイルごとに max_samples 比例で先にサブサンプリング→積み上げる。
    """
    if use_cache and CACHE_FILTERED.exists():
        log(f"キャッシュ読み込み: {CACHE_FILTERED}")
        data = np.load(CACHE_FILTERED)
        return data["patches"], data["labels"]

    # --- 第1パス: 各ファイルのフィルタ後件数だけ取得（concat せずにメタ情報） ---
    pdir = Path("data/training/parallel")
    npz_files = sorted(pdir.glob("*.npz"))
    log(f"処理対象: {len(npz_files)}ファイル (第1パス: カウント)")

    per_file_counts: list[int] = []
    per_file_keep: list[np.ndarray] = []  # keep mask を一時保持
    # 注意: mask だけなら 3M bit = 400KB 程度で軽い
    for idx, f in enumerate(npz_files):
        data = np.load(f)
        if "patches" not in data:
            per_file_counts.append(0)
            per_file_keep.append(np.zeros(0, dtype=bool))
            continue
        patches_f = data["patches"]
        labels_f = data["labels"]
        keep = np.zeros(len(labels_f), dtype=bool)
        for i in range(len(labels_f)):
            e = has_eyes(patches_f[i])
            keep[i] = (not e) if labels_f[i] == COLOR_EMPTY else e
        per_file_counts.append(int(keep.sum()))
        per_file_keep.append(keep)
        del patches_f, labels_f, data
        if (idx + 1) % 10 == 0:
            log(f"  カウント進捗: {idx+1}/{len(npz_files)} 累積残: {sum(per_file_counts)}")
        gc.collect()

    # multi3 も同様に事前カウント
    prev = Path("data/training/multi3_patches_balanced.npz")
    multi3_keep = None
    multi3_count = 0
    if prev.exists():
        ds = PatchDataset.load(prev)
        keep_m = np.zeros(len(ds.labels), dtype=bool)
        for i in range(len(ds.labels)):
            e = has_eyes(ds.patches[i])
            keep_m[i] = (not e) if ds.labels[i] == COLOR_EMPTY else e
        multi3_count = int(keep_m.sum())
        multi3_keep = keep_m
        log(f"  multi3 フィルタ後: {multi3_count}")
        del ds
        gc.collect()

    total_kept = sum(per_file_counts) + multi3_count
    log(f"目フィルタ後 合計: {total_kept}")

    # --- 第2パス: ファイル単位で比例サブサンプリング ---
    if total_kept > max_samples:
        ratio = max_samples / total_kept
        log(f"ファイル単位サブサンプル: ratio={ratio:.4f} ({total_kept}→{max_samples})")
    else:
        ratio = 1.0

    rng = np.random.default_rng(42)
    kept_p: list[np.ndarray] = []
    kept_l: list[np.ndarray] = []

    for idx, (f, keep) in enumerate(zip(npz_files, per_file_keep)):
        if keep.sum() == 0:
            continue
        data = np.load(f)
        patches_f = data["patches"]
        labels_f = data["labels"]
        # keep なインデックスのみ
        keep_idx = np.where(keep)[0]
        # 比例サブサンプリング
        target = max(1, int(len(keep_idx) * ratio)) if ratio < 1.0 else len(keep_idx)
        if target < len(keep_idx):
            chosen = rng.choice(keep_idx, size=target, replace=False)
        else:
            chosen = keep_idx
        kept_p.append(patches_f[chosen].copy())
        kept_l.append(labels_f[chosen].copy())
        del patches_f, labels_f, data
        if (idx + 1) % 10 == 0:
            log(f"  サブサンプル進捗: {idx+1}/{len(npz_files)} 累積: {sum(len(l) for l in kept_l)}")
        gc.collect()

    if multi3_keep is not None and multi3_count > 0:
        ds = PatchDataset.load(prev)
        keep_idx_m = np.where(multi3_keep)[0]
        target_m = max(1, int(len(keep_idx_m) * ratio)) if ratio < 1.0 else len(keep_idx_m)
        if target_m < len(keep_idx_m):
            chosen_m = rng.choice(keep_idx_m, size=target_m, replace=False)
        else:
            chosen_m = keep_idx_m
        kept_p.append(ds.patches[chosen_m].copy())
        kept_l.append(ds.labels[chosen_m].copy())
        del ds
        gc.collect()

    # --- 連結 ---
    patches = np.concatenate(kept_p)
    labels = np.concatenate(kept_l)
    del kept_p, kept_l, per_file_keep
    gc.collect()

    # 最終シャッフル
    perm = rng.permutation(len(labels))
    patches = patches[perm].copy()
    labels = labels[perm].copy()
    gc.collect()

    # クラス別内訳
    u, c = np.unique(labels, return_counts=True)
    for ul, cc in zip(u, c):
        log(f"  {NAMES.get(int(ul), '?')}: {cc}")

    # キャッシュ保存 (サブサンプル後なら小さい)
    try:
        np.savez_compressed(CACHE_FILTERED, patches=patches, labels=labels)
        log(f"キャッシュ保存: {CACHE_FILTERED}")
    except Exception as e:
        log(f"キャッシュ保存失敗: {e}")

    return patches, labels


# ================================================================
# 色拡張
# ================================================================

def augment_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """標準色拡張: HSV を小幅揺らす（全色共通）"""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.integers(-8, 9)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + rng.integers(-15, 16), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + rng.integers(-15, 16), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_dark_variant(
    patch: np.ndarray,
    rng: np.random.Generator,
    min_saturation: int = 35,
) -> np.ndarray:
    """暗所特化拡張: 輝度を 0.65〜0.85 倍に落として暗背景下の見え方を模倣。
    紫×空 誤認対策。確率 0.35 で α=0.15-0.30 の暗色オーバーレイを加える。

    生成後に「彩度が落ちすぎて空に見える」と判定したら元パッチを返す（ラベルノイズ化防止）。
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_gain = float(rng.uniform(0.65, 0.85))  # 旧 0.50-0.80 は暗すぎて空とクラス境界が曖昧化
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_gain, 0, 255)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(rng.uniform(0.90, 1.0)), 0, 255)  # S を手厚く保護
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.35:  # 旧 0.5 → 0.35
        alpha = float(rng.uniform(0.15, 0.30))  # 上限 0.40→0.30 で薄め
        dark = np.full_like(out, rng.integers(30, 55), dtype=np.uint8)  # 20-60 → 30-55
        out = cv2.addWeighted(out, 1.0 - alpha, dark, alpha, 0.0)
    # ラベルノイズ化防止ガード: 彩度が閾値以下なら紫の色味が失われたと見なし元を返す
    out_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    if float(out_hsv[:, :, 1].mean()) < min_saturation:
        return patch
    return out


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
    sample_size: int = 30000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """ハード例をマイニング。

    Returns:
        (hard_p, hard_l, is_purple_to_empty, purple_to_empty_count) — 4 要素タプル。
        is_purple_to_empty は紫×空誤認の bool mask、purple_to_empty_count はその総数。
        紫×空誤認は別途「暗所紫拡張」の対象になるため、呼び出し側で分岐できるよう返す。
        milestone 発火は呼び出し側 (phase1_iterative) で Round 間比較して制御する。
    """
    rng = np.random.default_rng(int(time.time()) % 10000)
    idx = rng.choice(len(labels), size=min(sample_size, len(labels)), replace=False)
    hard_p, hard_l, is_p2e = [], [], []
    purple_to_empty = 0  # 紫ラベルで空と誤認 → 暗所紫拡張の対象
    confusion = {}  # (true, pred) → count
    for i in idx:
        pred = cnn.classify(patches[i])
        true = int(labels[i])
        if pred != true:
            hard_p.append(patches[i])
            hard_l.append(true)
            key = (true, pred)
            confusion[key] = confusion.get(key, 0) + 1
            p2e = (true == COLOR_PURPLE and pred == COLOR_EMPTY)
            is_p2e.append(p2e)
            if p2e:
                purple_to_empty += 1
    if not hard_p:
        return (
            np.zeros((0,) + patches.shape[1:], np.uint8),
            np.zeros(0, np.int64),
            np.zeros(0, dtype=bool),
            0,
        )
    hard_p = np.array(hard_p)
    hard_l = np.array(hard_l)
    is_p2e_arr = np.array(is_p2e, dtype=bool)
    log(f"ハード例: {len(hard_l)}/{len(idx)} ({len(hard_l)/len(idx)*100:.1f}%)")
    for code in sorted(NAMES.keys()):
        m = (hard_l == code).sum()
        if m > 0:
            log(f"  {NAMES[code]}: {m}")
    # 紫×空 誤認は最大の hold-out ボトルネックなので個別にログ
    log(f"  [注目] 紫→空 誤認: {purple_to_empty}件 (暗所紫拡張の対象)")
    # 誤認上位 3 ペアを記録
    for (t, p), cnt in sorted(confusion.items(), key=lambda x: -x[1])[:3]:
        log(f"  誤認 {NAMES.get(t,'?')}→{NAMES.get(p,'?')}: {cnt}")
    # milestone は呼び出し側で Round 間比較して発火する (毎 Round 発火で膨張を防ぐ)
    return hard_p, hard_l, is_p2e_arr, int(purple_to_empty)


# ================================================================
# Tensor 変換
# ================================================================

def _patch_to_tensor_np(bgr_patch: np.ndarray) -> np.ndarray:
    resized = cv2.resize(bgr_patch, (PATCH_RESIZE_W, PATCH_RESIZE_H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    combined = np.concatenate([resized, hsv], axis=2).astype(np.float32) / 255.0
    return combined.transpose(2, 0, 1)


def patches_to_tensors(patches: np.ndarray, labels: np.ndarray):
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
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float().clamp(min=1.0)
    weight = (1.0 / counts)
    weight = weight / weight.sum() * NUM_CLASSES
    if use_focal:
        # gamma 2.0 → 1.5 に緩和 (ホールドアウト劣化対策)
        criterion = FocalLoss(alpha=weight, gamma=1.5)
    elif label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight)

    model.train()
    # weight_decay 追加で汎化を改善
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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
# Phase 1: 反復学習
# ================================================================

def phase1_iterative(patches, labels, max_rounds=10, patience=2) -> tuple[CnnPatchClassifier, float]:
    log("\n" + "=" * 60)
    log("Phase 1: 反復ハード例学習")
    # TODO(follow-up): patch_extraction.py にフレーム 1P/2P メタを追加し、
    # 暗所紫の根因である「片側プレイヤー領域の背景差」を層別サンプリングで解決する。
    # 現状は暗所拡張で近似しているがクラスレベル対処に留まる（検証Xで指摘）。
    log("=" * 60)

    best_acc = 0.0
    no_improve = 0
    best_cnn = None
    prev_p2e_count: int | None = None  # Round 間比較用

    for rnd in range(1, max_rounds + 1):
        log(f"\n--- Round {rnd}/{max_rounds} ---")

        hard_p, hard_l, is_p2e, p2e_count = None, None, None, 0
        if best_cnn is not None:
            hard_p, hard_l, is_p2e, p2e_count = mine_hard_examples(best_cnn, patches, labels)

        # milestone: Round 1 or 前 Round より悪化した時のみ発火（毎 Round 発火で膨張防止）
        if p2e_count >= 20 and (
            prev_p2e_count is None  # Round 1 の初回観測
            or p2e_count > prev_p2e_count * 1.5  # +50% 悪化
        ):
            milestone(
                "anomaly",
                f"紫→空 誤認 {p2e_count}件 (Round {rnd}, prev={prev_p2e_count})",
                purple_to_empty=int(p2e_count),
                prev_purple_to_empty=prev_p2e_count,
                round=rnd,
            )
        prev_p2e_count = p2e_count

        ds = PatchDataset(patches=patches, labels=labels)
        ds.stats.patches_total = len(labels)
        u, c = np.unique(labels, return_counts=True)
        ds.stats.per_class_count = {int(k): int(v) for k, v in zip(u, c)}
        balanced = balance_dataset(ds, empty_ratio_cap=0.40)

        tp, tl = balanced.patches, balanced.labels
        if hard_p is not None and len(hard_p) > 0:
            rng = np.random.default_rng(rnd)
            # 旧: min(3 + rnd, 8) → 過度なハード例過適合でホールドアウト劣化。緩和。
            oversample_normal = min(1 + rnd // 2, 3)

            # 紫×空誤認の oversample は背景差対策で通常より厚め。
            # - n_p2e < 5 (統計パワー不足) のときは 10× にブースト
            # - n_p2e ≥ 5 は基本 5×
            OVERSAMPLE_PURPLE_TO_EMPTY = 5
            OVERSAMPLE_PURPLE_TO_EMPTY_SMALL = 10
            PURPLE_RATIO_CAP = 0.20  # 紫が学習セットの 20% を超えたら倍率を 3 に戻す

            p2e_mask = is_p2e if is_p2e is not None else np.zeros(len(hard_p), dtype=bool)
            n_normal = int((~p2e_mask).sum())
            n_p2e = int(p2e_mask.sum())

            # 現在の紫比率を概算（balance_dataset 後）
            prepurple_ratio = float((tl == COLOR_PURPLE).mean()) if len(tl) > 0 else 0.0
            if prepurple_ratio >= PURPLE_RATIO_CAP:
                oversample_p2e = 3
                log(f"  紫比率 {prepurple_ratio:.3f} ≥ {PURPLE_RATIO_CAP} → oversample_p2e=3 に抑制")
            elif n_p2e < 5:
                oversample_p2e = OVERSAMPLE_PURPLE_TO_EMPTY_SMALL
                log(f"  n_p2e={n_p2e} < 5 → oversample_p2e={OVERSAMPLE_PURPLE_TO_EMPTY_SMALL} にブースト")
            else:
                oversample_p2e = OVERSAMPLE_PURPLE_TO_EMPTY

            aug_p: list[np.ndarray] = []
            aug_l: list[int] = []
            # 通常ハード例は augment_patch、紫→空誤認は augment_dark_variant を混ぜる
            for k, (hp, hl) in enumerate(zip(hard_p, hard_l)):
                if p2e_mask[k]:
                    # 半分は標準拡張、半分は暗所変種で学習シグナル強化
                    for _ in range(oversample_p2e):
                        if rng.random() < 0.5:
                            aug_p.append(augment_dark_variant(hp, rng))
                        else:
                            aug_p.append(augment_patch(hp, rng))
                        aug_l.append(hl)
                else:
                    for _ in range(oversample_normal):
                        aug_p.append(augment_patch(hp, rng))
                        aug_l.append(hl)

            if aug_p:
                tp = np.concatenate([tp, np.array(aug_p)])
                tl = np.concatenate([tl, np.array(aug_l)])

            # 追加後の紫比率をログ（前後比較で学習バランスの崩れを監視）
            postpurple_ratio = float((tl == COLOR_PURPLE).mean()) if len(tl) > 0 else 0.0
            log(
                f"ハード例追加: 通常 x{oversample_normal} ({n_normal}件) + "
                f"紫×空 x{oversample_p2e} 暗所拡張 ({n_p2e}件) → 合計{len(tl)} "
                f"紫比率 {prepurple_ratio:.3f}→{postpurple_ratio:.3f}"
            )

        # CutMix 合成サンプルを追加（混同しやすい色対の決定境界を強化）
        # 赤↔青、赤↔紫、青↔紫 の 3 ペアを重点的にミックス
        # round 0 のみ無効（最初は素のデータで学習させる）
        if rnd >= 1:
            cutmix_focus = {
                (COLOR_RED, COLOR_BLUE),
                (COLOR_RED, COLOR_PURPLE),
                (COLOR_BLUE, COLOR_PURPLE),
            }
            n_cutmix = max(50, len(tl) // 30)  # 約 3.3% 増し
            cm_p, cm_l = generate_cutmix_arrays(
                tp, tl,
                n_extra=n_cutmix,
                focus_pairs=cutmix_focus,
                seed=4242 + rnd,
            )
            if len(cm_l) > 0:
                tp = np.concatenate([tp, cm_p])
                tl = np.concatenate([tl, cm_l])
                log(
                    f"CutMix 追加: {len(cm_l)} 合成サンプル "
                    f"(focus=R-B/R-P/B-P) → 合計 {len(tl)}"
                )

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
            prev_best = best_acc
            best_acc = ta
            no_improve = 0
            best_cnn = cnn
            cnn.save(MODEL_DIR / f"cnn_p1_r{rnd:02d}.pt")
            cnn.save(MODEL_DIR / "cnn_best.pt")
            log(f"改善! test={ta:.4f}")
            # 0.5pt 以上の改善、または 99% 閾値クロスは milestone
            if ta - prev_best >= 0.005 or (prev_best < E2E_THRESHOLD <= ta):
                milestone(
                    "new_best" if ta < E2E_THRESHOLD else "threshold_99",
                    f"Phase 1 R{rnd} test={ta:.4f} (prev={prev_best:.4f}, +{(ta-prev_best)*100:.2f}pt)",
                    phase="phase1", round=rnd, acc=float(ta), prev=float(prev_best),
                )
        else:
            no_improve += 1
            log(f"改善なし ({no_improve}/{patience}) best={best_acc:.4f}")
            best_cnn = cnn

        del X_tr, y_tr, X_va, y_va, X_te, y_te
        gc.collect()

        if best_acc >= E2E_THRESHOLD:
            log(f"Phase 1: {E2E_THRESHOLD*100:.0f}%達成! フェーズ終了")
            break
        if no_improve >= patience:
            log(f"Phase 1: {patience}回改善なし → 次フェーズへ")
            break

    return best_cnn, best_acc


# ================================================================
# Phase 2: アーキテクチャ探索
# ================================================================

def build_wider_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        nn.Linear(32 * 4, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, NUM_CLASSES),
    )


def build_deeper_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(6, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        nn.Linear(32 * 4, 48), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(48, NUM_CLASSES),
    )


def build_residual_model() -> nn.Module:
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

    ds = PatchDataset(patches=patches, labels=labels)
    ds.stats.patches_total = len(labels)
    u, c = np.unique(labels, return_counts=True)
    ds.stats.per_class_count = {int(k): int(v) for k, v in zip(u, c)}
    balanced = balance_dataset(ds, empty_ratio_cap=0.40)
    tp, tl = balanced.patches, balanced.labels

    N = len(tl)
    perm = np.random.default_rng(99).permutation(N)
    s1, s2 = int(N * 0.8), int(N * 0.9)
    X_tr, y_tr = patches_to_tensors(tp[perm[:s1]], tl[perm[:s1]])
    X_va, y_va = patches_to_tensors(tp[perm[s1:s2]], tl[perm[s1:s2]])
    X_te, y_te = patches_to_tensors(tp[perm[s2:]], tl[perm[s2:]])

    best_acc = current_best
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
            best_name = name
            log(f"新ベスト! {name}: test={ta:.4f}")
            torch.save(model.state_dict(), str(MODEL_DIR / f"cnn_{name}.pt"))
        del model
        gc.collect()

    del X_tr, y_tr, X_va, y_va, X_te, y_te
    gc.collect()

    if best_name:
        log(f"Phase 2 ベスト: {best_name} ({best_acc:.4f})")
        milestone(
            "phase_complete" if best_acc - current_best < 0.005 else "new_best",
            f"Phase 2 完了: best={best_acc:.4f} arch={best_name}",
            phase="phase2", arch=best_name, acc=float(best_acc), prev=float(current_best),
        )
    else:
        log(f"Phase 2: 標準モデルが最良 ({current_best:.4f})")
        milestone(
            "phase_complete",
            f"Phase 2 完了: 新アーキでの改善なし ({current_best:.4f})",
            phase="phase2", acc=float(current_best),
        )

    return None, best_acc


# ================================================================
# Phase 3: 新規動画DL
# ================================================================

def _get_ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def try_download(vid_id: str, out_path: Path) -> bool:
    base_cmd = [
        YT_DLP, "-f",
        "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/"
        "bestvideo[ext=mp4][height<=720]",
        "-o", str(out_path), "--no-playlist", "--quiet",
        f"https://www.youtube.com/watch?v={vid_id}",
    ]
    strategies = [
        base_cmd[:1] + [f"--js-runtimes", f"node:{NODE_PATH}"] + base_cmd[1:],
        base_cmd[:1] + [f"--js-runtimes", f"node:{NODE_PATH}", "--cookies-from-browser", "chrome"] + base_cmd[1:],
        base_cmd[:1] + [f"--js-runtimes", f"node:{NODE_PATH}", "--cookies-from-browser", "firefox"] + base_cmd[1:],
        base_cmd[:1] + [f"--js-runtimes", f"node:{NODE_PATH}", "--cookies-from-browser", "edge"] + base_cmd[1:],
    ]
    for i, cmd in enumerate(strategies):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 5_000_000:
                log(f"  DL成功 (strategy {i+1})")
                return True
        except Exception as e:
            log(f"  DL strategy {i+1} 失敗: {e}")
        Path(str(out_path) + ".part").unlink(missing_ok=True)
    return False


def extract_patches_from_video(video_path: Path, tag: str, idx: int) -> Path | None:
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
        milestone(
            "phase_complete",
            f"Phase 3 DL完了: 新規{new_count}本追加 → 再学習トリガ",
            phase="phase3", new_videos=new_count,
        )
        CACHE_FILTERED.unlink(missing_ok=True)
        return True
    else:
        log("新規動画の追加なし")
        milestone(
            "phase_complete",
            "Phase 3 DL: 新規動画なし (既処理済み)",
            phase="phase3", new_videos=0,
        )
        return False


# ================================================================
# Phase 4: ラベルクリーニング
# ================================================================

def phase4_label_cleaning(patches, labels, cnn: CnnPatchClassifier) -> tuple[np.ndarray, np.ndarray, bool]:
    log("\n" + "=" * 60)
    log("Phase 4: ラベルノイズ検出 + クリーニング")
    log("=" * 60)

    rng = np.random.default_rng(777)
    sample_size = min(50000, len(labels))
    idx = rng.choice(len(labels), size=sample_size, replace=False)

    disagreements = {}
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

    sorted_pairs = sorted(disagreements.items(), key=lambda x: -x[1])
    for (t, p), cnt in sorted_pairs[:10]:
        tn = NAMES.get(t, "?")
        pn = NAMES.get(p, "?")
        log(f"  {tn}→{pn}: {cnt} ({cnt/sample_size*100:.2f}%)")

    if total_disagree / sample_size > 0.05:
        log("不一致率が高すぎるためクリーニングはスキップ")
        return patches, labels, False

    log("低不一致率 → 不一致サンプルを除去")
    keep = np.ones(len(labels), dtype=bool)
    removed = 0
    batch = 20000
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
# Phase 5: 高度技法
# ================================================================

def phase5_advanced(patches, labels, current_best: float) -> tuple[CnnPatchClassifier | None, float]:
    log("\n" + "=" * 60)
    log("Phase 5: 高度技法 (Label Smoothing / 長時間学習)")
    log("=" * 60)

    ds = PatchDataset(patches=patches, labels=labels)
    ds.stats.patches_total = len(labels)
    u, c = np.unique(labels, return_counts=True)
    ds.stats.per_class_count = {int(k): int(v) for k, v in zip(u, c)}
    balanced = balance_dataset(ds, empty_ratio_cap=0.40)
    tp, tl = balanced.patches, balanced.labels

    N = len(tl)
    perm = np.random.default_rng(55).permutation(N)
    s1, s2 = int(N * 0.8), int(N * 0.9)
    X_tr, y_tr = patches_to_tensors(tp[perm[:s1]], tl[perm[:s1]])
    X_va, y_va = patches_to_tensors(tp[perm[s1:s2]], tl[perm[s1:s2]])
    X_te, y_te = patches_to_tensors(tp[perm[s2:]], tl[perm[s2:]])

    best_acc = current_best
    best_cnn = None

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
        milestone(
            "new_best" if best_acc - current_best >= 0.005 else "phase_complete",
            f"Phase 5 完了: best={best_acc:.4f} (prev={current_best:.4f}, +{(best_acc-current_best)*100:.2f}pt)",
            phase="phase5", acc=float(best_acc), prev=float(current_best),
        )
    else:
        log(f"Phase 5: 改善なし (best={current_best:.4f})")
        milestone(
            "phase_complete",
            f"Phase 5 完了: 改善なし (best={current_best:.4f})",
            phase="phase5", acc=float(current_best),
        )

    return best_cnn, best_acc


# ================================================================
# 指標評価
# ================================================================

def evaluate_indicators(cnn: CnnPatchClassifier) -> None:
    config_path = Path("models/calibration_video01.json")
    if not config_path.exists():
        return
    config = CalibratedConfig.load(str(config_path))
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(classifier=gated, p1_region=config.p1_region, p2_region=config.p2_region)
    calc = IndicatorCalculator()
    scorer = Scorer()

    # 評価には 1920x1080 の生フレームを使用 (eval_cycle のモンタージュ画像は 848x1630 で
    # read_both_boards が歪みリサイズして 2P 領域が黒背景を指し、全セル空判定される既知バグ)
    sample_dir = Path("data/frames/sample")
    if not sample_dir.exists() or not any(sample_dir.iterdir()):
        # 次善: 生の read_NNNs.png があればそれを使う
        raw_candidates = sorted(Path("data/verify").glob("01_read_*.png"))
        frame_paths = raw_candidates[:4]
        if not frame_paths:
            log("評価用の生フレームが見つからず指標評価スキップ")
            return
    else:
        frame_paths = sorted(sample_dir.glob("*.png"))[:4] or sorted(sample_dir.glob("*.jpg"))[:4]

    log("--- 指標評価 (1P/2P 両側) ---")
    for fp in frame_paths:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        if frame.shape[:2] != (1080, 1920):
            log(f"  {fp.name}: 非標準解像度 {frame.shape[:2]} → スキップ")
            continue
        try:
            b1, b2 = reader.read_both_boards(frame)
            # 2P 側が全セル空なら異常として記録 (Board は count_puyos() API を使う)
            b1_puyos = b1.count_puyos()
            b2_puyos = b2.count_puyos()
            i1, i2 = calc.compute_all(b1), calc.compute_all(b2)
            result = scorer.score(i1, i2)
            log(f"  {fp.name}: {result.total_score:+.1f} ({result.advantage_side()}) 1P_puyos={b1_puyos} 2P_puyos={b2_puyos}")
            # 2P が 0 個なら anomaly milestone
            if b2_puyos == 0 and b1_puyos > 0:
                milestone("anomaly", f"2P読取り0件: {fp.name} (1P={b1_puyos})", frame=fp.name)
        except Exception as e:
            log(f"  {fp.name}: エラー {e}")


# ================================================================
# メインループ
# ================================================================

def run_one_cycle() -> float:
    """1サイクル (全Phase) 実行して best_acc を返す。"""
    patches, labels = load_all_data()
    best_acc = 0.0

    result = safe_run(lambda: phase1_iterative(patches, labels), "Phase 1")
    if result:
        best_cnn, best_acc = result
        if best_cnn:
            safe_run(lambda: evaluate_indicators(best_cnn), "Indicators 1")
        log(f"Phase 1 完了: best={best_acc:.4f}")

    # 40時間モードでは早期リターンせず全フェーズ実行（改善機会を逃さない）
    # FAST_MODE 環境変数が設定されていれば Phase 2 (アーキ探索) をスキップ
    # residual が既に最良として確立しているため、CutMix を入れた状態で
    # アーキ探索を回しても時間消費に対してリターン薄い。
    if os.environ.get("FAST_MODE", "0") in ("1", "true", "True"):
        log("Phase 2: スキップ (FAST_MODE)")
        milestone(
            "phase_complete",
            f"Phase 2: スキップ (FAST_MODE={os.environ.get('FAST_MODE')})",
            phase="phase2", acc=float(best_acc), skipped=True,
        )
    else:
        result = safe_run(lambda: phase2_architecture(patches, labels, best_acc), "Phase 2")
        if result:
            _, acc2 = result
            if acc2 > best_acc:
                best_acc = acc2
            log(f"Phase 2 完了: best={best_acc:.4f}")

    data_updated = safe_run(lambda: phase3_download_and_train(patches, labels, best_acc), "Phase 3")
    if data_updated:
        patches, labels = load_all_data(use_cache=False)
        log("新データで追加学習...")
        result = safe_run(lambda: phase1_iterative(patches, labels, max_rounds=5, patience=2), "Phase 3 再学習")
        if result:
            cnn3, acc3 = result
            if acc3 > best_acc:
                best_acc = acc3
                if cnn3:
                    safe_run(lambda: evaluate_indicators(cnn3), "Indicators 3")

    best_path = MODEL_DIR / "cnn_best.pt"
    if best_path.exists():
        best_cnn_for_clean = CnnPatchClassifier.load(best_path)
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
                            safe_run(lambda: evaluate_indicators(cnn4), "Indicators 4")

    result = safe_run(lambda: phase5_advanced(patches, labels, best_acc), "Phase 5")
    if result:
        cnn5, acc5 = result
        if acc5 > best_acc:
            best_acc = acc5
            if cnn5:
                safe_run(lambda: evaluate_indicators(cnn5), "Indicators 5")

    return best_acc


def _load_global_best_state() -> dict:
    """data/global_best.json を読む。存在しなければ空の辞書。"""
    import json
    if not GLOBAL_BEST_STATE.exists():
        return {"holdout_acc": 0.0, "internal_acc": 0.0, "ts": None, "cycle": None}
    try:
        return json.loads(GLOBAL_BEST_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {"holdout_acc": 0.0, "internal_acc": 0.0, "ts": None, "cycle": None}


def _promote_to_global_best(
    cnn_best_path: Path,
    best_acc: float,
    holdout_acc: float,
    sanity_ok: bool,
    symmetry_ok: bool,
    cycle: int | None = None,
) -> bool:
    """holdout 改善時に cnn_best.pt を cnn_global_best.pt にコピーして状態を記録。

    クラッシュ耐性のため tmp→os.replace の原子 rename で書き込む。
    sanity_ok=False の場合は昇格に必要な改善幅を大きめに (SANITY_FALSE_EPS=0.01) 要求する。

    Returns:
        昇格したら True、据え置きなら False。
    """
    import json
    import shutil
    SANITY_FALSE_EPS = 0.01  # sanity 違反下では holdout 1pt 以上の改善を要求
    state = _load_global_best_state()
    prev_holdout = float(state.get("holdout_acc", 0.0) or 0.0)

    required_eps = GLOBAL_BEST_EPS if sanity_ok else SANITY_FALSE_EPS
    if holdout_acc <= prev_holdout + required_eps:
        log(
            f"global best 据え置き: holdout {holdout_acc:.4f} <= prev {prev_holdout:.4f} "
            f"+ eps {required_eps} (sanity_ok={sanity_ok})"
        )
        return False

    # 原子的に model をコピー (tmp→replace)
    tmp_model = GLOBAL_BEST_MODEL.with_suffix(".pt.tmp")
    try:
        shutil.copyfile(cnn_best_path, tmp_model)
        os.replace(tmp_model, GLOBAL_BEST_MODEL)
    except Exception as e:
        log(f"[ERROR] global best copy 失敗: {e}")
        try:
            tmp_model.unlink(missing_ok=True)
        except Exception:
            pass
        return False

    new_state = {
        "holdout_acc": float(holdout_acc),
        "internal_acc": float(best_acc),
        "prev_holdout_acc": prev_holdout,
        "sanity_ok": bool(sanity_ok),
        "symmetry_ok": bool(symmetry_ok),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycle": cycle,
        "source": str(cnn_best_path),
    }
    # 原子的に state を書き込む。失敗したら model 実体と齟齬するので anomaly milestone を打つ。
    tmp_state = GLOBAL_BEST_STATE.with_suffix(".json.tmp")
    try:
        tmp_state.write_text(
            json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_state, GLOBAL_BEST_STATE)
    except Exception as e:
        log(f"[ERROR] global best state 書込失敗: {e} — model は更新済みで state と齟齬")
        try:
            tmp_state.unlink(missing_ok=True)
        except Exception:
            pass
        milestone(
            "anomaly",
            f"global best state 書込失敗: holdout {holdout_acc:.4f} model は更新済",
            holdout_acc=float(holdout_acc), error=str(e)[:200],
        )

    log(
        f"[GLOBAL BEST] holdout {prev_holdout:.4f} → {holdout_acc:.4f} (+{(holdout_acc-prev_holdout)*100:.2f}pt)"
    )
    milestone(
        "new_best",
        f"GLOBAL BEST 昇格 holdout={holdout_acc:.4f} (prev={prev_holdout:.4f})",
        kind_detail="global_best",
        holdout_acc=float(holdout_acc), prev_holdout=float(prev_holdout),
        internal_acc=float(best_acc), cycle=cycle,
    )
    return True


def _run_e2e_if_ready(best_acc: float, cycle: int | None = None) -> float | None:
    """99%以上に到達していたら E2E 検証を走らせ、holdout 精度を返す。未達なら None。

    holdout が過去のグローバルベストを上回っていれば cnn_global_best.pt に昇格保存する。

    FAST_MODE 時は CutMix で内部 acc が下がる傾向があるため、閾値を 0.97 に緩和。
    """
    fast_mode = os.environ.get("FAST_MODE", "0") in ("1", "true", "True")
    threshold = 0.97 if fast_mode else E2E_THRESHOLD
    if best_acc < threshold:
        log(f"精度 {best_acc:.4f} < {threshold:.2f} → E2E検証スキップ (FAST_MODE={fast_mode})")
        return None

    best_path = MODEL_DIR / "cnn_best.pt"
    if not best_path.exists():
        log("cnn_best.pt がないため E2E スキップ")
        return None

    try:
        from scripts.e2e_validate import run_e2e_validation
        cnn = CnnPatchClassifier.load(best_path)
        summary = run_e2e_validation(cnn, log=log)
        import json
        out = Path("data/e2e_log.jsonl")
        holdout_acc = summary.get("holdout", {}).get("overall_accuracy")
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "acc": best_acc,
            "holdout_acc": holdout_acc,
            "sanity_ok": summary.get("sanity", {}).get("ok"),
            "sanity_violations": len(summary.get("sanity", {}).get("violations", [])),
            "symmetry_ok": summary.get("symmetry", {}).get("ok"),
            "symmetry_violation": summary.get("symmetry", {}).get("max_symmetry_violation"),
        }
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"E2Eログ保存: {out}")
        # E2E 結果は必ず milestone へ送出
        sanity_ok = summary.get("sanity", {}).get("ok")
        sym_ok = summary.get("symmetry", {}).get("ok")
        kind = "e2e_result"
        if best_acc >= E2E_THRESHOLD and holdout_acc and holdout_acc >= 0.95:
            kind = "discovery"  # 実運用で使える水準に到達
        milestone(
            kind,
            f"E2E: internal={best_acc:.4f} holdout={holdout_acc:.4f} sanity={sanity_ok} sym={sym_ok}",
            acc=float(best_acc), holdout_acc=float(holdout_acc) if holdout_acc is not None else None,
            sanity_ok=sanity_ok, symmetry_ok=sym_ok,
        )

        # holdout 改善時のみ cnn_global_best.pt を更新
        if holdout_acc is not None:
            _promote_to_global_best(
                cnn_best_path=best_path,
                best_acc=best_acc,
                holdout_acc=float(holdout_acc),
                sanity_ok=bool(sanity_ok),
                symmetry_ok=bool(sym_ok),
                cycle=cycle,
            )

        return float(holdout_acc) if holdout_acc is not None else None
    except Exception as e:
        log(f"[ERROR] E2E検証: {e}")
        log(traceback.format_exc())
        return None


def main() -> None:
    log("\n" + "#" * 60)
    log("長時間自律改善パイプライン v2 開始 (40時間目安モード)")
    log("#" * 60)

    start_time = time.time()
    cycle = 0

    # 過去セッションの holdout best を読み込んで初期値にする。
    # 新規学習で過去値を下回っても global best は守られる。
    _prev_state = _load_global_best_state()
    global_best_acc = float(_prev_state.get("internal_acc", 0.0) or 0.0)
    global_best_holdout = float(_prev_state.get("holdout_acc", 0.0) or 0.0)
    if global_best_holdout > 0.0:
        log(f"前回の global best を継承: holdout={global_best_holdout:.4f} internal={global_best_acc:.4f}")
    stagnation = 0
    # FAST_MODE では stagnation 上限を縮めて早期 exit を促す
    STAGNATION_LIMIT = 2 if os.environ.get("FAST_MODE", "0") in ("1", "true", "True") else 4
    log(f"STAGNATION_LIMIT={STAGNATION_LIMIT} (FAST_MODE={os.environ.get('FAST_MODE', '0')})")
    MIN_CYCLES_BEFORE_EARLY_EXIT = 3  # 最初の数サイクルは様子見
    IMPROVE_EPS_INTERNAL = 0.0005  # 内部精度 0.05pt 以上で改善
    IMPROVE_EPS_HOLDOUT = 0.003  # ホールドアウト 0.3pt 以上で改善

    while True:
        cycle += 1
        elapsed_h = (time.time() - start_time) / 3600.0

        if elapsed_h >= MAX_HOURS:
            log(f"{MAX_HOURS}時間経過 → 終了")
            break
        if cycle > MAX_CYCLES:
            log(f"{MAX_CYCLES}サイクル到達 → 終了")
            break

        log(f"\n### Cycle {cycle} (経過 {elapsed_h:.2f}h / {MAX_HOURS}h) ###")
        milestone("cycle_start", f"Cycle {cycle} 開始 (経過 {elapsed_h:.2f}h)", cycle=cycle, elapsed_h=elapsed_h)
        t0 = time.time()
        try:
            best_acc = run_one_cycle()
        except Exception as e:
            log(f"[FATAL] Cycle {cycle}: {e}")
            log(traceback.format_exc())
            milestone("fatal", f"Cycle {cycle} FATAL: {str(e)[:80]}", cycle=cycle, error=str(e)[:200])
            time.sleep(60)
            continue

        elapsed = time.time() - t0
        log(f"\n### Cycle {cycle} 完了: best={best_acc:.4f} ({elapsed:.0f}s) ###")
        milestone(
            "cycle_complete",
            f"Cycle {cycle} 完了 best={best_acc:.4f} ({elapsed:.0f}s)",
            cycle=cycle, best_acc=float(best_acc), elapsed_s=float(elapsed),
            internal_best=float(global_best_acc), holdout_best=float(global_best_holdout),
        )

        # 99%以上なら E2E 検証を毎サイクル末に実行 (戻り値に holdout_acc を取得)
        holdout_acc = _run_e2e_if_ready(best_acc, cycle=cycle)
        if holdout_acc is None:
            holdout_acc = 0.0

        improved_internal = (best_acc > global_best_acc + IMPROVE_EPS_INTERNAL)
        improved_holdout = (holdout_acc > global_best_holdout + IMPROVE_EPS_HOLDOUT)
        if improved_internal or improved_holdout:
            stagnation = 0
            if improved_internal:
                global_best_acc = best_acc
            if improved_holdout:
                global_best_holdout = holdout_acc
            log(f"進捗あり: internal_best={global_best_acc:.4f} holdout_best={global_best_holdout:.4f}")
        else:
            stagnation += 1
            log(f"改善なし {stagnation}/{STAGNATION_LIMIT} (internal_best={global_best_acc:.4f} holdout_best={global_best_holdout:.4f})")

        if cycle >= MIN_CYCLES_BEFORE_EARLY_EXIT and stagnation >= STAGNATION_LIMIT:
            log(f"\n[EARLY EXIT] {STAGNATION_LIMIT}サイクル連続で内部もホールドアウトも改善せず → 意味のない処理と判定し終了")
            milestone(
                "early_exit",
                f"{STAGNATION_LIMIT}サイクル改善なしで早期終了",
                cycle=cycle, internal_best=float(global_best_acc), holdout_best=float(global_best_holdout),
            )
            break

        gc.collect()
        time.sleep(5)

    total_h = (time.time() - start_time) / 3600.0
    log("\n" + "#" * 60)
    log(f"長時間自律改善パイプライン v2 終了 (経過 {total_h:.2f}h, {cycle}サイクル)")
    log(f"最終 internal_best={global_best_acc:.4f} holdout_best={global_best_holdout:.4f}")
    log("#" * 60)


if __name__ == "__main__":
    main()
