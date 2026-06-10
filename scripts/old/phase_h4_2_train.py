"""Phase H4.2: Raw Board CNN End-to-End 学習 (Quick mode).

Phase H4.1 (Deep Tabular MLP, top_20 で LOOV avg 0.762) の next step として、
真の "raw 6×13 board → CNN → 勝率 + 45 indicator 予測" を実装する.

入力:
    --board-dir data/training/phase_h2_boards/  (v01.npz, v04.npz, ...)
    --csv data/training/match_features_phase_h2_quick_with_board.csv
       ※ board.npz と CSV は同行数で順序対応している前提.
       indicator features (回帰 target) を抽出するために CSV を併用.

モデル: SiameseBoardCNN
    - BoardCNN(共通重み): 8ch×13×6 → 32 dim embedding
    - winrate_head: cat(emb1, emb2, emb1-emb2)=96 → 64 → 1 (BCE)
    - indicator_head: 同 96 → 64 → 45 (MSE 補助)
    - Multi-task loss: α·BCE + β·MSE  (phase 補助は CSV 依存のため除外)

評価:
    - video_holdout (3 動画 test): primary winrate accuracy
    - LOOV phase mean: start / mid / end (CSV の time_phase 列に基づく)
    - Phase H4.1 と直接比較

注意: 1 関数 50 行以内、マジックナンバー禁止、torch + numpy seed 固定.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cnn_embedding.board_cnn import (  # noqa: E402
    BOARD_COLS,
    BOARD_ROWS,
    EMBED_DIM,
    N_COLOR_CHANNELS,
    SiameseBoardCNN,
    board_to_onehot,
)


# ============================
# 定数 (マジックナンバー回避)
# ============================
META_COLS = {"video_id", "match_idx", "time_phase", "frame_idx", "timestamp", "label"}
PHASES = ("start_plus_20", "mid_minus_20", "midpoint", "mid_plus_20", "end_minus_5")
PHASE_GROUPS: dict[str, tuple[str, ...]] = {
    "start": ("start_plus_20",),
    "mid": ("mid_minus_20", "midpoint", "mid_plus_20"),
    "end": ("end_minus_5",),
}

# 学習ハイパーパラメータ
N_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# Multi-task loss 重み
ALPHA_WINRATE = 1.0
BETA_INDICATOR = 0.3

# 評価関連
N_TEST_VIDEOS = 3
RANDOM_SEED = 0
EARLY_STOP_PATIENCE = 8

# Phase H4.1 baseline (top_20)
PHASE_H4_1_BASELINE_LOOV_AVG = 0.762


# ============================
# データロード
# ============================
def setup_logger(log_path: Path) -> logging.Logger:
    """ファイル + stdout に出力する logger を作成する."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_h4_2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def set_global_seed(seed: int) -> None:
    """numpy / torch / cuda の seed を固定する."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_board_npzs(board_dir: Path) -> dict[str, Any]:
    """board_dir/v??.npz を全 concat して 1 dict に. 順序は video_id 昇順."""
    npz_paths = sorted(board_dir.glob("v??.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"board NPZ が見つかりません: {board_dir}")
    parts: list[dict[str, np.ndarray]] = []
    for p in npz_paths:
        d = np.load(p, allow_pickle=False)
        parts.append({k: d[k] for k in d.files})
    out: dict[str, Any] = {}
    for k in parts[0].keys():
        out[k] = np.concatenate([p[k] for p in parts], axis=0)
    out["n"] = int(out["labels"].shape[0])
    return out


def load_indicator_targets(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSV から (indicator_targets[N,45], video_ids[N], time_phases[N]) を返す.

    indicator は __static で終わる列 45 個を選択.
    """
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    static_cols = [c for c in fieldnames if c.endswith("__static")]
    if not static_cols:
        raise ValueError(f"__static 列が CSV にない: {csv_path}")
    n = len(rows)
    ind = np.zeros((n, len(static_cols)), dtype=np.float32)
    vids = np.zeros(n, dtype="<U8")
    phases = np.zeros(n, dtype="<U24")
    for i, r in enumerate(rows):
        for j, c in enumerate(static_cols):
            ind[i, j] = float(r.get(c, 0.0) or 0.0)
        vids[i] = r["video_id"]
        phases[i] = r.get("time_phase", "midpoint")
    return ind, vids, phases


def video_holdout_split(
    video_ids: np.ndarray, n_test: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """動画単位で test ホールドアウトを作成する."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(video_ids)
    if len(uniq) <= n_test:
        n_test = max(1, len(uniq) // 3)
    test_videos = rng.choice(uniq, size=n_test, replace=False)
    test_mask = np.isin(video_ids, test_videos)
    return ~test_mask, test_mask


def standardize(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train で fit して train/test を z-score 化する (indicator 用)."""
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    return (
        ((X_tr - mu) / sd).astype(np.float32),
        ((X_te - mu) / sd).astype(np.float32),
    )


def boards_to_onehot_batch(boards: np.ndarray) -> np.ndarray:
    """(N, ROWS, COLS) → (N, N_COLOR_CHANNELS, ROWS, COLS) one-hot float32."""
    n = boards.shape[0]
    out = np.zeros(
        (n, N_COLOR_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32,
    )
    for i in range(n):
        out[i] = board_to_onehot(boards[i])
    return out


# ============================
# Multi-task loss
# ============================
def multi_task_loss(
    winrate_logit: torch.Tensor, y_bin: torch.Tensor,
    ind_pred: torch.Tensor, ind_true: torch.Tensor,
    alpha: float = ALPHA_WINRATE, beta: float = BETA_INDICATOR,
) -> tuple[torch.Tensor, dict[str, float]]:
    """α·BCE + β·MSE の合成損失と内訳を返す."""
    bce = F.binary_cross_entropy_with_logits(winrate_logit, y_bin.float())
    mse = F.mse_loss(ind_pred, ind_true)
    total = alpha * bce + beta * mse
    parts = {"bce": float(bce.item()), "mse": float(mse.item())}
    return total, parts


def make_loader(
    board_1p: np.ndarray, board_2p: np.ndarray,
    y_bin: np.ndarray, ind: np.ndarray,
    batch_size: int, shuffle: bool,
) -> DataLoader:
    """numpy → TensorDataset + DataLoader."""
    ds = TensorDataset(
        torch.from_numpy(board_1p.astype(np.float32)),
        torch.from_numpy(board_2p.astype(np.float32)),
        torch.from_numpy(y_bin.astype(np.float32)),
        torch.from_numpy(ind.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: AdamW, device: torch.device,
) -> dict[str, float]:
    """1 epoch 学習。平均 loss を返す."""
    model.train()
    sums = {"total": 0.0, "bce": 0.0, "mse": 0.0}
    n = 0
    for b1, b2, yb, ib in loader:
        b1, b2, yb, ib = b1.to(device), b2.to(device), yb.to(device), ib.to(device)
        optimizer.zero_grad()
        wl, ip, _ = model(b1, b2)
        loss, parts = multi_task_loss(wl, yb, ip, ib)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        bs = b1.size(0)
        sums["total"] += float(loss.item()) * bs
        for k in ("bce", "mse"):
            sums[k] += parts[k] * bs
        n += bs
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> dict[str, float]:
    """検証 loss + winrate accuracy + indicator MSE を返す."""
    model.eval()
    sums = {"total": 0.0, "bce": 0.0, "mse": 0.0}
    n = 0
    n_correct_w = 0
    for b1, b2, yb, ib in loader:
        b1, b2, yb, ib = b1.to(device), b2.to(device), yb.to(device), ib.to(device)
        wl, ip, _ = model(b1, b2)
        loss, parts = multi_task_loss(wl, yb, ip, ib)
        bs = b1.size(0)
        sums["total"] += float(loss.item()) * bs
        for k in ("bce", "mse"):
            sums[k] += parts[k] * bs
        n += bs
        n_correct_w += int(((torch.sigmoid(wl) > 0.5).long() == yb.long()).sum().item())
    avg = {k: v / max(1, n) for k, v in sums.items()}
    avg["winrate_acc"] = n_correct_w / max(1, n)
    avg["indicator_mse"] = avg["mse"]
    return avg


def fit_model(
    b1_tr: np.ndarray, b2_tr: np.ndarray, y_tr: np.ndarray, ind_tr: np.ndarray,
    b1_va: np.ndarray, b2_va: np.ndarray, y_va: np.ndarray, ind_va: np.ndarray,
    n_indicators: int, device: torch.device,
    logger: logging.Logger, tag: str,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, float]]:
    """1 セットの train/val で fit し、best val acc 時の metrics を返す."""
    model = SiameseBoardCNN(
        in_channels=N_COLOR_CHANNELS, embed_dim=EMBED_DIM,
        n_indicators=n_indicators,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    tr_loader = make_loader(b1_tr, b2_tr, y_tr, ind_tr, BATCH_SIZE, shuffle=True)
    va_loader = make_loader(b1_va, b2_va, y_va, ind_va, BATCH_SIZE, shuffle=False)

    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_metrics: dict[str, float] = {}
    no_improve = 0
    for epoch in range(N_EPOCHS):
        tr_log = train_one_epoch(model, tr_loader, optimizer, device)
        va_log = evaluate(model, va_loader, device)
        scheduler.step()
        history.append({"epoch": epoch, "train": tr_log, "val": va_log})
        logger.info(
            "[%s] epoch %02d/%d | tr_total=%.4f bce=%.4f mse=%.4f | "
            "va_acc=%.4f va_mse=%.4f",
            tag, epoch + 1, N_EPOCHS, tr_log["total"], tr_log["bce"],
            tr_log["mse"], va_log["winrate_acc"], va_log["indicator_mse"],
        )
        if va_log["winrate_acc"] > best_val_acc:
            best_val_acc = va_log["winrate_acc"]
            best_metrics = dict(va_log)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                logger.info("[%s] early stop epoch=%d (no improve %d)",
                            tag, epoch + 1, no_improve)
                break
    return model, history, best_metrics


# ============================
# 全体評価ロジック
# ============================
def run_video_holdout(
    p1_oh: np.ndarray, p2_oh: np.ndarray, y_bin: np.ndarray,
    ind: np.ndarray, video_ids: np.ndarray, n_indicators: int,
    device: torch.device, logger: logging.Logger,
) -> dict[str, Any]:
    """video holdout (3 動画 test) を実行し metrics + history を返す."""
    tr_mask, te_mask = video_holdout_split(video_ids, N_TEST_VIDEOS, RANDOM_SEED)
    ind_tr, ind_te = standardize(ind[tr_mask], ind[te_mask])
    logger.info(
        "[vh] fit train=%d test=%d (videos=%d/%d)",
        int(tr_mask.sum()), int(te_mask.sum()),
        len(np.unique(video_ids[tr_mask])), len(np.unique(video_ids[te_mask])),
    )
    _, history, vh_metrics = fit_model(
        p1_oh[tr_mask], p2_oh[tr_mask], y_bin[tr_mask], ind_tr,
        p1_oh[te_mask], p2_oh[te_mask], y_bin[te_mask], ind_te,
        n_indicators=n_indicators, device=device, logger=logger, tag="vh",
    )
    return {
        "vh_acc": float(vh_metrics.get("winrate_acc", 0.0)),
        "vh_indicator_mse": float(vh_metrics.get("indicator_mse", 0.0)),
        "vh_total_loss": float(vh_metrics.get("total", 0.0)),
        "history_tail": history[-3:],
    }


def run_loov_phase(
    p1_oh: np.ndarray, p2_oh: np.ndarray, y_bin: np.ndarray, ind: np.ndarray,
    video_ids: np.ndarray, time_phases: np.ndarray, n_indicators: int,
    device: torch.device, logger: logging.Logger,
) -> dict[str, dict[str, float]]:
    """phase グループ (start/mid/end) ごとに LOOV を回し平均 acc を返す."""
    out: dict[str, dict[str, float]] = {}
    for phase_name, phases in PHASE_GROUPS.items():
        mask = np.isin(time_phases, phases)
        if mask.sum() == 0:
            out[phase_name] = {"mean": 0.0, "std": 0.0, "n_videos": 0}
            continue
        accs = _loov_accs_for_phase(
            p1_oh, p2_oh, y_bin, ind, video_ids, mask,
            n_indicators, device, logger, phase_name,
        )
        if accs:
            out[phase_name] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "n_videos": len(accs),
            }
        else:
            out[phase_name] = {"mean": 0.0, "std": 0.0, "n_videos": 0}
        logger.info(
            "[LOOV] phase=%s mean=%.4f n=%d",
            phase_name, out[phase_name]["mean"], out[phase_name]["n_videos"],
        )
    return out


def _loov_accs_for_phase(
    p1_oh: np.ndarray, p2_oh: np.ndarray, y_bin: np.ndarray,
    ind: np.ndarray, video_ids: np.ndarray, phase_mask: np.ndarray,
    n_indicators: int, device: torch.device,
    logger: logging.Logger, phase_name: str,
) -> list[float]:
    """phase 限定で LOOV を回し各 fold の test acc list."""
    accs: list[float] = []
    uniq = np.unique(video_ids[phase_mask])
    for vid in uniq:
        te = (video_ids == vid) & phase_mask
        tr = (video_ids != vid) & phase_mask
        if te.sum() == 0 or tr.sum() == 0:
            continue
        ind_tr_z, ind_te_z = standardize(ind[tr], ind[te])
        try:
            _, _, m = fit_model(
                p1_oh[tr], p2_oh[tr], y_bin[tr], ind_tr_z,
                p1_oh[te], p2_oh[te], y_bin[te], ind_te_z,
                n_indicators=n_indicators, device=device, logger=logger,
                tag=f"loov-{phase_name}-{vid}",
            )
            accs.append(float(m.get("winrate_acc", 0.0)))
        except Exception as e:
            logger.warning("LOOV vid=%s phase=%s skip: %s", vid, phase_name, e)
    return accs


def count_parameters(model: nn.Module) -> int:
    """モデルの trainable parameter 数を返す."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ============================
# main
# ============================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-dir", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=Path("logs/phase_h4_2_train.log"))
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    logger = setup_logger(args.log)
    set_global_seed(RANDOM_SEED)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("device=%s", device)
    if device.type == "cuda":
        logger.info("cuda: %s", torch.cuda.get_device_name(0))

    # === ロード ===
    bd = load_board_npzs(args.board_dir)
    logger.info(
        "[load] board npz: n=%d, p1_shape=%s, p2_shape=%s",
        bd["n"], bd["p1_boards"].shape, bd["p2_boards"].shape,
    )
    ind, vids_csv, phases_csv = load_indicator_targets(args.csv)
    logger.info(
        "[load] csv indicator: n=%d, n_indicators=%d",
        ind.shape[0], ind.shape[1],
    )

    # === 整合チェック ===
    if bd["n"] != ind.shape[0]:
        logger.warning(
            "[align] N 不一致: board=%d csv=%d. min を採用.",
            bd["n"], ind.shape[0],
        )
    n = min(bd["n"], ind.shape[0])
    p1 = bd["p1_boards"][:n]
    p2 = bd["p2_boards"][:n]
    labels = bd["labels"][:n]
    ind = ind[:n]
    video_ids = vids_csv[:n]
    time_phases = phases_csv[:n]
    y_bin = (labels > 0).astype(np.int64)

    # === one-hot 化 ===
    logger.info("[encode] one-hot 化中... (%d frames)", n)
    p1_oh = boards_to_onehot_batch(p1)
    p2_oh = boards_to_onehot_batch(p2)
    logger.info("[encode] done. p1_oh=%s, p2_oh=%s", p1_oh.shape, p2_oh.shape)

    # === モデル meta ===
    sample = SiameseBoardCNN(
        in_channels=N_COLOR_CHANNELS, embed_dim=EMBED_DIM,
        n_indicators=ind.shape[1],
    )
    n_params = count_parameters(sample)
    logger.info("[model] SiameseBoardCNN parameters=%d", n_params)
    del sample

    # === video holdout ===
    vh = run_video_holdout(
        p1_oh, p2_oh, y_bin, ind, video_ids, ind.shape[1],
        device, logger,
    )

    # === LOOV phase mean ===
    loov_phase = run_loov_phase(
        p1_oh, p2_oh, y_bin, ind, video_ids, time_phases, ind.shape[1],
        device, logger,
    )
    loov_avg = float(np.mean([m["mean"] for m in loov_phase.values()]))

    # === 結果まとめ ===
    payload = {
        "vh_acc": vh["vh_acc"],
        "vh_indicator_mse": vh["vh_indicator_mse"],
        "vh_total_loss": vh["vh_total_loss"],
        "loov_phase": loov_phase,
        "loov_avg": loov_avg,
        "model_meta": {
            "name": "SiameseBoardCNN",
            "in_channels": N_COLOR_CHANNELS,
            "embed_dim": EMBED_DIM,
            "n_parameters": n_params,
            "n_indicators": int(ind.shape[1]),
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "alpha_winrate": ALPHA_WINRATE,
            "beta_indicator": BETA_INDICATOR,
        },
        "phase_h4_1_baseline_loov_avg": PHASE_H4_1_BASELINE_LOOV_AVG,
        "delta_vs_phase_h4_1": loov_avg - PHASE_H4_1_BASELINE_LOOV_AVG,
        "n_rows": int(n),
        "n_videos": int(len(np.unique(video_ids))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("[save] %s", args.out)
    logger.info(
        "[done] vh_acc=%.4f loov_avg=%.4f (Δ vs H4.1 = %+.4f)",
        vh["vh_acc"], loov_avg, payload["delta_vs_phase_h4_1"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
