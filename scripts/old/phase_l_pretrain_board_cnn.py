"""Phase L: SimCLR による BoardCNN の自己教師あり事前学習スクリプト.

目的:
    - data/training/phase_h2_boards/v??.npz の board state を contrastive learning で
      事前学習し、Phase L 用 board CNN encoder weights を models/board_cnn_pretrained.pt
      に保存する.
    - GPU (RTX 4060 8GB) を活用、mixed precision (torch.cuda.amp) で memory 半減.

設計:
    - dataset: phase_h2_boards/*.npz の p1_boards + p2_boards を全 concat
        (= 8,774 boards、11 動画).
    - augmentation: 色 permutation / 左右反転 / partial mask / row shift.
    - encoder: SimCLRBoardEncoder (BoardCNN + 2 層 projection).
    - loss: NT-Xent (temperature 0.5).
    - optimizer: AdamW + Cosine warmup (linear warmup → cosine annealing).
    - 学習結果は state_dict として encoder のみ models/board_cnn_pretrained.pt に保存.

使用例:
    PYTHONPATH=. python -m scripts.phase_l_pretrain_board_cnn \
        --boards-dir data/training/phase_h2_boards \
        --out models/board_cnn_pretrained.pt \
        --epochs 100 --batch-size 256

注意:
    - 1 関数 50 行以内、マジックナンバー禁止.
    - 既存 phase_h2 / phase_i / phase_h4 系ファイルには触らない.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cnn_embedding.board_cnn import (  # noqa: E402
    BOARD_COLS,
    BOARD_ROWS,
    N_COLOR_CHANNELS,
    board_to_onehot,
)
from src.cnn_embedding.pretrain import (  # noqa: E402
    AugmentConfig,
    SimCLRBoardEncoder,
    augment_board,
    nt_xent_loss,
)


# ============================
# 定数
# ============================
DEFAULT_EPOCHS: int = 100
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_LR: float = 3e-4
DEFAULT_WD: float = 1e-4
DEFAULT_TEMPERATURE: float = 0.5
DEFAULT_EMBED_DIM: int = 128
DEFAULT_PROJECTION_DIM: int = 64
DEFAULT_HIDDEN_DIM: int = 256
DEFAULT_WARMUP_EPOCHS: int = 5
DEFAULT_NUM_WORKERS: int = 4
DEFAULT_SEED: int = 0
GRAD_CLIP_NORM: float = 1.0
LOG_EVERY_N_STEPS: int = 50

# OOM retry: batch_size を半減する下限
MIN_BATCH_SIZE: int = 32


# ============================
# Dataset
# ============================
class BoardSimCLRDataset(Dataset):
    """phase_h2_boards から board state を読み込み 2 augmented view を返す."""

    def __init__(
        self,
        boards: np.ndarray,
        cfg: AugmentConfig | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if boards.ndim != 3 or boards.shape[1:] != (BOARD_ROWS, BOARD_COLS):
            raise ValueError(f"boards shape 不正: {boards.shape}")
        self.boards = boards.astype(np.int8, copy=False)
        self.cfg = cfg if cfg is not None else AugmentConfig()
        self._base_seed = seed

    def __len__(self) -> int:
        return int(self.boards.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # worker 毎・index 毎に独立な乱数源を作る (DataLoader の pickle 安全)
        rng = np.random.default_rng((self._base_seed, idx, time.time_ns() & 0xFFFF))
        grid = self.boards[idx]
        onehot = board_to_onehot(grid)
        v1 = augment_board(onehot, rng, self.cfg)
        v2 = augment_board(onehot, rng, self.cfg)
        return torch.from_numpy(v1), torch.from_numpy(v2)


# ============================
# Loader / setup
# ============================
def load_all_boards(boards_dir: Path) -> np.ndarray:
    """phase_h2_boards/v??.npz から p1_boards + p2_boards を全 concat."""
    npz_paths = sorted(boards_dir.glob("v??.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"NPZ が無い: {boards_dir}")
    parts: list[np.ndarray] = []
    for p in npz_paths:
        d = np.load(p, allow_pickle=False)
        if "p1_boards" in d.files:
            parts.append(d["p1_boards"].astype(np.int8))
        if "p2_boards" in d.files:
            parts.append(d["p2_boards"].astype(np.int8))
    if not parts:
        raise ValueError(f"boards が空: {boards_dir}")
    return np.concatenate(parts, axis=0)


def setup_logger(log_path: Path) -> logging.Logger:
    """stdout + file 出力 logger を返す."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_l_pretrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    """numpy / torch / cuda の seed を固定."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================
# LR schedule (linear warmup → cosine)
# ============================
def lr_at_step(
    step: int, total_steps: int, warmup_steps: int, base_lr: float,
) -> float:
    """linear warmup → cosine annealing."""
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def apply_lr(optimizer: AdamW, lr: float) -> None:
    """全 param group の lr を上書き."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ============================
# Train loop
# ============================
def train_one_epoch(
    model: SimCLRBoardEncoder,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: "torch.amp.GradScaler | None",
    device: torch.device,
    temperature: float,
    epoch: int,
    epochs_total: int,
    steps_per_epoch: int,
    warmup_steps: int,
    base_lr: float,
    logger: logging.Logger,
) -> dict[str, float]:
    """1 epoch SimCLR 学習. 平均 loss と最終 lr を返す."""
    model.train()
    use_amp = scaler is not None
    sum_loss = 0.0
    n_batches = 0
    last_lr = 0.0
    for batch_idx, (v1, v2) in enumerate(loader):
        global_step = epoch * steps_per_epoch + batch_idx
        total_steps = epochs_total * steps_per_epoch
        last_lr = lr_at_step(global_step, total_steps, warmup_steps, base_lr)
        apply_lr(optimizer, last_lr)
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, z1 = model(v1)
                _, z2 = model(v2)
                loss = nt_xent_loss(z1, z2, temperature=temperature)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        sum_loss += float(loss.item())
        n_batches += 1
        if (batch_idx + 1) % LOG_EVERY_N_STEPS == 0:
            logger.info(
                "  step %d/%d loss=%.4f lr=%.2e",
                batch_idx + 1, steps_per_epoch, loss.item(), last_lr,
            )
    return {"loss": sum_loss / max(1, n_batches), "lr": last_lr}


# ============================
# 保存
# ============================
def save_encoder_only(
    model: SimCLRBoardEncoder, out_path: Path, meta: dict,
) -> None:
    """encoder backbone (BoardCNN) のみ + meta 情報を保存."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder_state_dict": model.encoder.state_dict(),
        "full_state_dict": model.state_dict(),
        "meta": meta,
    }
    torch.save(payload, out_path)


# ============================
# CLI
# ============================
def parse_args() -> argparse.Namespace:
    """コマンドライン引数."""
    p = argparse.ArgumentParser(
        description="Phase L: SimCLR による BoardCNN 事前学習",
    )
    p.add_argument("--boards-dir", type=Path,
                   default=Path("data/training/phase_h2_boards"))
    p.add_argument("--out", type=Path,
                   default=Path("models/board_cnn_pretrained.pt"))
    p.add_argument("--log", type=Path,
                   default=Path("logs/phase_l_pretrain.log"))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WD)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    p.add_argument("--projection-dim", type=int, default=DEFAULT_PROJECTION_DIM)
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--no-amp", action="store_true",
                   help="mixed precision を無効化 (debug 用)")
    return p.parse_args()


def build_loader(
    boards: np.ndarray, batch_size: int, num_workers: int, seed: int,
) -> DataLoader:
    """SimCLR Dataset → DataLoader."""
    ds = BoardSimCLRDataset(boards, cfg=AugmentConfig(), seed=seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def run_training(
    boards: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int,
) -> tuple[SimCLRBoardEncoder, list[dict]]:
    """SimCLR 学習を実行し model + history を返す."""
    loader = build_loader(boards, batch_size, args.num_workers, args.seed)
    steps_per_epoch = max(1, len(loader))
    warmup_steps = args.warmup_epochs * steps_per_epoch
    model = SimCLRBoardEncoder(
        embed_dim=args.embed_dim,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        in_channels=N_COLOR_CHANNELS,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    history: list[dict] = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        log = train_one_epoch(
            model, loader, optimizer, scaler, device,
            temperature=args.temperature,
            epoch=epoch, epochs_total=args.epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_steps=warmup_steps,
            base_lr=args.lr, logger=logger,
        )
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch + 1, "loss": log["loss"],
            "lr": log["lr"], "elapsed_sec": elapsed,
            "batch_size": batch_size,
        })
        logger.info(
            "epoch %d/%d loss=%.4f lr=%.2e (%.1fs)",
            epoch + 1, args.epochs, log["loss"], log["lr"], elapsed,
        )
        if log["loss"] < best_loss:
            best_loss = log["loss"]
            save_encoder_only(model, args.out, meta={
                "best_loss": best_loss, "epoch": epoch + 1,
                "embed_dim": args.embed_dim,
                "batch_size": batch_size,
                "history_tail": history[-5:],
            })
    return model, history


def main() -> int:
    """CLI エントリポイント. OOM 時は batch_size 半減で retry."""
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger(args.log)
    boards = load_all_boards(args.boards_dir)
    logger.info("boards loaded: %s", boards.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s amp=%s", device, not args.no_amp)
    bs = args.batch_size
    history_dump: list[dict] = []
    while bs >= MIN_BATCH_SIZE:
        try:
            _, history = run_training(boards, args, device, logger, bs)
            history_dump = history
            break
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM at batch_size=%d → 半減して retry", bs)
            torch.cuda.empty_cache()
            bs //= 2
    if not history_dump:
        logger.error("MIN_BATCH_SIZE=%d でも OOM. 中断.", MIN_BATCH_SIZE)
        return 1
    hist_path = args.out.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history_dump, ensure_ascii=False, indent=2))
    logger.info("history saved: %s", hist_path)
    logger.info("done. final batch_size=%d", bs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
