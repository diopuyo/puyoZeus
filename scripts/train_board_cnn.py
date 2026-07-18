"""SiameseBoardCNN (S3) を board_pairs_fixed.npz で学習・評価するスクリプト。

## 概要
- 入力: board_pairs_fixed.npz (build_board_pairs_lean.py 出力)。
- モデル: src/cnn_embedding/board_cnn.py の SiameseBoardCNN。
- 分割: video_id 単位 holdout (val = video_id の 2 割)。
- 損失: BCEWithLogitsLoss (won 2 値分類)。
- 評価: 全体 val AUC + 位相別 val AUC (序盤/中盤/終盤)。
- 位相定義: 各ペアの 1P+2P 盤面ぷよ合計数を 3 分位で近似。

## CLI
    python -m scripts.train_board_cnn \\
        --pairs data/indicators_v2/board_pairs_fixed.npz \\
        --epochs 20 \\
        --out models/board_cnn_s3.pt \\
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch は import 成功するまで遅延しない (起動時に確認)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.cnn_embedding.board_cnn import (
    SiameseBoardCNN,
    board_to_onehot,
    BOARD_ROWS,
    BOARD_COLS,
)

# ============================
# 定数
# ============================
VAL_RATIO: float = 0.2          # video_id 単位で val に回す割合
DEFAULT_EPOCHS: int = 20
DEFAULT_BATCH_SIZE: int = 128
DEFAULT_LR: float = 1e-3
PHASE_LABELS: list[str] = ["序盤", "中盤", "終盤"]
# 盤面の全セル数 (1P + 2P): 最大値
MAX_PUYO_COUNT: int = BOARD_ROWS * BOARD_COLS * 2  # 156


# ============================
# データセット
# ============================
class BoardPairDataset(Dataset):
    """board_pairs npz をラップする Dataset。

    won=NaN のサンプルは __getitem__ に含まれないよう事前フィルタ済み前提。
    """

    def __init__(
        self,
        board_1p: np.ndarray,   # (N, 13, 6) int8
        board_2p: np.ndarray,   # (N, 13, 6) int8
        won: np.ndarray,         # (N,) float32
        puyo_count: np.ndarray,  # (N,) int  (1P+2P 合計ぷよ数)
    ) -> None:
        self.board_1p = board_1p
        self.board_2p = board_2p
        self.won = won.astype(np.float32)
        self.puyo_count = puyo_count

    def __len__(self) -> int:
        return len(self.won)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = torch.from_numpy(board_to_onehot(self.board_1p[idx]))
        x2 = torch.from_numpy(board_to_onehot(self.board_2p[idx]))
        y = torch.tensor(self.won[idx], dtype=torch.float32)
        return x1, x2, y


def _count_puyo(grid: np.ndarray) -> int:
    """(13, 6) int8 盤面の非空セル (color >= 1) 数を返す。"""
    return int(np.sum(grid > 0))


def _load_pairs(pairs_path: Path) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """npz を読み込み (board_1p, board_2p, won, video_id, puyo_count) を返す。

    won=NaN は除外する。
    """
    d = np.load(str(pairs_path), allow_pickle=True)
    board_1p: np.ndarray = d["board_1p"]   # (N, 13, 6)
    board_2p: np.ndarray = d["board_2p"]   # (N, 13, 6)
    won: np.ndarray = d["won"]             # (N,) float32
    video_id: np.ndarray = d["video_id"]  # (N,) str

    # won=NaN を除外
    valid_mask = ~np.isnan(won)
    board_1p = board_1p[valid_mask]
    board_2p = board_2p[valid_mask]
    won = won[valid_mask]
    video_id = video_id[valid_mask]

    # ぷよ合計数を算出 (位相分類に使用)
    puyo_count = np.array(
        [_count_puyo(board_1p[i]) + _count_puyo(board_2p[i]) for i in range(len(board_1p))],
        dtype=np.int32,
    )
    print(f"[train_board_cnn] 有効ペア数: {len(won)}  (won=1: {int(np.sum(won==1))}, won=0: {int(np.sum(won==0))})")
    return board_1p, board_2p, won, video_id, puyo_count


def _split_video_ids(
    video_id: np.ndarray,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """video_id 単位で train/val を分割する。

    Returns:
        train_mask, val_mask: bool 配列
    """
    rng = np.random.default_rng(seed)
    unique_vids = np.unique(video_id)
    rng.shuffle(unique_vids)
    n_val = max(1, int(len(unique_vids) * val_ratio))
    val_vids = set(unique_vids[:n_val])
    train_vids = set(unique_vids[n_val:])
    val_mask = np.isin(video_id, list(val_vids))
    train_mask = np.isin(video_id, list(train_vids))
    print(f"  train video: {len(train_vids)}本, val video: {len(val_vids)}本")
    print(f"  train ペア: {int(train_mask.sum())}, val ペア: {int(val_mask.sum())}")
    return train_mask, val_mask


def _phase_label(puyo_count: np.ndarray) -> np.ndarray:
    """ぷよ合計数から位相ラベル (0=序盤, 1=中盤, 2=終盤) を返す。

    3 分位数でカット: 下位 33% = 序盤, 中位 = 中盤, 上位 = 終盤。
    """
    q1 = np.percentile(puyo_count, 33)
    q2 = np.percentile(puyo_count, 67)
    labels = np.where(puyo_count <= q1, 0, np.where(puyo_count <= q2, 1, 2))
    return labels.astype(np.int32)


def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC を NumPy のみで計算する (sklearn 非依存)。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    # 陽性・陰性ペアの正答率 = AUC
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # ランダムサブサンプル (大規模対策: 最大 50000 ペア)
    n_max = 50000
    rng = np.random.default_rng(0)
    if len(pos) * len(neg) > n_max:
        pi = rng.choice(len(pos), min(len(pos), 224), replace=False)
        ni = rng.choice(len(neg), min(len(neg), 224), replace=False)
        pos, neg = pos[pi], neg[ni]
    wins = np.sum(pos[:, None] > neg[None, :])
    ties = np.sum(pos[:, None] == neg[None, :])
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def _train_one_epoch(
    model: SiameseBoardCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """1 エポック学習し、平均 loss を返す。"""
    model.train()
    total_loss = 0.0
    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        logit, _, _ = model(x1, x2)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / max(len(loader.dataset), 1)


def _evaluate(
    model: SiameseBoardCNN,
    dataset: BoardPairDataset,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[float, np.ndarray, np.ndarray]:
    """val セットで loss + logit を返す。

    Returns:
        val_loss, y_true (N,), y_score (N,)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0.0
    all_logit: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            logit, _, _ = model(x1, x2)
            loss = criterion(logit, y)
            total_loss += loss.item() * len(y)
            all_logit.append(logit.cpu().numpy())
            all_y.append(y.cpu().numpy())
    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_logit)
    val_loss = total_loss / max(len(dataset), 1)
    return val_loss, y_true, y_score


def train(
    pairs_path: Path,
    out_path: Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device_str: str = "auto",
) -> None:
    """学習のメインロジック。"""
    # デバイス選択
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[train_board_cnn] device: {device}")

    # データ読み込み
    board_1p, board_2p, won, video_id, puyo_count = _load_pairs(pairs_path)

    # train/val 分割 (video_id 単位)
    train_mask, val_mask = _split_video_ids(video_id)

    def _make_ds(mask: np.ndarray) -> BoardPairDataset:
        return BoardPairDataset(
            board_1p[mask], board_2p[mask], won[mask], puyo_count[mask],
        )

    train_ds = _make_ds(train_mask)
    val_ds = _make_ds(val_mask)
    val_puyo = puyo_count[val_mask]
    val_phases = _phase_label(val_puyo)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    # モデル・最適化
    model = SiameseBoardCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"[train_board_cnn] 学習開始: epochs={epochs}, batch={batch_size}, lr={lr}")
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_score = _evaluate(model, val_ds, device, batch_size)
        auc_all = _compute_auc(y_true, y_score)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  epoch {epoch:3d}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_AUC={auc_all:.4f}"
            )

    # 位相別 AUC
    print("\n[train_board_cnn] --- 位相別 val AUC ---")
    val_loss, y_true, y_score = _evaluate(model, val_ds, device, batch_size)
    auc_all = _compute_auc(y_true, y_score)
    print(f"  全体 AUC : {auc_all:.4f}  (n={len(y_true)})")
    for phase_idx, label in enumerate(PHASE_LABELS):
        mask = val_phases == phase_idx
        if mask.sum() < 10:
            print(f"  {label} AUC: サンプル不足 (n={mask.sum()})")
            continue
        auc = _compute_auc(y_true[mask], y_score[mask])
        print(f"  {label} AUC: {auc:.4f}  (n={int(mask.sum())})")

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    print(f"\n[train_board_cnn] モデル保存: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SiameseBoardCNN を board_pairs_fixed.npz で学習する"
    )
    parser.add_argument(
        "--pairs", type=Path,
        default=Path("data/indicators_v2/board_pairs_fixed.npz"),
        help="board_pairs_fixed.npz のパス",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"エポック数 (既定: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"バッチサイズ (既定: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"学習率 (既定: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("models/board_cnn_s3.pt"),
        help="出力モデルパス (既定: models/board_cnn_s3.pt)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="cuda / cpu / auto (既定: auto)",
    )
    args = parser.parse_args()

    train(
        pairs_path=args.pairs,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
