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

## 過学習対策 CLI (追加)
    --patience 5               早期終了 patience (既定 5)
    --early-stop-metric auc    監視指標: auc または loss (既定 auc)
    --weight-decay 1e-4        AdamW weight decay (既定 1e-4)
    --seed 0                   乱数シード (既定 0)
"""
from __future__ import annotations

import argparse
import copy
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
DEFAULT_PATIENCE: int = 5       # 早期終了 patience
DEFAULT_WEIGHT_DECAY: float = 1e-4  # AdamW weight decay
DEFAULT_SEED: int = 0
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
    seed: int = DEFAULT_SEED,
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


def _print_phase_auc(
    model: SiameseBoardCNN,
    val_ds: BoardPairDataset,
    val_phases: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> None:
    """ベストモデルで位相別 val AUC を計算・出力する。

    train() から切り出した補助関数 (1 関数 50 行制約)。
    """
    print("\n[train_board_cnn] --- 位相別 val AUC (ベストエポック) ---")
    val_loss, y_true, y_score = _evaluate(model, val_ds, device, batch_size)
    auc_all = _compute_auc(y_true, y_score)
    print(f"  全体 AUC : {auc_all:.4f}  (n={len(y_true)}, val_loss={val_loss:.4f})")
    for phase_idx, label in enumerate(PHASE_LABELS):
        mask = val_phases == phase_idx
        if mask.sum() < 10:
            print(f"  {label} AUC: サンプル不足 (n={mask.sum()})")
            continue
        auc = _compute_auc(y_true[mask], y_score[mask])
        print(f"  {label} AUC: {auc:.4f}  (n={int(mask.sum())})")


def _is_improved(
    metric: str,
    auc: float,
    val_loss: float,
    best_score: float,
) -> tuple[bool, float]:
    """改善判定を行い (improved, 新ベストスコア) を返す補助関数。

    metric == "auc"  → val AUC 最大化。
    metric == "loss" → val loss 最小化。
    """
    if metric == "auc":
        if auc > best_score:
            return True, auc
        return False, best_score
    else:  # "loss"
        if val_loss < best_score:
            return True, val_loss
        return False, best_score


def _run_training_loop(
    model: SiameseBoardCNN,
    train_loader: DataLoader,
    val_ds: BoardPairDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    early_stop_metric: str,
) -> tuple[SiameseBoardCNN, int]:
    """早期終了付き学習ループ。ベストエポックのモデルとエポック番号を返す。

    early_stop_metric: "auc" → val AUC 最大化、"loss" → val loss 最小化。
    patience=0 で早期終了を無効化。
    """
    init_score = -float("inf") if early_stop_metric == "auc" else float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_score: float = init_score
    best_epoch: int = 0
    no_improve: int = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_score = _evaluate(model, val_ds, device, batch_size)
        auc_all = _compute_auc(y_true, y_score)

        improved, best_score = _is_improved(early_stop_metric, auc_all, val_loss, best_score)
        if improved:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or improved:
            marker = " *best*" if improved else ""
            print(
                f"  epoch {epoch:3d}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_AUC={auc_all:.4f}{marker}"
            )

        if patience > 0 and no_improve >= patience:
            print(f"  [早期終了] {patience} エポック改善なし → epoch {epoch} で停止")
            break

    metric_name = "AUC" if early_stop_metric == "auc" else "loss"
    print(f"\n[train_board_cnn] ベストエポック: {best_epoch}  {metric_name}={best_score:.4f}")
    model.load_state_dict(best_state)
    return model, best_epoch


def _make_dataset(
    board_1p: np.ndarray,
    board_2p: np.ndarray,
    won: np.ndarray,
    puyo_count: np.ndarray,
    mask: np.ndarray,
) -> BoardPairDataset:
    """mask でフィルタして BoardPairDataset を生成する補助関数。"""
    return BoardPairDataset(
        board_1p[mask], board_2p[mask], won[mask], puyo_count[mask],
    )


def train(
    pairs_path: Path,
    out_path: Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device_str: str = "auto",
    patience: int = DEFAULT_PATIENCE,
    early_stop_metric: str = "auc",
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
) -> None:
    """学習のメインロジック。

    追加引数 (全てデフォルト付き、既存呼び出しへの後方互換維持):
        patience: 早期終了 patience エポック数 (0 で無効)。
        early_stop_metric: 監視指標 "auc" または "loss"。
        weight_decay: AdamW weight decay。
        seed: 乱数シード。
    """
    # 再現性: 乱数シードを固定
    torch.manual_seed(seed)
    np.random.seed(seed)

    # dropout 非対応の通知 (SiameseBoardCNN に dropout 引数なし)
    print(
        "[train_board_cnn] 注意: SiameseBoardCNN に dropout 引数がないため "
        "Dropout は適用しません。正則化は weight_decay と early stopping のみ。"
    )

    # デバイス選択
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto" else torch.device(device_str)
    )
    print(f"[train_board_cnn] device: {device}")

    # データ読み込み & 分割
    board_1p, board_2p, won, video_id, puyo_count = _load_pairs(pairs_path)
    train_mask, val_mask = _split_video_ids(video_id, seed=seed)
    train_ds = _make_dataset(board_1p, board_2p, won, puyo_count, train_mask)
    val_ds = _make_dataset(board_1p, board_2p, won, puyo_count, val_mask)
    val_phases = _phase_label(puyo_count[val_mask])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # モデル・最適化 (Adam → AdamW で weight_decay を適切に適用)
    model = SiameseBoardCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    print(
        f"[train_board_cnn] 学習開始: epochs={epochs}, batch={batch_size}, "
        f"lr={lr}, weight_decay={weight_decay}, "
        f"patience={patience}, early_stop_metric={early_stop_metric}, seed={seed}"
    )

    # 早期終了付き学習ループ → ベストエポックのモデルを返す
    model, best_epoch = _run_training_loop(
        model=model, train_loader=train_loader, val_ds=val_ds,
        optimizer=optimizer, criterion=criterion, device=device,
        epochs=epochs, batch_size=batch_size,
        patience=patience, early_stop_metric=early_stop_metric,
    )

    # ベストエポックモデルで位相別 AUC を出力
    _print_phase_auc(model, val_ds, val_phases, device, batch_size)

    # ベストエポックのモデルを保存 (最終エポックでなくベスト)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    print(f"[train_board_cnn] モデル保存 (ベストエポック {best_epoch}): {out_path}")


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
    # ---- 過学習対策オプション (全て後方互換デフォルト付き) ----
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help=f"早期終了 patience エポック数 (0 で無効、既定: {DEFAULT_PATIENCE})",
    )
    parser.add_argument(
        "--early-stop-metric", type=str, default="auc",
        choices=["auc", "loss"],
        help="早期終了の監視指標: auc (最大化) または loss (最小化) (既定: auc)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY,
        help=f"AdamW weight decay (既定: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"乱数シード (既定: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    train(
        pairs_path=args.pairs,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
