"""Phase H4.1: Deep Tabular MLP with Multi-task Heads (end-to-end prototype).

End-to-end CNN base + 補助 indicator の prototype として、Deep MLP 多 head 学習を実装。
真の board CNN (raw 6×13 input) は H4.2 で予定 (データ不足のため H4.1 では tabular)。

入力:
    --csv data/training/match_features_phase_h2_quick_phased.csv (4,387 行 / 285 列)

モデル:
    DeepTabularMLP
      - Backbone: Linear(n_in→256) → ReLU → Dropout → Linear(256→128) → ReLU →
                  Dropout → Linear(128→64) → ReLU
      - Heads:
          1. winrate_head: 64 → 64 → 1 (sigmoid) [primary, BCE]
          2. indicator_head: 64 → 64 → 45 (回帰) [aux, MSE]
          3. phase_head: 64 → 64 → 5 (softmax) [aux, CE]
      - Multi-task loss: α·BCE + β·MSE + γ·CE

3 つの feature 構成で比較:
    full_280 / top_50 (S+A) / top_20 (S)

評価:
    - video_holdout (3 動画 test): primary winrate accuracy
    - LOOV (11 動画): primary winrate phase mean (start/mid/end)
    - Indicator regression MSE (auxiliary)
    - Phase classification accuracy (auxiliary)

注意: 1 関数 50 行以内、マジックナンバー禁止、torch + numpy seed 固定。
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
PHASE_TO_IDX = {p: i for i, p in enumerate(PHASES)}

# モデル構造定数
BACKBONE_DIMS = (256, 128, 64)  # Linear stack
HEAD_HIDDEN_DIM = 64            # 各 head の中間層
DROPOUT_RATE = 0.3
EMBEDDING_DIM = BACKBONE_DIMS[-1]  # = 64

# 学習ハイパーパラメータ
N_EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# Multi-task loss 重み
ALPHA_WINRATE = 1.0
BETA_INDICATOR = 0.3
GAMMA_PHASE = 0.1

# 評価関連
N_TEST_VIDEOS = 3
RANDOM_SEED = 0
EARLY_STOP_PATIENCE = 8

# Top features 設定 (H3 ablation 結果から)
TOP_20_FEATURES = (
    "incoming_ojama_pressure__static",
    "form_gtr__hist_max",
    "second_chain_potential__hist_max",
    "shape_score__hist_max",
    "maximum_fire_power__hist_min",
    "upper_board_density__hist_max",
    "next_acceptance__hist_mean",
    "gtr_orientation__hist_mean",
    "base_flatness__hist_min",
    "gtr_orientation__hist_min",
    "self_chain_power_x_opp_chain_power",
    "key_flexibility__hist_mean",
    "shape_score__hist_mean",
    "next_acceptance__static",
    "structure_solidity__hist_mean",
    "harassment_readiness__hist_mean",
    "color_variance__static",
    "sub_chain_quality__hist_mean",
    "upper_board_density__hist_mean",
    "color_variance__hist_mean",
)
# Tier A は H3 ablation 結果から (rank 20..49)
TOP_50_TIER_A = (
    "structure_solidity__hist_min",
    "base_flatness__static",
    "planning_entropy__hist_min",
    "planning_entropy__static",
    "color_variance__hist_max",
    "planning_entropy__hist_max",
    "harassment_resistance__hist_min",
    "mid_game_response_capacity__hist_max",
    "sub_chain_quality__hist_max",
    "field_efficiency__hist_max",
    "next_acceptance__accel",
    "sub_chain_independence__accel",
    "touching_density__delta",
    "form_gtr__accel",
    "next_acceptance__hist_max",
    "gtr_orientation__hist_max",
    "mid_game_response_capacity__accel",
    "main_chain_maturity__static",
    "field_efficiency__accel",
    "form_llr__accel",
    "incoming_ojama_pressure__accel",
    "key_flexibility__hist_min",
    "structure_solidity__static",
    "opponent_chain_threat__hist_max",
    "upper_board_density__accel",
    "chain_timing_pressure__accel",
    "maximum_fire_power__hist_mean",
    "ojama_defense_capacity__delta",
    "mid_game_response_capacity__delta",
    "high_connection_count__accel",
)


def setup_logger(log_path: Path) -> logging.Logger:
    """ファイル + stdout に出力する logger を作成する."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_h4_1")
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


def load_h2_csv(path: Path) -> dict[str, Any]:
    """H2 csv を読み込み、自動で feature columns を抽出する."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    feat_cols = [c for c in fieldnames if c not in META_COLS]
    n, d = len(rows), len(feat_cols)
    X = np.zeros((n, d), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    video_ids: list[str] = []
    time_phases: list[str] = []
    for i, r in enumerate(rows):
        for j, c in enumerate(feat_cols):
            X[i, j] = float(r.get(c, 0.0) or 0.0)
        y[i] = int(r["label"])
        video_ids.append(r["video_id"])
        time_phases.append(r.get("time_phase", "midpoint"))
    return {
        "X": X, "y": y,
        "video_ids": np.array(video_ids),
        "time_phases": np.array(time_phases),
        "feat_cols": feat_cols,
        "n": n, "d": d,
    }


def find_indicator_indices(feat_cols: list[str]) -> list[int]:
    """feat_cols のうち '__static' で終わる列の index を返す (auxiliary 回帰 target)."""
    return [i for i, c in enumerate(feat_cols) if c.endswith("__static")]


def build_feature_subset(feat_cols: list[str], names: tuple[str, ...]) -> list[int]:
    """指定 feature 名から index list を構築 (CSV に存在する順を保つ)."""
    name_set = set(names)
    return [i for i, c in enumerate(feat_cols) if c in name_set]


def video_holdout_split(
    video_ids: np.ndarray, n_test: int, seed: int
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
    """train で fit して train/test を z-score 化する."""
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    return ((X_tr - mu) / sd).astype(np.float32), ((X_te - mu) / sd).astype(np.float32)


# ============================
# モデル定義
# ============================
class DeepTabularMLP(nn.Module):
    """Backbone MLP + 3 head (winrate / indicator / phase) のマルチタスクモデル."""

    def __init__(
        self,
        n_features: int,
        n_indicators: int,
        n_phases: int,
        backbone_dims: tuple[int, ...] = BACKBONE_DIMS,
        head_hidden: int = HEAD_HIDDEN_DIM,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()
        self.backbone = self._build_backbone(n_features, backbone_dims, dropout)
        embed_dim = backbone_dims[-1]
        self.winrate_head = self._build_head(embed_dim, head_hidden, 1)
        self.indicator_head = self._build_head(embed_dim, head_hidden, n_indicators)
        self.phase_head = self._build_head(embed_dim, head_hidden, n_phases)

    @staticmethod
    def _build_backbone(
        n_in: int, dims: tuple[int, ...], dropout: float
    ) -> nn.Sequential:
        """Linear → ReLU → Dropout の積み重ねで backbone を構築する."""
        layers: list[nn.Module] = []
        prev = n_in
        for h in dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        return nn.Sequential(*layers)

    @staticmethod
    def _build_head(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
        """中間 1 層の head (Linear → ReLU → Linear)."""
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """戻り値: (winrate_logit, indicators_pred, phase_logits, embedding)."""
        embed = self.backbone(x)
        winrate_logit = self.winrate_head(embed).squeeze(-1)
        indicators = self.indicator_head(embed)
        phase_logits = self.phase_head(embed)
        return winrate_logit, indicators, phase_logits, embed


def multi_task_loss(
    winrate_logit: torch.Tensor, y_bin: torch.Tensor,
    indicators_pred: torch.Tensor, indicators_true: torch.Tensor,
    phase_logits: torch.Tensor, phase_true: torch.Tensor,
    alpha: float = ALPHA_WINRATE, beta: float = BETA_INDICATOR, gamma: float = GAMMA_PHASE,
) -> tuple[torch.Tensor, dict[str, float]]:
    """α·BCE + β·MSE + γ·CE の合成損失と各内訳を返す."""
    bce = F.binary_cross_entropy_with_logits(winrate_logit, y_bin.float())
    mse = F.mse_loss(indicators_pred, indicators_true)
    ce = F.cross_entropy(phase_logits, phase_true)
    total = alpha * bce + beta * mse + gamma * ce
    parts = {"bce": float(bce.item()), "mse": float(mse.item()), "ce": float(ce.item())}
    return total, parts


# ============================
# 学習 / 評価
# ============================
def make_loader(
    X: np.ndarray, y_bin: np.ndarray, ind: np.ndarray, phase: np.ndarray,
    batch_size: int, shuffle: bool,
) -> DataLoader:
    """numpy 配列から TensorDataset + DataLoader を作る."""
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y_bin.astype(np.float32)),
        torch.from_numpy(ind.astype(np.float32)),
        torch.from_numpy(phase.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: AdamW, device: torch.device,
) -> dict[str, float]:
    """1 epoch 分の学習。平均 loss 内訳を返す."""
    model.train()
    sums = {"total": 0.0, "bce": 0.0, "mse": 0.0, "ce": 0.0}
    n = 0
    for xb, yb, ib, pb in loader:
        xb, yb, ib, pb = xb.to(device), yb.to(device), ib.to(device), pb.to(device)
        optimizer.zero_grad()
        wl, ip, pl, _ = model(xb)
        loss, parts = multi_task_loss(wl, yb, ip, ib, pl, pb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        bs = xb.size(0)
        sums["total"] += float(loss.item()) * bs
        for k in ("bce", "mse", "ce"):
            sums[k] += parts[k] * bs
        n += bs
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> dict[str, float]:
    """検証データで loss + winrate accuracy + phase acc + indicator MSE を返す."""
    model.eval()
    sums = {"total": 0.0, "bce": 0.0, "mse": 0.0, "ce": 0.0}
    n = 0
    n_correct_w = 0
    n_correct_p = 0
    for xb, yb, ib, pb in loader:
        xb, yb, ib, pb = xb.to(device), yb.to(device), ib.to(device), pb.to(device)
        wl, ip, pl, _ = model(xb)
        loss, parts = multi_task_loss(wl, yb, ip, ib, pl, pb)
        bs = xb.size(0)
        sums["total"] += float(loss.item()) * bs
        for k in ("bce", "mse", "ce"):
            sums[k] += parts[k] * bs
        n += bs
        n_correct_w += int(((torch.sigmoid(wl) > 0.5).long() == yb.long()).sum().item())
        n_correct_p += int((pl.argmax(dim=1) == pb).sum().item())
    avg = {k: v / max(1, n) for k, v in sums.items()}
    avg["winrate_acc"] = n_correct_w / max(1, n)
    avg["phase_acc"] = n_correct_p / max(1, n)
    avg["indicator_mse"] = avg["mse"]
    return avg


def fit_model(
    X_tr: np.ndarray, y_tr_bin: np.ndarray, ind_tr: np.ndarray, ph_tr: np.ndarray,
    X_va: np.ndarray, y_va_bin: np.ndarray, ind_va: np.ndarray, ph_va: np.ndarray,
    n_features: int, n_indicators: int, n_phases: int,
    device: torch.device, logger: logging.Logger, tag: str,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, float]]:
    """1 セットの train/val で fit し、best val acc 時のモデル + log を返す."""
    model = DeepTabularMLP(n_features, n_indicators, n_phases).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    tr_loader = make_loader(X_tr, y_tr_bin, ind_tr, ph_tr, BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X_va, y_va_bin, ind_va, ph_va, BATCH_SIZE, shuffle=False)

    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_state = None
    best_metrics: dict[str, float] = {}
    no_improve = 0
    for epoch in range(N_EPOCHS):
        tr_log = train_one_epoch(model, tr_loader, optimizer, device)
        va_log = evaluate(model, va_loader, device)
        scheduler.step()
        history.append({"epoch": epoch, "train": tr_log, "val": va_log})
        logger.info(
            "[%s] epoch %02d/%d | tr_total=%.4f bce=%.4f mse=%.4f ce=%.4f | "
            "va_acc=%.4f va_phase_acc=%.4f va_mse=%.4f",
            tag, epoch + 1, N_EPOCHS, tr_log["total"], tr_log["bce"], tr_log["mse"],
            tr_log["ce"], va_log["winrate_acc"], va_log["phase_acc"], va_log["indicator_mse"],
        )
        if va_log["winrate_acc"] > best_val_acc:
            best_val_acc = va_log["winrate_acc"]
            best_metrics = dict(va_log)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                logger.info("[%s] early stop at epoch %d (no improve %d)",
                            tag, epoch + 1, no_improve)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_metrics


# ============================
# サブセット評価
# ============================
def evaluate_subset(
    ds: dict[str, Any], indices: list[int], cfg_name: str,
    device: torch.device, logger: logging.Logger,
) -> dict[str, Any]:
    """指定 feature index 部分集合で fit + video holdout + LOOV phase mean を測る."""
    feat_cols = ds["feat_cols"]
    sel_cols = [feat_cols[i] for i in indices]
    ind_idx_full = find_indicator_indices(feat_cols)
    n_indicators = len(ind_idx_full)

    X = ds["X"][:, indices]
    y = ds["y"]
    ind_targets = ds["X"][:, ind_idx_full]
    phase_int = np.array(
        [PHASE_TO_IDX.get(p, PHASE_TO_IDX["midpoint"]) for p in ds["time_phases"]],
        dtype=np.int64,
    )
    y_bin = (y > 0).astype(np.int64)

    tr_mask, te_mask = video_holdout_split(ds["video_ids"], N_TEST_VIDEOS, RANDOM_SEED)
    X_tr, X_te = standardize(X[tr_mask], X[te_mask])
    ind_tr, ind_te = standardize(ind_targets[tr_mask], ind_targets[te_mask])

    logger.info("[%s] vh fit: train=%d, test=%d, n_feat=%d",
                cfg_name, int(tr_mask.sum()), int(te_mask.sum()), len(indices))
    _, history, vh_metrics = fit_model(
        X_tr, y_bin[tr_mask], ind_tr, phase_int[tr_mask],
        X_te, y_bin[te_mask], ind_te, phase_int[te_mask],
        n_features=len(indices), n_indicators=n_indicators, n_phases=len(PHASES),
        device=device, logger=logger, tag=f"{cfg_name}/vh",
    )

    # LOOV phase mean
    loov_phase = run_loov_phase(
        ds, indices, ind_idx_full, phase_int, y_bin, n_indicators, device, logger, cfg_name,
    )

    return {
        "name": cfg_name,
        "n_feat": len(indices),
        "feature_cols_sample": sel_cols[:10],
        "vh_acc": float(vh_metrics.get("winrate_acc", 0.0)),
        "vh_phase_acc": float(vh_metrics.get("phase_acc", 0.0)),
        "vh_indicator_mse": float(vh_metrics.get("indicator_mse", 0.0)),
        "vh_total_loss": float(vh_metrics.get("total", 0.0)),
        "loov_phase": loov_phase,
        "loov_avg": float(np.mean([m["mean"] for m in loov_phase.values()])),
        "training_log_tail": history[-3:],
    }


def run_loov_phase(
    ds: dict[str, Any], indices: list[int], ind_idx_full: list[int],
    phase_int: np.ndarray, y_bin: np.ndarray, n_indicators: int,
    device: torch.device, logger: logging.Logger, cfg_name: str,
) -> dict[str, dict[str, float]]:
    """phase グループごとに LOOV winrate accuracy を返す."""
    out: dict[str, dict[str, float]] = {}
    for phase_name, phases in PHASE_GROUPS.items():
        phase_mask = np.isin(ds["time_phases"], phases)
        if phase_mask.sum() == 0:
            out[phase_name] = {"mean": 0.0, "std": 0.0, "n_videos": 0}
            continue
        accs = loov_winrate_accs(
            ds, indices, ind_idx_full, phase_int, y_bin, phase_mask,
            n_indicators, device, logger, cfg_name, phase_name,
        )
        if accs:
            out[phase_name] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "n_videos": len(accs),
            }
        else:
            out[phase_name] = {"mean": 0.0, "std": 0.0, "n_videos": 0}
        logger.info("[%s] LOOV phase=%s mean=%.4f n=%d",
                    cfg_name, phase_name, out[phase_name]["mean"], out[phase_name]["n_videos"])
    return out


def loov_winrate_accs(
    ds: dict[str, Any], indices: list[int], ind_idx_full: list[int],
    phase_int: np.ndarray, y_bin: np.ndarray, phase_mask: np.ndarray,
    n_indicators: int, device: torch.device, logger: logging.Logger,
    cfg_name: str, phase_name: str,
) -> list[float]:
    """phase 限定で LOOV を回し各 fold の test acc list を返す."""
    accs: list[float] = []
    X_full = ds["X"][:, indices]
    ind_full = ds["X"][:, ind_idx_full]
    uniq = np.unique(ds["video_ids"][phase_mask])
    for vid in uniq:
        te = (ds["video_ids"] == vid) & phase_mask
        tr = (ds["video_ids"] != vid) & phase_mask
        if te.sum() == 0 or tr.sum() == 0:
            continue
        X_tr_z, X_te_z = standardize(X_full[tr], X_full[te])
        ind_tr_z, ind_te_z = standardize(ind_full[tr], ind_full[te])
        try:
            _, _, m = fit_model(
                X_tr_z, y_bin[tr], ind_tr_z, phase_int[tr],
                X_te_z, y_bin[te], ind_te_z, phase_int[te],
                n_features=len(indices), n_indicators=n_indicators, n_phases=len(PHASES),
                device=device, logger=logger, tag=f"{cfg_name}/loov-{phase_name}-{vid}",
            )
            accs.append(float(m.get("winrate_acc", 0.0)))
        except Exception as e:
            logger.warning("LOOV vid=%s phase=%s skip: %s", vid, phase_name, e)
    return accs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=Path("logs/phase_h4_1_train.log"))
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

    ds = load_h2_csv(args.csv)
    logger.info("[load] n=%d, d=%d, videos=%d", ds["n"], ds["d"],
                len(np.unique(ds["video_ids"])))
    ind_idx_full = find_indicator_indices(ds["feat_cols"])
    logger.info("[load] auxiliary indicator (static) cols=%d", len(ind_idx_full))

    # === 3 構成 ===
    full_indices = list(range(ds["d"]))
    top20_indices = build_feature_subset(ds["feat_cols"], TOP_20_FEATURES)
    top50_indices = build_feature_subset(
        ds["feat_cols"], TOP_20_FEATURES + TOP_50_TIER_A
    )
    logger.info("[subsets] full=%d, top20=%d, top50=%d",
                len(full_indices), len(top20_indices), len(top50_indices))

    configs = [
        ("full_280", full_indices),
        ("top_50", top50_indices),
        ("top_20", top20_indices),
    ]
    results: list[dict[str, Any]] = []
    for name, idxs in configs:
        logger.info("=" * 60)
        logger.info("[config] start %s (#feat=%d)", name, len(idxs))
        r = evaluate_subset(ds, idxs, name, device, logger)
        results.append(r)
        logger.info("[config] %s vh_acc=%.4f loov_avg=%.4f mse=%.4f phase_acc=%.4f",
                    name, r["vh_acc"], r["loov_avg"],
                    r["vh_indicator_mse"], r["vh_phase_acc"])

    best = max(results, key=lambda r: r["vh_acc"] + r["loov_avg"])
    payload = {
        "configs": results,
        "best_config": best["name"],
        "best_vh_acc": best["vh_acc"],
        "best_loov_avg": best["loov_avg"],
        "embedding_dim": EMBEDDING_DIM,
        "model_meta": {
            "backbone_dims": list(BACKBONE_DIMS),
            "head_hidden_dim": HEAD_HIDDEN_DIM,
            "dropout": DROPOUT_RATE,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "alpha_winrate": ALPHA_WINRATE,
            "beta_indicator": BETA_INDICATOR,
            "gamma_phase": GAMMA_PHASE,
        },
        "baseline_lr_h3_top20_vh": 0.7567,
        "n_rows": ds["n"],
        "n_videos": int(len(np.unique(ds["video_ids"]))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("[save] %s", args.out)
    logger.info("[done] best config = %s (vh=%.4f, loov=%.4f)",
                best["name"], best["vh_acc"], best["loov_avg"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
