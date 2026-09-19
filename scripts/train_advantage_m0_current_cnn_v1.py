"""検証済み盤面50本で current-state CNN M0を学習する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.advantage_m0_current_cnn_v1 import (
    AdvantageM0CurrentCNNV1,
    M0_MODEL_VERSION,
    board_categories_from_raw,
    equal_game_weighted_bce,
    queue_categories_from_raw,
)
from src.board_quality import phantom_board_mask


TRAINING_VERSION = "advantage-m0-training/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_ROOT = Path("data/verify/board_training_source_index_dev97_2026-09-04_v5")
REVIEW_SOURCE_GROUP_ID = "c0BQoMJwwQU"
DEFAULT_SOURCE_COUNT = 50
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 3
DEFAULT_SEED = 20260904
MAX_STATES_PER_GAME = 96
QUEUE_FIELDS = ("next1_a", "next1_b", "dnext_a", "dnext_b")


class AdvantageM0TrainingError(RuntimeError):
    """M0学習入力または成果物が安全契約を満たさない。"""


@dataclass(frozen=True, slots=True)
class M0Samples:
    boards: np.ndarray
    queues: np.ndarray
    labels: np.ndarray
    source_groups: np.ndarray
    game_keys: np.ndarray
    weights: np.ndarray

    def subset(self, mask: np.ndarray) -> "M0Samples":
        return M0Samples(*(value[mask] for value in (
            self.boards, self.queues, self.labels, self.source_groups,
            self.game_keys, self.weights,
        )))


class M0Dataset(Dataset):
    """事前category化したM0サンプル。"""

    def __init__(
        self, samples: M0Samples, *, augment: bool, seed: int,
        mirror_augment: bool | None = None, side_swap_augment: bool | None = None,
    ) -> None:
        self.samples = samples
        self.mirror_augment = augment if mirror_augment is None else mirror_augment
        self.side_swap_augment = augment if side_swap_augment is None else side_swap_augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        boards = self.samples.boards[index]
        queues = self.samples.queues[index]
        label = float(self.samples.labels[index])
        if self.mirror_augment and self.rng.random() < 0.5:
            boards = np.ascontiguousarray(np.flip(boards, axis=2))
        if self.side_swap_augment and self.rng.random() < 0.5:
            boards = np.ascontiguousarray(boards[::-1])
            queues = np.ascontiguousarray(queues[::-1])
            label = 1.0 - label
        return (
            torch.from_numpy(boards.astype(np.int64, copy=False)),
            torch.from_numpy(queues.astype(np.int64, copy=False)),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(self.samples.weights[index], dtype=torch.float32),
        )


def load_index(index_root: Path) -> dict[str, Any]:
    """完成済みsource indexを読み、要約hashを検証する。"""

    summary_path = index_root / "SUMMARY.json"
    complete_path = index_root / "COMPLETE"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("summary_sha256") != file_sha256(summary_path):
        raise AdvantageM0TrainingError("source indexの完了hashが一致しません")
    if int(summary.get("source_count", 0)) != len(summary.get("sources", [])):
        raise AdvantageM0TrainingError("source indexの本数が一致しません")
    return summary


def select_sources(
    summary: Mapping[str, Any], source_count: int,
) -> tuple[dict[str, Any], ...]:
    """現行pilotを優先し、不足分をhistoricalから行順で補う。"""

    rows = [dict(row) for row in summary.get("sources", [])]
    pilot = [row for row in rows if row.get("dataset_role") == "pilot"]
    historical = [row for row in rows if row.get("dataset_role") == "historical_development"]
    selected = (pilot + historical)[:source_count]
    groups = [str(row.get("source_group_id", "")) for row in selected]
    if len(selected) != source_count or len(groups) != len(set(groups)):
        raise AdvantageM0TrainingError("選択sourceの本数または重複が不正です")
    if REVIEW_SOURCE_GROUP_ID in groups:
        raise AdvantageM0TrainingError("レビュー元映像が学習sourceへ混入しています")
    return tuple(selected)


def build_samples(sources: Sequence[Mapping[str, Any]]) -> M0Samples:
    """選択sourceを一件ずつ因果的にpair化して結合する。"""

    parts = [_source_samples(source) for source in sources]
    if not parts:
        raise AdvantageM0TrainingError("学習sourceが空です")
    arrays = [np.concatenate([getattr(part, name) for part in parts], axis=0)
              for name in ("boards", "queues", "labels", "source_groups", "game_keys")]
    weights = _equal_game_weights(arrays[4])
    return M0Samples(*arrays, weights)


def _source_samples(source: Mapping[str, Any]) -> M0Samples:
    path = Path(str(source["npz_path"]))
    if file_sha256(path) != source.get("npz_sha256"):
        raise AdvantageM0TrainingError(f"NPZ hash不一致: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in (
            "grids", "side", "frame_idx", "game_idx", "won", *QUEUE_FIELDS,
        )}
    return _causal_samples_from_arrays(arrays, str(source["source_group_id"]))


def _causal_samples_from_arrays(
    arrays: Mapping[str, np.ndarray], source_group: str,
) -> M0Samples:
    valid = ~phantom_board_mask(arrays["grids"])
    order = np.lexsort((np.arange(len(valid)), arrays["frame_idx"], arrays["game_idx"]))
    records: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    for game in np.unique(arrays["game_idx"][order]):
        indices = order[(arrays["game_idx"][order] == game) & valid[order]]
        records.extend(_game_records(arrays, indices, source_group, int(game)))
    records = _limit_records_by_game(records)
    if not records:
        raise AdvantageM0TrainingError(f"因果pairが作れません: {source_group}")
    boards = np.stack([row[0] for row in records]).astype(np.int8)
    queues = np.stack([row[1] for row in records]).astype(np.int8)
    labels = np.asarray([row[2] for row in records], dtype=np.float32)
    games = np.asarray([row[3] for row in records])
    groups = np.full(len(records), source_group)
    return M0Samples(boards, queues, labels, groups, games, np.ones(len(records)))


def _game_records(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray,
    source_group: str, game: int,
) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    latest: dict[str, int] = {}
    records: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    label = _game_label(arrays, indices)
    if label is None:
        return records
    for frame in np.unique(arrays["frame_idx"][indices]):
        at_frame = indices[arrays["frame_idx"][indices] == frame]
        for index in at_frame:
            side = str(arrays["side"][index])
            if side in {"1P", "2P"}:
                latest[side] = int(index)
        if set(latest) != {"1P", "2P"}:
            continue
        pair = (latest["1P"], latest["2P"])
        boards = np.stack([board_categories_from_raw(arrays["grids"][i]) for i in pair])
        queues = np.stack([_queue_categories(arrays, i) for i in pair])
        records.append((boards, queues, label, f"{source_group}:game={game}"))
    return records


def _game_label(arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> float | None:
    p1 = indices[arrays["side"][indices] == "1P"]
    values = np.unique(arrays["won"][p1][np.isfinite(arrays["won"][p1])])
    if len(values) == 0:
        return None
    if len(values) != 1 or float(values[0]) not in {0.0, 1.0}:
        raise AdvantageM0TrainingError("一試合内の1P勝敗labelが不整合です")
    return float(values[0])


def _limit_records_by_game(
    records: Sequence[tuple[np.ndarray, np.ndarray, float, str]],
) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray, float, str]]] = {}
    for row in records:
        grouped.setdefault(row[3], []).append(row)
    selected: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    for rows in grouped.values():
        if len(rows) <= MAX_STATES_PER_GAME:
            selected.extend(rows)
            continue
        positions = np.linspace(0, len(rows) - 1, MAX_STATES_PER_GAME, dtype=np.int64)
        selected.extend(rows[int(index)] for index in positions)
    return selected


def _queue_categories(arrays: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    values = [int(arrays[name][index]) for name in QUEUE_FIELDS]
    return queue_categories_from_raw(values)


def _equal_game_weights(game_keys: np.ndarray) -> np.ndarray:
    counts = Counter(str(value) for value in game_keys)
    game_count = len(counts)
    sample_count = len(game_keys)
    return np.asarray([
        sample_count / (game_count * counts[str(key)]) for key in game_keys
    ], dtype=np.float32)


def validation_groups(sources: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """source-group hash順の20%を検証用に固定する。"""

    groups = [str(row["source_group_id"]) for row in sources]
    ranked = sorted(groups, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    return frozenset(ranked[:max(1, math.ceil(len(ranked) * 0.2))])


def _train_epoch(
    model: AdvantageM0CurrentCNNV1, loader: DataLoader,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    model.train()
    total, denominator = 0.0, 0.0
    for boards, queues, labels, weights in loader:
        boards, queues = boards.to(device), queues.to(device)
        labels, weights = labels.to(device), weights.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = equal_game_weighted_bce(model(boards, queues), labels, weights)
        loss.backward()
        optimizer.step()
        total += float((loss.detach() * weights.sum()).cpu())
        denominator += float(weights.sum().cpu())
    return total / denominator


def _predict(
    model: AdvantageM0CurrentCNNV1, samples: M0Samples,
    batch_size: int, device: torch.device,
) -> np.ndarray:
    loader = DataLoader(M0Dataset(samples, augment=False, seed=0), batch_size=batch_size)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for boards, queues, _labels, _weights in loader:
            output = model(boards.to(device), queues.to(device))
            values.append(output.raw_probability.cpu().numpy())
    return np.concatenate(values)


def _metrics(samples: M0Samples, probability: np.ndarray) -> dict[str, float]:
    eps = 1e-7
    clipped = np.clip(probability.astype(np.float64), eps, 1.0 - eps)
    labels = samples.labels.astype(np.float64)
    weights = samples.weights.astype(np.float64)
    losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    brier = np.average((clipped - labels) ** 2, weights=weights)
    return {"game_equal_log_loss": float(np.average(losses, weights=weights)),
            "game_equal_brier": float(brier), "auc": _auc(labels, clipped)}


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1.0
    negative = labels == 0.0
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = int(positive.sum()), int(negative.sum())
    return float((ranks[positive].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _fit_with_validation(
    train: M0Samples, validation: M0Samples, args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, dict[str, float]]:
    model = AdvantageM0CurrentCNNV1().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loader = DataLoader(M0Dataset(train, augment=True, seed=args.seed),
                        batch_size=args.batch_size, shuffle=True)
    best_state, best_epoch, best_loss, stale = None, 0, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, loader, optimizer, device)
        metrics = _metrics(validation, _predict(model, validation, args.batch_size, device))
        print(f"[m0] epoch={epoch} train={train_loss:.6f} val={metrics['game_equal_log_loss']:.6f}", flush=True)
        if metrics["game_equal_log_loss"] < best_loss:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch, best_loss, stale = epoch, metrics["game_equal_log_loss"], 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise AdvantageM0TrainingError("best modelを選べませんでした")
    model.load_state_dict(best_state)
    return best_state, best_epoch, _metrics(
        validation, _predict(model, validation, args.batch_size, device),
    )


def _fit_all(
    samples: M0Samples, epochs: int, args: argparse.Namespace, device: torch.device,
) -> AdvantageM0CurrentCNNV1:
    torch.manual_seed(args.seed + 1)
    model = AdvantageM0CurrentCNNV1().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loader = DataLoader(M0Dataset(samples, augment=True, seed=args.seed + 1),
                        batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, loader, optimizer, device)
        print(f"[m0-final] epoch={epoch}/{epochs} loss={loss:.6f}", flush=True)
    return model


def train(args: argparse.Namespace) -> dict[str, Any]:
    """50本選択、検証、全50本fit、排他的成果物確定を行う。"""

    _set_seed(args.seed)
    summary = load_index(args.index_root)
    sources = select_sources(summary, args.source_count)
    samples = build_samples(sources)
    heldout = validation_groups(sources)
    val_mask = np.isin(samples.source_groups, list(heldout))
    device = _device(args.device)
    _, best_epoch, metrics = _fit_with_validation(
        samples.subset(~val_mask), samples.subset(val_mask), args, device,
    )
    model = _fit_all(samples, best_epoch, args, device)
    return _write_outputs(args, sources, samples, heldout, metrics, best_epoch, model)


def _write_outputs(
    args: argparse.Namespace, sources: Sequence[Mapping[str, Any]], samples: M0Samples,
    heldout: frozenset[str], metrics: Mapping[str, float], best_epoch: int,
    model: AdvantageM0CurrentCNNV1,
) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=False)
    model_path = args.output_root / "model.pt"
    torch.save({"model_version": M0_MODEL_VERSION, "state_dict": model.state_dict()}, model_path)
    manifest = _manifest(args, sources, samples, heldout, metrics, best_epoch, model_path)
    manifest_path = _write_json_exclusive(args.output_root / "manifest.json", manifest)
    complete = {"format_version": "advantage-m0-training-complete/v1",
                "manifest_sha256": file_sha256(manifest_path),
                "model_sha256": file_sha256(model_path)}
    _write_json_exclusive(args.output_root / "COMPLETE", complete)
    print(json.dumps(manifest["result"], ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def _manifest(
    args: argparse.Namespace, sources: Sequence[Mapping[str, Any]], samples: M0Samples,
    heldout: frozenset[str], metrics: Mapping[str, float], best_epoch: int,
    model_path: Path,
) -> dict[str, Any]:
    tier_counts = Counter(str(row["tier"]) for row in sources)
    role_counts = Counter(str(row["dataset_role"]) for row in sources)
    return {
        "format_version": TRAINING_VERSION, "model_version": M0_MODEL_VERSION,
        "not_production": True, "source_count": len(sources),
        "source_selection": "all_available_pilot_then_historical_in_verified_index_order",
        "review_source_group_id": REVIEW_SOURCE_GROUP_ID, "review_source_excluded": True,
        "source_index": str(args.index_root.resolve()),
        "source_index_summary_sha256": file_sha256(args.index_root / "SUMMARY.json"),
        "code_sha256": {
            "model": file_sha256(REPO_ROOT / "src/advantage_m0_current_cnn_v1.py"),
            "trainer": file_sha256(REPO_ROOT / "scripts/train_advantage_m0_current_cnn_v1.py"),
        },
        "target_ids": [str(row["target_id"]) for row in sources],
        "source_group_ids": [str(row["source_group_id"]) for row in sources],
        "source_npz_sha256": {str(row["target_id"]): row["npz_sha256"] for row in sources},
        "tier_counts": dict(sorted(tier_counts.items())), "role_counts": dict(role_counts),
        "validation_source_group_ids": sorted(heldout),
        "sample_policy": {"causal_latest_per_side": True, "future_pairing": False,
                          "phantom_board_filter": True,
                          "max_states_per_game": MAX_STATES_PER_GAME,
                          "equal_total_weight_per_game": True},
        "training": {"seed": args.seed, "requested_epochs": args.epochs,
                     "best_epoch": best_epoch, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate, "weight_decay": args.weight_decay},
        "result": {"sample_count": len(samples.labels),
                   "game_count": len(np.unique(samples.game_keys)), **dict(metrics)},
        "model": {"name": model_path.name, "sha256": file_sha256(model_path)},
    }


def _set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=DEFAULT_SOURCE_COUNT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    train(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
