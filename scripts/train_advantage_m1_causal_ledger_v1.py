"""同一canonical行でM0とM1 causal ledger branchを固定OOF比較する。"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import train_advantage_m0_current_cnn_v1 as base
from src.advantage_m0_current_cnn_v1 import (
    AdvantageM0CurrentCNNV2,
    equal_game_weighted_bce,
)
from src.advantage_m1_causal_ledger_v1 import AdvantageM1CausalLedgerCNNV1
from src.event_provisional_oof_v1 import FoldPlan, fixed_fold_plan


TRAINING_VERSION = "advantage-m1-fixed-oof-training/v1"
VARIANTS = (
    "m0", "m1_values", "m1_masks", "m1_values_and_masks", "m1_random_control",
)
DEFAULT_SEEDS = (20260904, 20260905, 20260906)
DEFAULT_EPOCHS = 12
DEFAULT_PATIENCE = 3
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
REPO_ROOT = Path(__file__).resolve().parents[1]


class AdvantageM1TrainingError(RuntimeError):
    """M0/M1比較契約または成果物が不正。"""


@dataclass(frozen=True, slots=True)
class CanonicalSamples:
    boards: np.ndarray
    queues: np.ndarray
    ledger_values: np.ndarray
    ledger_availability: np.ndarray
    labels: np.ndarray
    source_groups: np.ndarray
    game_keys: np.ndarray
    weights: np.ndarray
    folds: np.ndarray

    def subset(self, indices: np.ndarray) -> "CanonicalSamples":
        return CanonicalSamples(*(value[indices] for value in (
            self.boards, self.queues, self.ledger_values,
            self.ledger_availability, self.labels, self.source_groups,
            self.game_keys, self.weights, self.folds,
        )))


class CanonicalDataset(Dataset):
    """同じcanonical行をM0/M1へ供給する決定論的dataset。"""

    def __init__(self, samples: CanonicalSamples, *, augment: bool, seed: int) -> None:
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        boards = self.samples.boards[index]
        queues = self.samples.queues[index]
        values = self.samples.ledger_values[index]
        availability = self.samples.ledger_availability[index]
        label = float(self.samples.labels[index])
        if self.augment and self.rng.random() < 0.5:
            boards, queues = np.ascontiguousarray(boards[::-1]), np.ascontiguousarray(queues[::-1])
            values = np.ascontiguousarray(values[::-1])
            availability, label = np.ascontiguousarray(availability[::-1]), 1.0 - label
        return _tensor_row(boards, queues, values, availability, label, self.samples.weights[index])


def _tensor_row(
    boards: np.ndarray, queues: np.ndarray, values: np.ndarray,
    availability: np.ndarray, label: float, weight: float,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(boards.astype(np.int64, copy=False)),
        torch.from_numpy(queues.astype(np.int64, copy=False)),
        torch.from_numpy(values.astype(np.float32, copy=False)),
        torch.from_numpy(availability.astype(np.float32, copy=False)),
        torch.tensor(label, dtype=torch.float32),
        torch.tensor(weight, dtype=torch.float32),
    )


def load_canonical_dataset(root: Path) -> tuple[CanonicalSamples, dict[str, Any]]:
    """排他的完了hashを検査して46動画datasetを読む。"""

    manifest_path, data_path = root / "manifest.json", root / "dataset.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads((root / "COMPLETE").read_text(encoding="utf-8"))
    if complete.get("manifest_sha256") != base.file_sha256(manifest_path):
        raise AdvantageM1TrainingError("dataset manifest hashが一致しません")
    if complete.get("dataset_sha256") != base.file_sha256(data_path):
        raise AdvantageM1TrainingError("dataset NPZ hashが一致しません")
    with np.load(data_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    samples = _samples_from_arrays(arrays, manifest)
    _validate_samples(samples, manifest)
    return samples, manifest


def _samples_from_arrays(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any],
) -> CanonicalSamples:
    groups = np.asarray(arrays["source_groups"])
    folds = np.asarray([_fold_map(manifest)[str(value)] for value in groups], dtype=np.int8)
    return CanonicalSamples(
        boards=np.asarray(arrays["boards"], dtype=np.int8),
        queues=np.asarray(arrays["queues"], dtype=np.int8),
        ledger_values=np.asarray(arrays["ledger_values"], dtype=np.float32),
        ledger_availability=np.asarray(arrays["ledger_availability"], dtype=np.float32),
        labels=np.asarray(arrays["labels"], dtype=np.float32),
        source_groups=groups, game_keys=np.asarray(arrays["game_keys"]),
        weights=np.asarray(arrays["weights"], dtype=np.float32), folds=folds,
    )


def _fold_map(manifest: Mapping[str, Any]) -> dict[str, int]:
    groups = tuple(str(value) for value in manifest.get("source_group_ids", ()))
    sources = tuple(manifest.get("sources", ()))
    if len(groups) != len(sources) or len(set(groups)) != len(groups):
        raise AdvantageM1TrainingError("source group/fold台帳が不正です")
    result = {group: int(row["partition_fold"]) for group, row in zip(groups, sources, strict=True)}
    if set(result.values()) != set(range(1, 7)):
        raise AdvantageM1TrainingError("固定fold 1〜6が揃っていません")
    return result


def _validate_samples(samples: CanonicalSamples, manifest: Mapping[str, Any]) -> None:
    count = len(samples.labels)
    arrays = (
        samples.boards, samples.queues, samples.ledger_values,
        samples.ledger_availability, samples.source_groups,
        samples.game_keys, samples.weights, samples.folds,
    )
    if count == 0 or any(len(value) != count for value in arrays):
        raise AdvantageM1TrainingError("dataset行数が不正です")
    if samples.boards.shape[1:] != (2, 13, 6) or samples.queues.shape[1:] != (2, 4):
        raise AdvantageM1TrainingError("盤面またはqueue shapeが不正です")
    if samples.ledger_values.shape[1:] != (2, 17):
        raise AdvantageM1TrainingError("ledger value shapeが不正です")
    if samples.ledger_availability.shape[1:] != (2, 17, 5):
        raise AdvantageM1TrainingError("ledger availability shapeが不正です")
    if not np.isfinite(samples.weights).all() or np.any(samples.weights <= 0):
        raise AdvantageM1TrainingError("weightは有限な正値必須です")
    if int(manifest.get("state_count", -1)) != count:
        raise AdvantageM1TrainingError("manifest state_countが一致しません")


def randomized_ledger(samples: CanonicalSamples, seed: int) -> CanonicalSamples:
    """分割内でledger行だけを並べ替えたmatched nullを返す。"""

    if len(samples.labels) < 2:
        raise AdvantageM1TrainingError("random controlには2行以上必要です")
    permutation = np.random.default_rng(seed).permutation(len(samples.labels))
    if np.array_equal(permutation, np.arange(len(permutation))):
        permutation = np.roll(permutation, 1)
    return replace(
        samples,
        ledger_values=np.ascontiguousarray(samples.ledger_values[permutation]),
        ledger_availability=np.ascontiguousarray(samples.ledger_availability[permutation]),
    )


def _model(variant: str, device: torch.device) -> torch.nn.Module:
    if variant == "m0":
        return AdvantageM0CurrentCNNV2().to(device)
    modes = {
        "m1_values": "values", "m1_masks": "masks",
        "m1_values_and_masks": "values_and_masks",
        "m1_random_control": "values_and_masks",
    }
    if variant not in modes:
        raise AdvantageM1TrainingError(f"未対応variantです: {variant}")
    return AdvantageM1CausalLedgerCNNV1(modes[variant]).to(device)


def _forward(model: torch.nn.Module, batch: tuple[torch.Tensor, ...], variant: str) -> Any:
    boards, queues, values, availability = batch[:4]
    if variant == "m0":
        return model(boards, queues)
    return model(boards, queues, values, availability)


def _train_epoch(
    model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    device: torch.device, variant: str,
) -> float:
    model.train()
    total, denominator = 0.0, 0.0
    for batch in loader:
        moved = tuple(value.to(device) for value in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = equal_game_weighted_bce(_forward(model, moved, variant), moved[4], moved[5])
        loss.backward()
        optimizer.step()
        total += float((loss.detach() * moved[5].sum()).cpu())
        denominator += float(moved[5].sum().cpu())
    return total / denominator


def _predict(
    model: torch.nn.Module, samples: CanonicalSamples, batch_size: int,
    device: torch.device, variant: str,
) -> np.ndarray:
    loader = DataLoader(CanonicalDataset(samples, augment=False, seed=0), batch_size=batch_size)
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = tuple(value.to(device) for value in batch)
            output.append(_forward(model, moved, variant).raw_probability.cpu().numpy())
    return np.concatenate(output)


def _fit_fold(
    samples: CanonicalSamples, plan: FoldPlan, variant: str, seed: int,
    args: argparse.Namespace, device: torch.device,
) -> tuple[torch.nn.Module, float, int, dict[str, float], dict[str, float]]:
    train = samples.subset(np.isin(samples.folds, plan.train_folds))
    tune = samples.subset(samples.folds == plan.tune_fold)
    if variant == "m1_random_control":
        train, tune = randomized_ledger(train, seed + 11), randomized_ledger(tune, seed + 12)
    base._set_seed(seed)
    model = _model(variant, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        CanonicalDataset(train, augment=True, seed=seed),
        batch_size=args.batch_size, shuffle=True,
    )
    best_state, best_epoch, best_loss, stale = None, 0, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, loader, optimizer, device, variant)
        tune_probability = _predict(model, tune, args.batch_size, device, variant)
        tune_loss = metrics(tune, tune_probability)["game_equal_log_loss"]
        print(f"[{variant}] seed={seed} eval={plan.eval_fold} epoch={epoch} train={train_loss:.6f} tune={tune_loss:.6f}", flush=True)
        if tune_loss < best_loss:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch, best_loss, stale = epoch, tune_loss, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    return _selected_model(model, best_state, tune, variant, args, device, best_epoch)


def _selected_model(
    model: torch.nn.Module, state: Mapping[str, torch.Tensor] | None,
    tune: CanonicalSamples, variant: str, args: argparse.Namespace,
    device: torch.device, epoch: int,
) -> tuple[torch.nn.Module, float, int, dict[str, float], dict[str, float]]:
    if state is None:
        raise AdvantageM1TrainingError("best checkpointを選べませんでした")
    model.load_state_dict(state)
    raw = _predict(model, tune, args.batch_size, device, variant)
    slope = symmetric_calibration_slope(tune, raw)
    return model, slope, epoch, metrics(tune, raw), metrics(tune, calibrate(raw, slope))


def symmetric_calibration_slope(samples: CanonicalSamples, probability: np.ndarray) -> float:
    """切片0の対称Platt slopeを一次元黄金分割でfitする。"""

    logits = _logits(probability)
    left, right = 0.05, 10.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    first, second = right - ratio * (right - left), left + ratio * (right - left)
    for _ in range(64):
        if _slope_loss(samples, logits, first) <= _slope_loss(samples, logits, second):
            right, second = second, first
            first = right - ratio * (right - left)
        else:
            left, first = first, second
            second = left + ratio * (right - left)
    return float((left + right) / 2.0)


def _slope_loss(samples: CanonicalSamples, logits: np.ndarray, slope: float) -> float:
    probability = 1.0 / (1.0 + np.exp(-np.clip(slope * logits, -30.0, 30.0)))
    return metrics(samples, probability)["game_equal_log_loss"]


def calibrate(probability: np.ndarray, slope: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(slope * _logits(probability), -30.0, 30.0)))


def _logits(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def metrics(samples: CanonicalSamples, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    labels, weights = samples.labels.astype(np.float64), samples.weights.astype(np.float64)
    losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return {
        "game_equal_log_loss": float(np.average(losses, weights=weights)),
        "game_equal_brier": float(np.average((clipped - labels) ** 2, weights=weights)),
        "auc": base._auc(labels, clipped),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    samples, dataset_manifest = load_canonical_dataset(args.dataset_root)
    plans = {plan.eval_fold: plan for plan in fixed_fold_plan(samples.folds)}
    args.output_root.mkdir(parents=True, exist_ok=False)
    plan_receipt = _plan_receipt(args, samples, dataset_manifest)
    base._write_json_exclusive(args.output_root / "PLAN.json", plan_receipt)
    results: dict[str, Any] = {}
    for seed in args.seeds:
        for variant in args.variants:
            results[f"{variant}__seed_{seed}"] = _run_oof(
                samples, plans, variant, seed, args, base._device(args.device),
            )
    return _write_results(args, samples, dataset_manifest, plan_receipt, results)


def _run_oof(
    samples: CanonicalSamples, plans: Mapping[int, FoldPlan], variant: str,
    seed: int, args: argparse.Namespace, device: torch.device,
) -> dict[str, Any]:
    raw = np.full(len(samples.labels), np.nan, dtype=np.float32)
    calibrated = np.full(len(samples.labels), np.nan, dtype=np.float32)
    records = []
    for fold in args.folds:
        plan = plans[fold]
        model, slope, epoch, tune_raw, tune_cal = _fit_fold(
            samples, plan, variant, seed + fold * 100, args, device,
        )
        evaluation = samples.subset(samples.folds == fold)
        if variant == "m1_random_control":
            evaluation = randomized_ledger(evaluation, seed + fold * 100 + 13)
        prediction = _predict(model, evaluation, args.batch_size, device, variant)
        selected = np.flatnonzero(samples.folds == fold)
        raw[selected], calibrated[selected] = prediction, calibrate(prediction, slope)
        model_name = f"{variant}__seed_{seed}__fold_{fold}.pt"
        torch.save({"variant": variant, "seed": seed, "fold": fold,
                    "state_dict": model.state_dict()}, args.output_root / model_name)
        records.append(_fold_record(plan, epoch, slope, tune_raw, tune_cal, evaluation, prediction, model_name, args))
    mask = np.isfinite(raw)
    selected_samples = samples.subset(mask)
    return {
        "raw": raw, "calibrated": calibrated, "records": records,
        "metrics_raw": metrics(selected_samples, raw[mask]),
        "metrics_calibrated": metrics(selected_samples, calibrated[mask]),
    }


def _fold_record(
    plan: FoldPlan, epoch: int, slope: float,
    tune_raw: Mapping[str, float], tune_cal: Mapping[str, float],
    evaluation: CanonicalSamples, prediction: np.ndarray,
    model_name: str, args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "eval_fold": plan.eval_fold, "tune_fold": plan.tune_fold,
        "train_folds": list(plan.train_folds), "best_epoch": epoch,
        "symmetric_platt_slope": slope, "tune_raw": dict(tune_raw),
        "tune_calibrated": dict(tune_cal), "eval_raw": metrics(evaluation, prediction),
        "eval_calibrated": metrics(evaluation, calibrate(prediction, slope)),
        "model": model_name,
        "model_sha256": base.file_sha256(args.output_root / model_name),
    }


def _plan_receipt(
    args: argparse.Namespace, samples: CanonicalSamples,
    dataset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": "advantage-m1-fixed-oof-plan/v1", "not_production": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_sha256": dataset_manifest["dataset"]["sha256"],
        "state_count": len(samples.labels), "game_count": len(np.unique(samples.game_keys)),
        "source_count": len(np.unique(samples.source_groups)),
        "variants": list(args.variants), "seeds": list(args.seeds), "folds": list(args.folds),
        "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "fixed_outer_folds": 6,
        "inner_tune_policy": "eval k, tune k+1, train remaining four",
        "symmetric_calibration_intercept": 0.0,
        "code_sha256": {
            "trainer": base.file_sha256(Path(__file__)),
            "m0": base.file_sha256(REPO_ROOT / "src/advantage_m0_current_cnn_v1.py"),
            "m1": base.file_sha256(REPO_ROOT / "src/advantage_m1_causal_ledger_v1.py"),
            "production_config": base.file_sha256(REPO_ROOT / "src/production_config.py"),
        },
    }


def _write_results(
    args: argparse.Namespace, samples: CanonicalSamples,
    dataset_manifest: Mapping[str, Any], plan: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    prediction_payload: dict[str, np.ndarray] = {"labels": samples.labels, "folds": samples.folds}
    serializable: dict[str, Any] = {}
    for key, result in results.items():
        prediction_payload[f"{key}__raw"] = result["raw"]
        prediction_payload[f"{key}__calibrated"] = result["calibrated"]
        serializable[key] = {name: value for name, value in result.items() if name not in {"raw", "calibrated"}}
    prediction_path = args.output_root / "oof_predictions.npz"
    _save_npz_exclusive(prediction_path, prediction_payload)
    report = {
        "format_version": TRAINING_VERSION, "not_production": True,
        "plan": dict(plan), "dataset_source_count": dataset_manifest["source_count"],
        "results": serializable,
    }
    report_path = base._write_json_exclusive(args.output_root / "results.json", report)
    complete = {
        "format_version": "advantage-m1-fixed-oof-complete/v1",
        "results_sha256": base.file_sha256(report_path),
        "predictions_sha256": base.file_sha256(prediction_path),
    }
    base._write_json_exclusive(args.output_root / "COMPLETE", complete)
    print(json.dumps({key: value["metrics_calibrated"] for key, value in serializable.items()}, sort_keys=True), flush=True)
    return report


def _save_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _csv_strings(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("整数のカンマ区切りが必要です") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", type=_csv_strings, default=VARIANTS)
    parser.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--folds", type=_csv_ints, default=tuple(range(1, 7)))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if not args.variants or any(value not in VARIANTS for value in args.variants):
        parser.error(f"variantsは{VARIANTS}から指定してください")
    if not args.seeds or not args.folds or any(value not in range(1, 7) for value in args.folds):
        parser.error("seedsは非空、foldsは1..6の非空指定が必要です")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    train(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
