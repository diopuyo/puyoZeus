"""M2材料・空間8補助教師のauxなし/ありを同一foldで固定OOF比較する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import build_advantage_m2_dataset_v1 as builder
from scripts import train_advantage_m0_current_cnn_v1 as base
from scripts import train_advantage_m1_causal_ledger_v1 as legacy
from scripts import train_advantage_m1_frozen_residual_v1 as frozen
from scripts import train_advantage_m1_zero_counterfactual_v3 as m1_trainer
from src import advantage_m2_auxiliary_cnn_v1 as m2
from src.canonical_observation_v3 import AvailabilityState
from src.event_provisional_oof_v1 import FoldPlan, fixed_fold_plan


TRAINING_VERSION = "advantage-m2-auxiliary-fixed-oof-training/v1"
PLAN_VERSION = "advantage-m2-auxiliary-fixed-oof-plan/v1"
COMPLETE_VERSION = "advantage-m2-auxiliary-fixed-oof-complete/v1"
DEFAULT_COEFFICIENTS = (0.0, m2.M2_AUXILIARY_LOSS_COEFFICIENT)
DEFAULT_SEEDS = legacy.DEFAULT_SEEDS
DEFAULT_EPOCHS = legacy.DEFAULT_EPOCHS
DEFAULT_PATIENCE = legacy.DEFAULT_PATIENCE
DEFAULT_BATCH_SIZE = legacy.DEFAULT_BATCH_SIZE
DEFAULT_LEARNING_RATE = legacy.DEFAULT_LEARNING_RATE
DEFAULT_WEIGHT_DECAY = legacy.DEFAULT_WEIGHT_DECAY
REPO_ROOT = Path(__file__).resolve().parents[1]
_FAULT_INDEX = m2.AVAILABILITY_ORDER.index(AvailabilityState.INTEGRITY_FAULT)


class AdvantageM2TrainingError(RuntimeError):
    """M2 dataset、paired学習、OOF保存契約に違反した。"""


@dataclass(frozen=True, slots=True)
class M2SamplesV1:
    """親M1行へM2追加tensorを結合した固定学習集合。"""

    boards: np.ndarray
    queues: np.ndarray
    all_clear_values: np.ndarray
    all_clear_availability: np.ndarray
    ledger_values: np.ndarray
    ledger_availability: np.ndarray
    labels: np.ndarray
    source_groups: np.ndarray
    game_keys: np.ndarray
    weights: np.ndarray
    folds: np.ndarray
    state_ids: np.ndarray
    auxiliary_targets: np.ndarray
    auxiliary_mask: np.ndarray

    def subset(self, indices: np.ndarray) -> "M2SamplesV1":
        return M2SamplesV1(*(value[indices] for value in (
            self.boards, self.queues, self.all_clear_values,
            self.all_clear_availability, self.ledger_values,
            self.ledger_availability, self.labels, self.source_groups,
            self.game_keys, self.weights, self.folds, self.state_ids,
            self.auxiliary_targets, self.auxiliary_mask,
        )))


@dataclass(frozen=True, slots=True)
class PredictionV1:
    probability: np.ndarray
    auxiliary_mse_rows: np.ndarray


@dataclass(frozen=True, slots=True)
class SelectionV1:
    model: m2.AdvantageM2AuxiliaryCNNV1
    slope: float
    best_epoch: int
    initial_state_sha256: str
    tune_raw: Mapping[str, float]
    tune_calibrated: Mapping[str, float]
    tune_auxiliary_mse: float


@dataclass(slots=True)
class AccumulatorV1:
    raw: np.ndarray
    calibrated: np.ndarray
    auxiliary_mse_rows: np.ndarray
    records: list[Mapping[str, Any]]


class M2DatasetV1(Dataset):
    """side-swap時に全入力・補助教師・勝敗を同時反転する。"""

    def __init__(self, samples: M2SamplesV1, *, augment: bool, seed: int) -> None:
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pairs = [getattr(self.samples, name)[index] for name in (
            "boards", "queues", "all_clear_values", "all_clear_availability",
            "ledger_values", "ledger_availability", "auxiliary_targets",
            "auxiliary_mask",
        )]
        label = float(self.samples.labels[index])
        if self.augment and self.rng.random() < 0.5:
            pairs = [np.ascontiguousarray(value[::-1]) for value in pairs]
            label = 1.0 - label
        return _tensor_row(*pairs[:6], label, self.samples.weights[index], *pairs[6:])


def _tensor_row(
    boards: np.ndarray, queues: np.ndarray, all_clear_values: np.ndarray,
    all_clear_availability: np.ndarray, ledger_values: np.ndarray,
    ledger_availability: np.ndarray, label: float, weight: float,
    auxiliary_targets: np.ndarray, auxiliary_mask: np.ndarray,
) -> tuple[torch.Tensor, ...]:
    arrays = (
        boards.astype(np.int64, copy=False), queues.astype(np.int64, copy=False),
        all_clear_values.astype(np.float32, copy=False),
        all_clear_availability.astype(np.float32, copy=False),
        ledger_values.astype(np.float32, copy=False),
        ledger_availability.astype(np.float32, copy=False),
    )
    return (*map(torch.from_numpy, arrays), torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
            torch.from_numpy(auxiliary_targets.astype(np.float32, copy=False)),
            torch.from_numpy(auxiliary_mask.astype(np.bool_, copy=False)))


def load_m2_dataset(root: Path) -> tuple[M2SamplesV1, dict[str, Any]]:
    """M2と親M1の排他receiptを検査して同一行学習集合を読む。"""

    manifest_path, data_path = root / "manifest.json", root / "dataset.npz"
    manifest, complete = _load_receipts(root)
    _validate_dataset_receipts(manifest, complete, manifest_path, data_path)
    parent_root = Path(str(manifest["parent_dataset_root"]))
    parent_samples, parent_manifest = m1_trainer.load_canonical_dataset_v2(parent_root)
    _validate_parent_receipts(manifest, parent_root)
    try:
        with np.load(data_path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as error:
        raise AdvantageM2TrainingError("M2 dataset NPZを読めません") from error
    _validate_parent_arrays(arrays, parent_samples)
    samples = _samples_from_arrays(arrays, parent_samples)
    _validate_m2_samples(samples, manifest, arrays)
    return samples, manifest | {"parent_manifest": parent_manifest}


def _load_receipts(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        complete = json.loads((root / "COMPLETE").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM2TrainingError("M2 dataset receiptを読めません") from error
    if not isinstance(manifest, dict) or not isinstance(complete, dict):
        raise AdvantageM2TrainingError("M2 dataset receiptはobject必須です")
    return manifest, complete


def _validate_dataset_receipts(
    manifest: Mapping[str, Any], complete: Mapping[str, Any],
    manifest_path: Path, data_path: Path,
) -> None:
    if manifest.get("format_version") != builder.FORMAT_VERSION:
        raise AdvantageM2TrainingError("M2 dataset formatが不正です")
    if manifest.get("not_production") is not True:
        raise AdvantageM2TrainingError("M2 datasetはnot_production必須です")
    if complete.get("format_version") != builder.COMPLETE_VERSION:
        raise AdvantageM2TrainingError("M2 COMPLETE formatが不正です")
    if complete.get("manifest_sha256") != base.file_sha256(manifest_path):
        raise AdvantageM2TrainingError("M2 manifest hashが一致しません")
    if complete.get("dataset_sha256") != base.file_sha256(data_path):
        raise AdvantageM2TrainingError("M2 dataset hashが一致しません")


def _validate_parent_receipts(manifest: Mapping[str, Any], root: Path) -> None:
    names = (
        ("manifest.json", "parent_manifest_sha256"),
        ("COMPLETE", "parent_complete_sha256"),
        ("dataset.npz", "parent_dataset_sha256"),
    )
    for filename, field in names:
        if base.file_sha256(root / filename) != manifest.get(field):
            raise AdvantageM2TrainingError(f"親M1 {filename} hashが一致しません")


def _validate_parent_arrays(
    arrays: Mapping[str, np.ndarray], parent: m1_trainer.CanonicalSamplesV3,
) -> None:
    expected = set(builder._parent_arrays(parent)) | {
        "all_clear_values", "all_clear_availability",
        "auxiliary_targets", "auxiliary_mask",
    }
    if set(arrays) != expected:
        raise AdvantageM2TrainingError("M2 NPZ array schemaが一致しません")
    for name, value in builder._parent_arrays(parent).items():
        if not np.array_equal(arrays[name], value):
            raise AdvantageM2TrainingError(f"親M1 arrayが変化しています: {name}")


def _samples_from_arrays(
    arrays: Mapping[str, np.ndarray], parent: m1_trainer.CanonicalSamplesV3,
) -> M2SamplesV1:
    return M2SamplesV1(
        parent.boards, parent.queues,
        np.asarray(arrays["all_clear_values"], dtype=np.float32),
        np.asarray(arrays["all_clear_availability"], dtype=np.float32),
        parent.ledger_values, parent.ledger_availability, parent.labels,
        parent.source_groups, parent.game_keys, parent.weights, parent.folds,
        parent.state_ids, np.asarray(arrays["auxiliary_targets"], dtype=np.float32),
        np.asarray(arrays["auxiliary_mask"], dtype=np.bool_),
    )


def _validate_m2_samples(
    samples: M2SamplesV1, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
) -> None:
    count = len(samples.labels)
    if samples.all_clear_values.shape != (count, 2):
        raise AdvantageM2TrainingError("全消し値shapeが不正です")
    _validate_availability(samples.all_clear_availability, (count, 2, m2.AVAILABILITY_COUNT))
    aux_shape = (count, 2, len(m2.AUXILIARY_FEATURE_NAMES))
    if samples.auxiliary_targets.shape != aux_shape or samples.auxiliary_mask.shape != aux_shape:
        raise AdvantageM2TrainingError("補助教師shapeが不正です")
    if np.any(samples.all_clear_availability[..., _FAULT_INDEX] > 0.5):
        raise AdvantageM2TrainingError("全消しintegrity faultを学習できません")
    if (not np.isfinite(samples.all_clear_values).all()
            or not np.logical_or(
                samples.all_clear_values == 0.0, samples.all_clear_values == 1.0,
            ).all()):
        raise AdvantageM2TrainingError("全消し値は0/1必須です")
    _validate_auxiliary(samples.auxiliary_targets, samples.auxiliary_mask)
    receipt = manifest.get("m2_receipt", {})
    if receipt.get("m2_input_schema_version") != m2.M2_INPUT_SCHEMA_VERSION:
        raise AdvantageM2TrainingError("M2 input schema versionが不正です")
    for name, expected in receipt.get("array_sha256", {}).items():
        if name not in arrays or builder._array_sha256(arrays[name]) != expected:
            raise AdvantageM2TrainingError(f"M2 array receiptが不一致です: {name}")


def _validate_availability(value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.dtype != np.float32 or value.shape != shape:
        raise AdvantageM2TrainingError("availability shape/dtypeが不正です")
    if not np.logical_or(value == 0.0, value == 1.0).all():
        raise AdvantageM2TrainingError("availabilityは0/1必須です")
    if not np.array_equal(value.sum(axis=-1), np.ones(value.shape[:-1])):
        raise AdvantageM2TrainingError("availabilityはone-hot必須です")


def _validate_auxiliary(targets: np.ndarray, mask: np.ndarray) -> None:
    if targets.dtype != np.float32 or mask.dtype != np.bool_:
        raise AdvantageM2TrainingError("補助教師dtypeが不正です")
    selected = targets[mask]
    if not np.isfinite(selected).all() or np.any(selected < 0.0) or np.any(selected > 1.0):
        raise AdvantageM2TrainingError("補助教師は有限な0..1必須です")


def _loader(
    samples: M2SamplesV1, args: argparse.Namespace, *, augment: bool, seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        M2DatasetV1(samples, augment=augment, seed=seed),
        batch_size=args.batch_size, shuffle=augment, generator=generator,
    )


def _forward(
    model: m2.AdvantageM2AuxiliaryCNNV1, batch: tuple[torch.Tensor, ...],
) -> m2.AdvantageM2OutputV1:
    return model(*batch[:6])


def _train_epoch(
    model: m2.AdvantageM2AuxiliaryCNNV1, loader: DataLoader,
    optimizer: torch.optim.Optimizer, device: torch.device, coefficient: float,
) -> tuple[float, float, float]:
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    denominator = 0.0
    for batch in loader:
        moved = tuple(value.to(device) for value in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = m2.advantage_m2_weighted_loss(
            _forward(model, moved), moved[6], moved[7], moved[8], moved[9],
            auxiliary_coefficient=coefficient,
        )
        loss.total.backward()
        optimizer.step()
        weight = float(moved[7].sum().cpu())
        totals += weight * np.asarray([float(item.detach().cpu()) for item in (
            loss.total, loss.binary_cross_entropy, loss.auxiliary_mse,
        )])
        denominator += weight
    return tuple(float(value / denominator) for value in totals)


def _predict(
    model: m2.AdvantageM2AuxiliaryCNNV1, samples: M2SamplesV1,
    args: argparse.Namespace, device: torch.device,
) -> PredictionV1:
    probabilities: list[np.ndarray] = []
    auxiliary_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in _loader(samples, args, augment=False, seed=0):
            moved = tuple(value.to(device) for value in batch)
            output = _forward(model, moved)
            probabilities.append(output.raw_probability.cpu().numpy())
            auxiliary_rows.append(_auxiliary_mse_rows(
                output.auxiliary, moved[8], moved[9],
            ).cpu().numpy())
    return PredictionV1(np.concatenate(probabilities), np.concatenate(auxiliary_rows))


def _auxiliary_mse_rows(
    predicted: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    squared = torch.where(mask, (predicted - targets).square(), torch.zeros_like(predicted))
    denominator = mask.sum((1, 2)).clamp_min(1).to(squared.dtype)
    return squared.sum((1, 2)) / denominator


def _weighted_auxiliary_mse(samples: M2SamplesV1, rows: np.ndarray) -> float:
    if len(rows) != len(samples.weights) or not np.isfinite(rows).all():
        raise AdvantageM2TrainingError("補助MSE行が不正です")
    return float(np.average(rows.astype(np.float64), weights=samples.weights))


def _fit_fold(
    samples: M2SamplesV1, plan: FoldPlan, coefficient: float, seed: int,
    args: argparse.Namespace, device: torch.device,
) -> SelectionV1:
    train = samples.subset(np.isin(samples.folds, plan.train_folds))
    tune = samples.subset(samples.folds == plan.tune_fold)
    base._set_seed(seed)
    model = m2.AdvantageM2AuxiliaryCNNV1().to(device)
    initial_sha256 = frozen.model_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    best_state, best_epoch, best_loss, stale = None, 0, float("inf"), 0
    loader = _loader(train, args, augment=True, seed=seed)
    for epoch in range(1, args.epochs + 1):
        losses = _train_epoch(model, loader, optimizer, device, coefficient)
        prediction = _predict(model, tune, args, device)
        tune_loss = legacy.metrics(tune, prediction.probability)["game_equal_log_loss"]
        _print_epoch(coefficient, seed, plan.eval_fold, epoch, losses, tune_loss)
        if tune_loss < best_loss:
            best_state, best_epoch, best_loss, stale = frozen._cpu_state(model), epoch, tune_loss, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    return _restore_selection(
        model, best_state, best_epoch, initial_sha256, tune, args, device,
    )


def _print_epoch(
    coefficient: float, seed: int, fold: int, epoch: int,
    losses: tuple[float, float, float], tune_loss: float,
) -> None:
    print(
        f"[aux={coefficient:.1f}] seed={seed} eval={fold} epoch={epoch} "
        f"total={losses[0]:.6f} bce={losses[1]:.6f} aux={losses[2]:.6f} "
        f"tune={tune_loss:.6f}", flush=True,
    )


def _restore_selection(
    model: m2.AdvantageM2AuxiliaryCNNV1,
    state: Mapping[str, torch.Tensor] | None, epoch: int, initial_sha256: str,
    tune: M2SamplesV1, args: argparse.Namespace, device: torch.device,
) -> SelectionV1:
    if state is None:
        raise AdvantageM2TrainingError("best checkpointを選べませんでした")
    model.load_state_dict(state)
    prediction = _predict(model, tune, args, device)
    slope = legacy.symmetric_calibration_slope(tune, prediction.probability)
    calibrated = legacy.calibrate(prediction.probability, slope)
    return SelectionV1(
        model, slope, epoch, initial_sha256,
        legacy.metrics(tune, prediction.probability), legacy.metrics(tune, calibrated),
        _weighted_auxiliary_mse(tune, prediction.auxiliary_mse_rows),
    )


def _run_fold(
    samples: M2SamplesV1, plan: FoldPlan, coefficient: float, seed: int,
    args: argparse.Namespace, device: torch.device,
) -> tuple[PredictionV1, np.ndarray, Mapping[str, Any]]:
    fold_seed = seed + plan.eval_fold * 100
    selection = _fit_fold(samples, plan, coefficient, fold_seed, args, device)
    evaluation = samples.subset(samples.folds == plan.eval_fold)
    prediction = _predict(selection.model, evaluation, args, device)
    calibrated = legacy.calibrate(prediction.probability, selection.slope)
    model_name = f"{_variant_name(coefficient)}__seed_{seed}__fold_{plan.eval_fold}.pt"
    path = frozen._save_torch_exclusive(args.output_root / model_name, {
        "not_production": True, "coefficient": coefficient, "seed": seed,
        "fold": plan.eval_fold, "model_version": m2.M2_MODEL_VERSION,
        "input_schema_version": m2.M2_INPUT_SCHEMA_VERSION,
        "symmetric_platt_slope": selection.slope,
        "state_dict": frozen._cpu_state(selection.model),
    })
    record = _fold_record(
        selection, evaluation, prediction, calibrated, coefficient,
        seed, plan, model_name, base.file_sha256(path),
    )
    return prediction, calibrated, record


def _fold_record(
    selection: SelectionV1, evaluation: M2SamplesV1, prediction: PredictionV1,
    calibrated: np.ndarray, coefficient: float, seed: int, plan: FoldPlan,
    model_name: str, model_sha256: str,
) -> dict[str, Any]:
    return {
        "coefficient": coefficient, "seed": seed,
        "eval_fold": plan.eval_fold, "tune_fold": plan.tune_fold,
        "train_folds": list(plan.train_folds), "best_epoch": selection.best_epoch,
        "initial_state_sha256": selection.initial_state_sha256,
        "symmetric_platt_slope": selection.slope,
        "tune_raw": dict(selection.tune_raw),
        "tune_calibrated": dict(selection.tune_calibrated),
        "tune_auxiliary_mse": selection.tune_auxiliary_mse,
        "eval_raw": legacy.metrics(evaluation, prediction.probability),
        "eval_calibrated": legacy.metrics(evaluation, calibrated),
        "eval_auxiliary_mse": _weighted_auxiliary_mse(
            evaluation, prediction.auxiliary_mse_rows,
        ),
        "model": model_name, "model_sha256": model_sha256,
        "model_version": m2.M2_MODEL_VERSION,
        "input_schema_version": m2.M2_INPUT_SCHEMA_VERSION,
    }


def _variant_name(coefficient: float) -> str:
    return "m2_aux_off" if coefficient == 0.0 else "m2_aux_on_0p3"


def _new_accumulators(count: int) -> dict[str, AccumulatorV1]:
    return {
        _variant_name(value): AccumulatorV1(
            np.full(count, np.nan, dtype=np.float32),
            np.full(count, np.nan, dtype=np.float32),
            np.full(count, np.nan, dtype=np.float32), [],
        ) for value in DEFAULT_COEFFICIENTS
    }


def _assign_fold(
    accumulator: AccumulatorV1, indices: np.ndarray, prediction: PredictionV1,
    calibrated: np.ndarray, record: Mapping[str, Any],
) -> None:
    if np.isfinite(accumulator.raw[indices]).any():
        raise AdvantageM2TrainingError("同じOOF行へ重複代入しています")
    if len(indices) != len(prediction.probability) or len(indices) != len(calibrated):
        raise AdvantageM2TrainingError("OOF fold行数が一致しません")
    accumulator.raw[indices] = prediction.probability
    accumulator.calibrated[indices] = calibrated
    accumulator.auxiliary_mse_rows[indices] = prediction.auxiliary_mse_rows
    accumulator.records.append(record)


def _finalize(
    samples: M2SamplesV1, accumulator: AccumulatorV1, expected: np.ndarray,
) -> dict[str, Any]:
    mask = np.isfinite(accumulator.raw)
    if not np.array_equal(mask, np.isfinite(accumulator.calibrated)):
        raise AdvantageM2TrainingError("raw/calibrated coverageが一致しません")
    if not np.array_equal(mask, np.isfinite(accumulator.auxiliary_mse_rows)):
        raise AdvantageM2TrainingError("勝率/補助MSE coverageが一致しません")
    if not np.array_equal(mask, expected):
        raise AdvantageM2TrainingError("requested fold OOF coverageが一致しません")
    selected = samples.subset(mask)
    return {
        "raw": accumulator.raw, "calibrated": accumulator.calibrated,
        "auxiliary_mse_rows": accumulator.auxiliary_mse_rows,
        "records": accumulator.records,
        "metrics_raw": legacy.metrics(selected, accumulator.raw[mask]),
        "metrics_calibrated": legacy.metrics(selected, accumulator.calibrated[mask]),
        "auxiliary_mse": _weighted_auxiliary_mse(
            selected, accumulator.auxiliary_mse_rows[mask],
        ),
    }


def _run_seed(
    samples: M2SamplesV1, plans: Mapping[int, FoldPlan], seed: int,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, dict[str, Any]]:
    accumulators = _new_accumulators(len(samples.labels))
    for fold in args.folds:
        for coefficient in args.coefficients:
            output = _run_fold(samples, plans[fold], coefficient, seed, args, device)
            indices = np.flatnonzero(samples.folds == fold)
            _assign_fold(accumulators[_variant_name(coefficient)], indices, *output)
    expected = np.isin(samples.folds, args.folds)
    return {
        f"{name}__seed_{seed}": _finalize(samples, value, expected)
        for name, value in accumulators.items() if name in {
            _variant_name(coefficient) for coefficient in args.coefficients
        }
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    """M2 paired OOFを新規rootへ排他的に保存する。"""

    if args.output_root.exists():
        raise AdvantageM2TrainingError(f"出力先は新規必須です: {args.output_root}")
    samples, manifest = load_m2_dataset(args.dataset_root)
    plans = {plan.eval_fold: plan for plan in fixed_fold_plan(samples.folds)}
    args.output_root.mkdir(parents=True, exist_ok=False)
    plan = _plan(args, samples, manifest)
    plan_path = base._write_json_exclusive(args.output_root / "PLAN.json", plan)
    device = base._device(args.device)
    results: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        results.update(_run_seed(samples, plans, seed, args, device))
    return _write_results(args, samples, manifest, plan, plan_path, results)


def _plan(
    args: argparse.Namespace, samples: M2SamplesV1, manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": PLAN_VERSION, "not_production": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_sha256": manifest["dataset"]["sha256"],
        "state_count": len(samples.labels),
        "game_count": len(np.unique(samples.game_keys)),
        "source_count": len(np.unique(samples.source_groups)),
        "coefficients": list(args.coefficients), "seeds": list(args.seeds),
        "folds": list(args.folds), "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "fixed_outer_folds": 6,
        "paired_protocol": "same_rows_folds_seed_initialization_batch_order",
        "selection_metric": "tune_game_equal_raw_log_loss_not_auxiliary_loss",
        "calibration": "per_candidate_tune_only_symmetric_zero_intercept_platt",
        "auxiliary_role": "training_only_side_head_not_win_probability_input",
        "production_config_changed": False,
        "code_sha256": _code_hashes(),
    }


def _code_hashes() -> dict[str, str]:
    return {
        "trainer": base.file_sha256(Path(__file__)),
        "model": base.file_sha256(REPO_ROOT / "src/advantage_m2_auxiliary_cnn_v1.py"),
        "dataset_builder": base.file_sha256(
            REPO_ROOT / "scripts/build_advantage_m2_dataset_v1.py"
        ),
        "production_config": base.file_sha256(REPO_ROOT / "src/production_config.py"),
    }


def _write_results(
    args: argparse.Namespace, samples: M2SamplesV1, manifest: Mapping[str, Any],
    plan: Mapping[str, Any], plan_path: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    predictions: dict[str, np.ndarray] = {
        "labels": samples.labels, "folds": samples.folds, "state_ids": samples.state_ids,
    }
    serializable: dict[str, Any] = {}
    for name, result in results.items():
        for field in ("raw", "calibrated", "auxiliary_mse_rows"):
            predictions[f"{name}__{field}"] = result[field]
        serializable[name] = {
            key: value for key, value in result.items()
            if key not in {"raw", "calibrated", "auxiliary_mse_rows"}
        }
    prediction_path = args.output_root / "oof_predictions.npz"
    legacy._save_npz_exclusive(prediction_path, predictions)
    report = {
        "format_version": TRAINING_VERSION, "not_production": True,
        "dataset_source_count": manifest["source_count"],
        "plan": dict(plan), "results": serializable,
    }
    result_path = base._write_json_exclusive(args.output_root / "results.json", report)
    base._write_json_exclusive(args.output_root / "COMPLETE", {
        "format_version": COMPLETE_VERSION,
        "plan_sha256": base.file_sha256(plan_path),
        "results_sha256": base.file_sha256(result_path),
        "predictions_sha256": base.file_sha256(prediction_path),
    })
    return report


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("整数のカンマ区切りが必要です") from error


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("小数のカンマ区切りが必要です") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coefficients", type=_csv_floats, default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--folds", type=_csv_ints, default=tuple(range(1, 7)))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    allowed = set(DEFAULT_COEFFICIENTS)
    if (not args.coefficients or set(args.coefficients) - allowed
            or len(args.coefficients) != len(set(args.coefficients))):
        parser.error("coefficientsは0.0または0.3だけです")
    if not args.seeds or not args.folds or set(args.folds) - set(range(1, 7)):
        parser.error("seedを1つ以上、foldを1〜6から指定してください")
    if any(value <= 0 for value in (
        args.epochs, args.patience, args.batch_size, args.learning_rate,
    )) or args.weight_decay < 0:
        parser.error("学習hyperparameterが不正です")
    if args.output_root.exists():
        parser.error("output-rootは新規path必須です")


def main(argv: Sequence[str] | None = None) -> int:
    report = train(parse_args(argv))
    print(json.dumps(report["results"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
