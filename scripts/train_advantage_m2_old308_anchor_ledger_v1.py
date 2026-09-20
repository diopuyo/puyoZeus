"""同一46 sourceで旧308 treeと6原始ledger残差を診断比較する。"""

from __future__ import annotations

from scripts.production_dependency_contract import dependency_receipt

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import build_advantage_m2_dataset_v1 as base
from scripts import build_advantage_m2_old308_anchor_dataset_v1 as builder
from src.advantage_m2_old308_ledger_residual_v1 import (
    LEDGER_VARIANTS,
    MODEL_VERSION,
    AdvantageM2Old308LedgerResidualV1,
)
from src.event_provisional_oof_v1 import (
    FoldPlan,
    PlattCalibration,
    apply_platt_calibration,
    fixed_fold_plan,
    run_fixed_oof,
    symmetric_predict_probability,
    weighted_probability_metrics,
)


FORMAT_VERSION = "advantage-m2-old308-anchor-ledger-oof/v1"
PLAN_VERSION = "advantage-m2-old308-anchor-ledger-plan/v1"
DEFAULT_DATASET_ROOT = Path(
    "data/verify/advantage_m2_old308_anchor_dataset_46v_2026-09-05_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/verify/advantage_m2_old308_anchor_ledger_oof_46v_2026-09-05_v1"
)
DEFAULT_SEEDS = (20260904, 20260905, 20260906)
PROBABILITY_EPSILON = 1e-6


class Old308LedgerTrainingError(RuntimeError):
    """診断OOFのデータ・分割・保存契約違反。"""


@dataclass(frozen=True, slots=True)
class Samples:
    """join済み308列とcanonical ledgerの全行。"""

    features: np.ndarray
    ledger_values: np.ndarray
    ledger_availability: np.ndarray
    ledger_usable: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    folds: np.ndarray
    groups: np.ndarray
    games: np.ndarray
    state_ids: np.ndarray
    anchor_input_usable: np.ndarray
    anchor_training_usable: np.ndarray
    feature_names: tuple[str, ...]

    def subset(self, indices: np.ndarray) -> Samples:
        fields = self.__dataclass_fields__
        values = [getattr(self, name)[indices] for name in tuple(fields)[:-1]]
        return Samples(*values, self.feature_names)


@dataclass(frozen=True, slots=True)
class PreparedLedger:
    """variantに応じて実ledgerまたはmatched-randomを保持する。"""

    values: np.ndarray
    availability: np.ndarray
    supported: np.ndarray
    receipt: dict[str, int]


@dataclass(frozen=True, slots=True)
class Selection:
    """tune foldが選んだ残差状態。"""

    state: dict[str, torch.Tensor]
    epoch: int
    tune_loss: float
    fallback: bool
    train_receipt: dict[str, int]
    tune_receipt: dict[str, int]


class LedgerDataset(Dataset[tuple[torch.Tensor, ...]]):
    """anchor logitとledgerを左右交換augmentation付きで供給する。"""

    def __init__(
        self, anchor_logit: np.ndarray, prepared: PreparedLedger,
        labels: np.ndarray, weights: np.ndarray, seed: int,
    ) -> None:
        self.anchor = torch.from_numpy(anchor_logit.astype(np.float32, copy=False))
        self.values = torch.from_numpy(prepared.values.astype(np.float32, copy=False))
        self.masks = torch.from_numpy(prepared.availability.astype(np.float32, copy=False))
        self.supported = torch.from_numpy(prepared.supported.astype(np.bool_, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.float32, copy=False))
        self.weights = torch.from_numpy(weights.astype(np.float32, copy=False))
        rng = np.random.default_rng(seed)
        self.swap = torch.from_numpy((rng.random(len(labels)) < 0.5).astype(np.bool_))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if not bool(self.swap[index]):
            return self._direct(index)
        return (
            -self.anchor[index], self.values[index].flip(0), self.masks[index].flip(0),
            self.supported[index], 1.0 - self.labels[index], self.weights[index],
        )

    def _direct(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.anchor[index], self.values[index], self.masks[index],
            self.supported[index], self.labels[index], self.weights[index],
        )


def load_dataset(root: Path) -> tuple[Samples, dict[str, Any]]:
    """排他的receiptと各array hashを検証して読む。"""

    manifest_path, dataset_path = root / "manifest.json", root / "dataset.npz"
    manifest = _load_json(manifest_path)
    complete = _load_json(root / "COMPLETE")
    if complete.get("manifest_sha256") != base.file_sha256(manifest_path):
        raise Old308LedgerTrainingError("dataset manifest receiptが一致しません")
    if complete.get("dataset_sha256") != base.file_sha256(dataset_path):
        raise Old308LedgerTrainingError("dataset NPZ receiptが一致しません")
    if manifest.get("format_version") != builder.FORMAT_VERSION:
        raise Old308LedgerTrainingError("dataset formatが不正です")
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    _validate_hashes(arrays, manifest)
    samples = _samples(arrays, manifest)
    _validate_samples(samples, manifest)
    return samples, manifest


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Old308LedgerTrainingError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise Old308LedgerTrainingError(f"JSON objectではありません: {path}")
    return value


def _validate_hashes(arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("array_sha256")
    if not isinstance(expected, Mapping) or set(expected) != set(arrays):
        raise Old308LedgerTrainingError("array hash一覧が一致しません")
    for name, value in arrays.items():
        if expected[name] != base._array_sha256(value):
            raise Old308LedgerTrainingError(f"array hashが一致しません: {name}")


def _samples(arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]) -> Samples:
    names = tuple(map(str, manifest.get("feature_names", ())))
    return Samples(
        np.asarray(arrays["old308_features"], dtype=np.float32),
        np.asarray(arrays["ledger_values"], dtype=np.float32),
        np.asarray(arrays["ledger_availability"], dtype=np.float32),
        np.asarray(arrays["ledger_usable"], dtype=np.bool_),
        np.asarray(arrays["labels"], dtype=np.int8),
        np.asarray(arrays["weights"], dtype=np.float32),
        np.asarray(arrays["folds"], dtype=np.int8),
        np.asarray(arrays["source_groups"], dtype=str),
        np.asarray(arrays["game_keys"], dtype=str),
        np.asarray(arrays["state_ids"], dtype=str),
        np.asarray(arrays["anchor_input_usable"], dtype=np.bool_),
        np.asarray(arrays["anchor_training_usable"], dtype=np.bool_), names,
    )


def _validate_samples(samples: Samples, manifest: Mapping[str, Any]) -> None:
    count = len(samples.labels)
    array_names = tuple(samples.__dataclass_fields__)[:-1]
    if any(len(getattr(samples, name)) != count for name in array_names):
        raise Old308LedgerTrainingError("dataset array行数が一致しません")
    if samples.features.shape != (count, builder.EXPECTED_FEATURE_COUNT):
        raise Old308LedgerTrainingError("old308 feature shapeが不正です")
    if len(samples.feature_names) != samples.features.shape[1]:
        raise Old308LedgerTrainingError("feature name数が一致しません")
    if set(map(int, np.unique(samples.folds))) != {1, 2, 3, 4, 5, 6}:
        raise Old308LedgerTrainingError("固定6foldが揃っていません")
    if int(manifest.get("state_count", -1)) != count:
        raise Old308LedgerTrainingError("manifest state数が一致しません")
    if len(set(samples.state_ids.tolist())) != count:
        raise Old308LedgerTrainingError("state_idが一意ではありません")
    _validate_dataset_contract(samples.weights, manifest)


def _validate_dataset_contract(
    weights: np.ndarray, manifest: Mapping[str, Any],
) -> None:
    """診断専用・canonical重み固定の入力契約を検証する。"""

    contract = manifest.get("weight_contract")
    if not isinstance(contract, Mapping):
        raise Old308LedgerTrainingError("weight contractがありません")
    if contract.get("source") != "m2_canonical_dataset":
        raise Old308LedgerTrainingError("M2 canonical weightではありません")
    if contract.get("legacy_sample_weight_used") is not False:
        raise Old308LedgerTrainingError("旧sample weightは監査専用です")
    if weights.dtype != np.float32 or not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise Old308LedgerTrainingError("canonical weightは有限正数float32必須です")
    flags = (
        manifest.get("not_production") is True,
        manifest.get("production_config_changed") is False,
        manifest.get("attack_difference_ten_percent_correction") is False,
        manifest.get("uses_337_or_338_feature_family") is False,
    )
    if not all(flags):
        raise Old308LedgerTrainingError("診断専用dataset契約が不正です")


def _anchor_oof(samples: Samples, seed: int) -> Any:
    training = samples.anchor_training_usable & samples.anchor_input_usable
    return run_fixed_oof(
        samples.features, samples.labels, samples.weights, samples.groups,
        samples.folds, samples.feature_names, training, samples.anchor_input_usable,
        random_state=seed, model_family="tree",
    )


def _anchor_raw(
    model: Any, samples: Samples, indices: np.ndarray,
) -> np.ndarray:
    return symmetric_predict_probability(
        model, samples.features[indices], samples.feature_names,
    )


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), PROBABILITY_EPSILON, 1.0)
    clipped = np.clip(clipped, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def _prepare(samples: Samples, indices: np.ndarray, variant: str, seed: int) -> PreparedLedger:
    values = samples.ledger_values[indices].copy()
    masks = samples.ledger_availability[indices].copy()
    supported = samples.ledger_usable[indices].copy()
    if variant != "random_control":
        return PreparedLedger(values, masks, supported, _receipt(supported))
    return _randomize_ledger(values, masks, supported, seed)


def _randomize_ledger(
    values: np.ndarray, masks: np.ndarray, supported: np.ndarray, seed: int,
) -> PreparedLedger:
    keys = [item.tobytes() for item in masks]
    strata: dict[bytes, list[int]] = {}
    for index, key in enumerate(keys):
        strata.setdefault(key, []).append(index)
    rng, structural, singleton = np.random.default_rng(seed), 0, 0
    randomized = values.copy()
    effective = supported.copy()
    for members in strata.values():
        structural += _shuffle_stratum(randomized, values, effective, members, rng)
        singleton += int(len(members) == 1)
    unchanged = int(np.all(randomized == values, axis=(1, 2)).sum())
    return PreparedLedger(
        randomized, masks, effective,
        _receipt(effective, structural, singleton, unchanged),
    )


def _shuffle_stratum(
    target: np.ndarray, source: np.ndarray, effective: np.ndarray,
    members: list[int], rng: np.random.Generator,
) -> int:
    if len(members) < 2:
        effective[members] = False
        return 0
    order = np.asarray(members, dtype=int)
    for index in range(len(order) - 1, 0, -1):
        swap = int(rng.integers(0, index))
        order[index], order[swap] = order[swap], order[index]
    target[np.asarray(members, dtype=int)] = source[order]
    return len(members)


def _receipt(
    supported: np.ndarray, structural: int = 0, singleton: int = 0, unchanged: int = 0,
) -> dict[str, int]:
    return {
        "row_count": len(supported), "effective_count": int(supported.sum()),
        "structural_deranged_count": structural, "singleton_count": singleton,
        "unchanged_value_count": unchanged,
    }


def _new_model(variant: str, seed: int, device: torch.device) -> Any:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return AdvantageM2Old308LedgerResidualV1(variant).to(device)


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _loader(
    logits: np.ndarray, prepared: PreparedLedger, labels: np.ndarray,
    weights: np.ndarray, args: argparse.Namespace, seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    dataset = LedgerDataset(logits, prepared, labels, weights, seed)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        pin_memory=args.device.startswith("cuda"),
    )


def _train_epoch(
    model: Any, loader: DataLoader[Any], optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = denominator = 0.0
    for batch in loader:
        moved = tuple(value.to(device, non_blocking=True) for value in batch)
        optimizer.zero_grad(set_to_none=True)
        output = model(moved[0], moved[1], moved[2], moved[3])
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            output.logit, moved[4], reduction="none",
        )
        loss = (losses * moved[5]).sum() / moved[5].sum()
        loss.backward()
        optimizer.step()
        total += float((loss.detach() * moved[5].sum()).cpu())
        denominator += float(moved[5].sum().cpu())
    return total / denominator


def _predict(
    model: Any, logits: np.ndarray, prepared: PreparedLedger,
    args: argparse.Namespace, device: torch.device,
) -> np.ndarray:
    tensors = (
        torch.from_numpy(logits.astype(np.float32, copy=False)),
        torch.from_numpy(prepared.values.astype(np.float32, copy=False)),
        torch.from_numpy(prepared.availability.astype(np.float32, copy=False)),
        torch.from_numpy(prepared.supported.astype(np.bool_, copy=False)),
    )
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(logits), args.batch_size):
            batch = tuple(value[start:start + args.batch_size].to(device) for value in tensors)
            output.append(model(*batch).raw_probability.cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def _weighted_loss(
    labels: np.ndarray, probability: np.ndarray, weights: np.ndarray,
) -> float:
    return weighted_probability_metrics(labels, probability, weights)["log_loss"]


def _fit_residual(
    samples: Samples, plan: FoldPlan, model: Any, calibration: PlattCalibration,
    variant: str, seed: int, args: argparse.Namespace, device: torch.device,
) -> Selection:
    train_idx = np.flatnonzero(
        samples.anchor_training_usable & samples.ledger_usable
        & np.isin(samples.folds, plan.train_folds)
    )
    tune_idx = np.flatnonzero(samples.anchor_input_usable & (samples.folds == plan.tune_fold))
    train_anchor, tune_anchor = _anchor_raw(model, samples, train_idx), _anchor_raw(
        model, samples, tune_idx,
    )
    train = _prepare(samples, train_idx, variant, seed + 11)
    tune = _prepare(samples, tune_idx, variant, seed + 12)
    candidate = _new_model(variant, seed, device)
    baseline = apply_platt_calibration(tune_anchor, calibration)
    best_state, best_epoch = _cpu_state(candidate), 0
    best_loss = _weighted_loss(samples.labels[tune_idx], baseline, samples.weights[tune_idx])
    optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loader = _loader(
        _logit(train_anchor), train, samples.labels[train_idx], samples.weights[train_idx],
        args, seed,
    )
    best_state, best_epoch, best_loss = _train_loop(
        candidate, loader, _logit(tune_anchor), tune, samples.labels[tune_idx],
        samples.weights[tune_idx], calibration, optimizer, best_state, best_epoch,
        best_loss, args, device,
    )
    return Selection(
        best_state, best_epoch, best_loss, best_epoch == 0, train.receipt, tune.receipt,
    )


def _train_loop(
    model: Any, loader: DataLoader[Any], tune_logits: np.ndarray,
    tune: PreparedLedger, labels: np.ndarray, weights: np.ndarray,
    calibration: PlattCalibration, optimizer: torch.optim.Optimizer,
    best_state: dict[str, torch.Tensor], best_epoch: int, best_loss: float,
    args: argparse.Namespace, device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    stale = 0
    for epoch in range(1, args.epochs + 1):
        _train_epoch(model, loader, optimizer, device)
        raw = _predict(model, tune_logits, tune, args, device)
        loss = _weighted_loss(labels, apply_platt_calibration(raw, calibration), weights)
        if loss < best_loss:
            best_state, best_epoch, best_loss, stale = _cpu_state(model), epoch, loss, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    return best_state, best_epoch, best_loss


def _evaluate_variant(
    samples: Samples, plan: FoldPlan, anchor: Any, record: Any,
    variant: str, seed: int, args: argparse.Namespace, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fold_seed = seed + plan.eval_fold * 100
    selection = _fit_residual(
        samples, plan, anchor, record.calibration, variant, fold_seed, args, device,
    )
    eval_idx = np.flatnonzero(samples.anchor_input_usable & (samples.folds == plan.eval_fold))
    anchor_raw = _anchor_raw(anchor, samples, eval_idx)
    prepared = _prepare(samples, eval_idx, variant, fold_seed + 13)
    model = _new_model(variant, fold_seed, device)
    model.load_state_dict(selection.state)
    raw = anchor_raw.copy() if selection.fallback else _predict(
        model, _logit(anchor_raw), prepared, args, device,
    )
    raw = _restore_unsupported(raw, anchor_raw, prepared.supported)
    calibrated = apply_platt_calibration(raw, record.calibration)
    model_path = _save_residual_model(
        model, selection, record, variant, seed, plan.eval_fold, args.output_root,
    )
    metadata = _fold_metadata(
        samples, eval_idx, raw, calibrated, selection, prepared, record,
        model_path, variant, seed, plan,
    )
    return raw, calibrated, metadata


def _restore_unsupported(
    candidate: np.ndarray, anchor: np.ndarray, supported: np.ndarray,
) -> np.ndarray:
    """logit往復誤差を残さずunsupported確率をanchorへ厳密復元する。"""

    restored = np.asarray(candidate, dtype=np.float64).copy()
    restored[~supported] = np.asarray(anchor, dtype=np.float64)[~supported]
    return restored


def _save_residual_model(
    model: Any, selection: Selection, record: Any, variant: str,
    seed: int, fold: int, root: Path,
) -> Path:
    path = root / f"m2_old308_{variant}__seed_{seed}__fold_{fold}.pt"
    if path.exists():
        raise Old308LedgerTrainingError(f"model出力は新規必須です: {path}")
    torch.save({
        "not_production": True, "diagnostic_only": True, "model_version": MODEL_VERSION,
        "variant": variant, "seed": seed, "fold": fold,
        "fallback_to_anchor": selection.fallback,
        "fixed_anchor_platt": asdict(record.calibration),
        "state_dict": _cpu_state(model),
    }, path)
    return path


def _fold_metadata(
    samples: Samples, indices: np.ndarray, raw: np.ndarray, calibrated: np.ndarray,
    selection: Selection, prepared: PreparedLedger, record: Any, model_path: Path,
    variant: str, seed: int, plan: FoldPlan,
) -> dict[str, Any]:
    return {
        "variant": variant, "seed": seed, "eval_fold": plan.eval_fold,
        "tune_fold": plan.tune_fold, "train_folds": list(plan.train_folds),
        "selected_epoch": selection.epoch, "fallback_to_anchor": selection.fallback,
        "tune_game_equal_log_loss": selection.tune_loss,
        "fixed_anchor_platt": asdict(record.calibration),
        "anchor_model_hash": record.model_hash,
        "model_path": str(model_path.resolve()), "model_sha256": base.file_sha256(model_path),
        "train_random_receipt": selection.train_receipt,
        "tune_random_receipt": selection.tune_receipt,
        "eval_random_receipt": prepared.receipt,
        "metrics_raw": weighted_probability_metrics(
            samples.labels[indices], raw, samples.weights[indices],
        ),
        "metrics_calibrated": weighted_probability_metrics(
            samples.labels[indices], calibrated, samples.weights[indices],
        ),
    }


def _save_anchor_models(
    result: Any, seed: int, names: Sequence[str], root: Path,
) -> list[dict[str, Any]]:
    output = []
    for record, model in zip(result.records, result.models, strict=True):
        path = root / f"old308_anchor__seed_{seed}__fold_{record.plan.eval_fold}.pkl"
        _save_pickle_exclusive(path, {
            "not_production": True, "diagnostic_only": True,
            "feature_names": list(names), "record": record, "model": model,
        })
        output.append({
            "eval_fold": record.plan.eval_fold, "path": str(path.resolve()),
            "sha256": base.file_sha256(path), "record": _record_json(record),
        })
    return output


def _record_json(record: Any) -> dict[str, Any]:
    return {
        "plan": asdict(record.plan), "model_family": record.model_family,
        "model_hash": record.model_hash, "calibration": asdict(record.calibration),
        "metrics_raw": record.metrics_raw, "metrics_calibrated": record.metrics_calibrated,
        "train_row_count": record.train_row_count, "tune_row_count": record.tune_row_count,
        "eval_row_count": record.eval_row_count,
    }


def _save_pickle_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise Old308LedgerTrainingError(f"pickle出力は新規必須です: {path}")
    path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _run_seed(
    samples: Samples, seed: int, args: argparse.Namespace, device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    anchor = _anchor_oof(samples, seed)
    predictions = {
        f"old308_anchor__seed_{seed}__raw": anchor.raw_probability,
        f"old308_anchor__seed_{seed}__calibrated": anchor.calibrated_probability,
    }
    records: dict[str, Any] = {
        "anchor_models": _save_anchor_models(anchor, seed, samples.feature_names, args.output_root),
        "variants": {},
    }
    plans = fixed_fold_plan(samples.folds.tolist())
    for variant in args.variants:
        raw = np.full(len(samples.labels), np.nan, dtype=np.float64)
        calibrated = np.full(len(samples.labels), np.nan, dtype=np.float64)
        variant_records = []
        for plan, model, record in zip(plans, anchor.models, anchor.records, strict=True):
            indices = np.flatnonzero(samples.anchor_input_usable & (samples.folds == plan.eval_fold))
            fold_raw, fold_calibrated, metadata = _evaluate_variant(
                samples, plan, model, record, variant, seed, args, device,
            )
            raw[indices], calibrated[indices] = fold_raw, fold_calibrated
            variant_records.append(metadata)
        key = f"m2_old308_{variant}__seed_{seed}"
        predictions[f"{key}__raw"], predictions[f"{key}__calibrated"] = raw, calibrated
        records["variants"][variant] = {
            "folds": variant_records,
            "metrics_raw": _masked_metrics(samples, raw),
            "metrics_calibrated": _masked_metrics(samples, calibrated),
        }
    records["anchor_metrics_raw"] = _masked_metrics(samples, anchor.raw_probability)
    records["anchor_metrics_calibrated"] = _masked_metrics(
        samples, anchor.calibrated_probability,
    )
    return predictions, records


def _masked_metrics(samples: Samples, probability: np.ndarray) -> dict[str, float]:
    mask = samples.anchor_input_usable & np.isfinite(probability)
    return weighted_probability_metrics(
        samples.labels[mask], probability[mask], samples.weights[mask],
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    """3 seed OOFを新規rootへ保存し、本番資産には触れない。"""

    if args.output_root.exists():
        raise Old308LedgerTrainingError(f"出力先は新規必須です: {args.output_root}")
    samples, manifest = load_dataset(args.dataset_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    plan = _plan(args, samples, manifest)
    plan_path = args.output_root / "PLAN.json"
    _write_json_exclusive(plan_path, plan)
    device = _device(args.device)
    predictions: dict[str, np.ndarray] = {
        "labels": samples.labels, "weights": samples.weights, "folds": samples.folds,
        "state_ids": samples.state_ids, "source_groups": samples.groups,
        "game_keys": samples.games, "anchor_input_usable": samples.anchor_input_usable,
    }
    results: dict[str, Any] = {}
    for seed in args.seeds:
        seed_predictions, seed_result = _run_seed(samples, seed, args, device)
        predictions.update(seed_predictions)
        results[str(seed)] = seed_result
    return _write_complete(args, plan, plan_path, predictions, results)


def _plan(
    args: argparse.Namespace, samples: Samples, manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": PLAN_VERSION, "not_production": True, "diagnostic_only": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_manifest_sha256": base.file_sha256(args.dataset_root / "manifest.json"),
        "dataset_sha256": manifest["dataset"]["sha256"],
        "state_count": len(samples.labels), "source_count": len(set(samples.groups)),
        "game_count": len(set(samples.games)), "feature_count": len(samples.feature_names),
        "seeds": list(args.seeds), "folds": [1, 2, 3, 4, 5, 6],
        "variants": list(args.variants), "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "device": args.device,
        "anchor": "A_common_original old308 HistGradientBoosting current 46-source refit",
        "residual": "raw typed 6-field causal ledger actual-minus-zero antisymmetric",
        "calibration": "paired anchor tune-fold symmetric zero-intercept Platt slope fixed",
        "fallback": "epoch0/unsupported exact anchor; integrity fault HOLD/FAULT",
        "attack_difference_ten_percent_correction": False,
        "uses_337_or_338_feature_family": False,
        "production_config_changed": False,
        "code_sha256": _code_hashes(),
        "production_dependency_contract": dependency_receipt(),
    }


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        "trainer": base.file_sha256(Path(__file__)),
        "model": base.file_sha256(root / "src/advantage_m2_old308_ledger_residual_v1.py"),
        "builder": base.file_sha256(root / "scripts/build_advantage_m2_old308_anchor_dataset_v1.py"),
        "oof": base.file_sha256(root / "src/event_provisional_oof_v1.py"),
        "production_config": base.file_sha256(root / "src/production_config.py"),
    }


def _write_complete(
    args: argparse.Namespace, plan: Mapping[str, Any], plan_path: Path,
    predictions: Mapping[str, np.ndarray], results: Mapping[str, Any],
) -> dict[str, Any]:
    prediction_path, result_path = args.output_root / "oof_predictions.npz", args.output_root / "RESULTS.json"
    _write_npz_exclusive(prediction_path, predictions)
    _write_json_exclusive(result_path, {
        "format_version": FORMAT_VERSION, "not_production": True,
        "diagnostic_only": True, "plan": plan, "seeds": results,
    })
    complete = {
        "format_version": FORMAT_VERSION, "not_production": True,
        "plan_sha256": base.file_sha256(plan_path),
        "predictions_sha256": base.file_sha256(prediction_path),
        "results_sha256": base.file_sha256(result_path),
        "production_config_changed": False,
    }
    _write_json_exclusive(args.output_root / "COMPLETE", complete)
    return complete


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Old308LedgerTrainingError(f"JSON出力は新規必須です: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_npz_exclusive(path: Path, values: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise Old308LedgerTrainingError(f"NPZ出力は新規必須です: {path}")
    np.savez_compressed(path, **values)


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise Old308LedgerTrainingError("CUDAが利用できません")
    return torch.device(name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--variants", default=",".join(LEDGER_VARIANTS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    args.seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    args.variants = tuple(value for value in args.variants.split(",") if value)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.seeds or not args.variants:
        parser.error("seed/variantは1個以上必要です")
    if any(value not in LEDGER_VARIANTS for value in args.variants):
        parser.error("未対応variantがあります")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variantが重複しています")
    if min(args.epochs, args.patience, args.batch_size) <= 0:
        parser.error("epoch/patience/batch-sizeは正数必須です")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("learning-rateは有限正数必須です")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        parser.error("weight-decayは有限非負必須です")


def main(argv: Sequence[str] | None = None) -> int:
    result = train(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
