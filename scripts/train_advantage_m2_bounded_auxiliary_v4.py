"""M1非拡大残差へ火力・接続auxiliaryを二段階学習する固定OOF。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import build_advantage_m2_auxiliary_family_dataset_v4 as family_data
from scripts import build_advantage_m2_dataset_v1 as receipt
from scripts import train_advantage_m0_current_cnn_v1 as train_base
from scripts import train_advantage_m1_causal_ledger_v1 as legacy
from scripts import train_advantage_m1_frozen_residual_v1 as frozen
from scripts import train_advantage_m2_anchored_auxiliary_v2 as anchor_train
from scripts import train_advantage_m2_auxiliary_v1 as m2_data
from src import advantage_m2_bounded_auxiliary_v4 as model_v4
from src.event_provisional_oof_v1 import FoldPlan, fixed_fold_plan


TRAINING_VERSION = "advantage-m2-bounded-auxiliary-fixed-oof-training/v4"
PLAN_VERSION = "advantage-m2-bounded-auxiliary-fixed-oof-plan/v4"
COMPLETE_VERSION = "advantage-m2-bounded-auxiliary-fixed-oof-complete/v4"
DEFAULT_DATASET_ROOT = family_data.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "data/verify/advantage_m2_bounded_auxiliary_oof_46v_2026-09-06_v4"
)
AUXILIARY_VARIANTS = (
    "aux_off", "firepower_and_ignition", "connectivity_and_shape",
    "firepower_and_connectivity", "matched_random",
)
DEFAULT_SEEDS = (20260904, 20260905, 20260906)
DEFAULT_FOLDS = tuple(range(1, 7))
AUXILIARY_COEFFICIENT = 0.3
DEFAULT_AUXILIARY_EPOCHS = 2
DEFAULT_AUXILIARY_LEARNING_RATE = 1e-4
REPO_ROOT = Path(__file__).resolve().parents[1]


class AdvantageM2BoundedTrainingError(RuntimeError):
    """V4 dataset、fallback、OOF保存契約の違反。"""


@dataclass(frozen=True, slots=True)
class PreparedAuxiliary:
    samples: m2_data.M2SamplesV1
    receipt: Mapping[str, int | str]


@dataclass(frozen=True, slots=True)
class PredictionV4:
    probability: np.ndarray
    auxiliary_mse_rows: np.ndarray
    shrink_fraction: np.ndarray


@dataclass(frozen=True, slots=True)
class SelectionV4:
    model: model_v4.AdvantageM2BoundedAuxiliaryV4
    best_epoch: int
    fallback_to_m1: bool
    tune_loss: float
    auxiliary_pretrain_mse: float
    train_auxiliary_receipt: Mapping[str, int | str]
    tune_auxiliary_receipt: Mapping[str, int | str]


@dataclass(slots=True)
class AccumulatorV4:
    raw: np.ndarray
    calibrated: np.ndarray
    auxiliary_mse_rows: np.ndarray
    shrink_fraction: np.ndarray
    records: list[Mapping[str, Any]]


def load_family_dataset(root: Path) -> tuple[m2_data.M2SamplesV1, dict[str, Any]]:
    """family sidecarと親M2の排他receipt・同一行性を検査する。"""

    manifest = _read_json(root / "manifest.json")
    complete = _read_json(root / "COMPLETE")
    data_path = root / "dataset.npz"
    _validate_family_receipts(root, manifest, complete, data_path)
    parent_root = Path(str(manifest["parent_m2_root"]))
    samples, parent_manifest = m2_data.load_m2_dataset(parent_root)
    with np.load(data_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    _validate_family_arrays(samples, manifest, arrays)
    combined = replace(
        samples, auxiliary_targets=np.asarray(arrays["auxiliary_targets"], np.float32),
        auxiliary_mask=np.asarray(arrays["auxiliary_mask"], np.bool_),
    )
    return combined, manifest | {"parent_m2_manifest": parent_manifest}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM2BoundedTrainingError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise AdvantageM2BoundedTrainingError(f"JSON objectではありません: {path}")
    return value


def _validate_family_receipts(
    root: Path, manifest: Mapping[str, Any], complete: Mapping[str, Any], data_path: Path,
) -> None:
    checks = (
        manifest.get("format_version") == family_data.FORMAT_VERSION,
        complete.get("format_version") == family_data.COMPLETE_VERSION,
        manifest.get("not_production") is True,
        manifest.get("production_config_changed") is False,
        manifest.get("attack_difference_ten_percent_correction") is False,
        complete.get("manifest_sha256") == receipt.file_sha256(root / "manifest.json"),
        complete.get("dataset_sha256") == receipt.file_sha256(data_path),
    )
    if not all(checks):
        raise AdvantageM2BoundedTrainingError("family dataset receiptが不正です")


def _validate_family_arrays(
    samples: m2_data.M2SamplesV1, manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    if set(arrays) != {"state_ids", "auxiliary_targets", "auxiliary_mask"}:
        raise AdvantageM2BoundedTrainingError("family dataset array schemaが不正です")
    if not np.array_equal(arrays["state_ids"], samples.state_ids):
        raise AdvantageM2BoundedTrainingError("family dataset state ID順が不一致です")
    shape = (len(samples.labels), 2, len(model_v4.AUXILIARY_FEATURE_NAMES_V4))
    targets, mask = arrays["auxiliary_targets"], arrays["auxiliary_mask"]
    if targets.dtype != np.float32 or mask.dtype != np.bool_ or targets.shape != shape:
        raise AdvantageM2BoundedTrainingError("family target shape/dtypeが不正です")
    if mask.shape != shape or tuple(manifest.get("feature_order", ())) != model_v4.AUXILIARY_FEATURE_NAMES_V4:
        raise AdvantageM2BoundedTrainingError("family target maskまたは列順が不正です")
    if not np.isfinite(targets[mask]).all() or np.any((targets[mask] < 0) | (targets[mask] > 1)):
        raise AdvantageM2BoundedTrainingError("family targetは有限な0..1必須です")
    for name, expected in manifest.get("array_sha256", {}).items():
        if name not in arrays or receipt._array_sha256(arrays[name]) != expected:
            raise AdvantageM2BoundedTrainingError(f"family array hashが不一致です: {name}")


def prepare_auxiliary(
    samples: m2_data.M2SamplesV1, variant: str, seed: int,
) -> PreparedAuxiliary:
    """variant対象列だけを有効化し、nullはsource内で行をderangeする。"""

    if variant not in AUXILIARY_VARIANTS:
        raise AdvantageM2BoundedTrainingError(f"未対応variantです: {variant}")
    targets, mask = samples.auxiliary_targets.copy(), np.zeros_like(samples.auxiliary_mask)
    family = "firepower_and_connectivity" if variant == "matched_random" else variant
    if family != "aux_off":
        mask[..., model_v4.family_indices(family)] = samples.auxiliary_mask[
            ..., model_v4.family_indices(family)
        ]
    receipt_row: dict[str, int | str] = {
        "variant": variant, "row_count": len(samples.labels),
        "active_value_count": int(mask.sum()), "deranged_row_count": 0,
        "singleton_row_count": 0,
    }
    if variant == "matched_random":
        targets, mask, deranged, singleton = _randomize_targets(
            targets, mask, samples.source_groups, seed,
        )
        receipt_row["deranged_row_count"] = deranged
        receipt_row["singleton_row_count"] = singleton
    return PreparedAuxiliary(
        replace(samples, auxiliary_targets=targets, auxiliary_mask=mask), receipt_row,
    )


def _randomize_targets(
    targets: np.ndarray, mask: np.ndarray, groups: np.ndarray, seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    randomized_targets, randomized_mask = targets.copy(), mask.copy()
    rng, deranged, singleton = np.random.default_rng(seed), 0, 0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            randomized_mask[indices] = False
            singleton += len(indices)
            continue
        order = indices.copy()
        for position in range(len(order) - 1, 0, -1):
            selected = int(rng.integers(0, position))
            order[position], order[selected] = order[selected], order[position]
        randomized_targets[indices] = targets[order]
        randomized_mask[indices] = mask[order]
        deranged += len(indices)
    return randomized_targets, randomized_mask, deranged, singleton


def _new_model(
    reference: anchor_train.AnchorReferenceV2, device: torch.device, seed: int,
) -> model_v4.AdvantageM2BoundedAuxiliaryV4:
    train_base._set_seed(seed)
    return model_v4.AdvantageM2BoundedAuxiliaryV4(
        anchor_train._load_anchor(reference, device),
    ).to(device)


def _auxiliary_parameters(
    model: model_v4.AdvantageM2BoundedAuxiliaryV4,
) -> list[torch.nn.Parameter]:
    parameters = [*model.adapted_side_encoder.parameters(), *model.auxiliary_head.parameters()]
    if not parameters or not all(value.requires_grad for value in parameters):
        raise AdvantageM2BoundedTrainingError("auxiliary parameterが不正です")
    return parameters


def _win_parameters(
    model: model_v4.AdvantageM2BoundedAuxiliaryV4,
) -> list[torch.nn.Parameter]:
    parameters = [*model.all_clear_encoder.parameters(), *model.shrink_scorer.parameters()]
    excluded = {id(value) for module in (
        model.anchor, model.adapted_side_encoder, model.auxiliary_head,
    ) for value in module.parameters()}
    if not parameters or any(id(value) in excluded for value in parameters):
        raise AdvantageM2BoundedTrainingError("勝率optimizerへ凍結parameterが混入しました")
    if not all(value.requires_grad for value in parameters):
        raise AdvantageM2BoundedTrainingError("勝率parameterが学習不能です")
    return parameters


def _pretrain_auxiliary(
    model: model_v4.AdvantageM2BoundedAuxiliaryV4,
    prepared: PreparedAuxiliary, args: argparse.Namespace, seed: int,
    device: torch.device,
) -> float:
    if prepared.receipt["active_value_count"] == 0:
        model.freeze_auxiliary_encoder()
        return _auxiliary_mse(model, prepared.samples, args, device)
    optimizer = torch.optim.AdamW(
        _auxiliary_parameters(model), lr=args.auxiliary_learning_rate,
        weight_decay=args.weight_decay,
    )
    loader = m2_data._loader(prepared.samples, args, augment=True, seed=seed)
    last = float("nan")
    for epoch in range(1, args.auxiliary_epochs + 1):
        last = _auxiliary_epoch(model, loader, optimizer, device)
        print(
            f"[aux {prepared.receipt['variant']}] seed={seed} epoch={epoch} mse={last:.6f}",
            flush=True,
        )
    model.freeze_auxiliary_encoder()
    return last


def _auxiliary_epoch(
    model: model_v4.AdvantageM2BoundedAuxiliaryV4, loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    model.train()
    total = denominator = 0.0
    for batch in loader:
        moved = tuple(value.to(device) for value in batch)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.predict_auxiliary(moved[0], moved[1])
        rows = m2_data._auxiliary_mse_rows(prediction, moved[8], moved[9])
        loss = (rows * moved[7]).sum() / moved[7].sum()
        (AUXILIARY_COEFFICIENT * loss).backward()
        optimizer.step()
        weight = float(moved[7].sum().cpu())
        total, denominator = total + float(loss.detach().cpu()) * weight, denominator + weight
    return total / denominator


def _auxiliary_mse(
    model: model_v4.AdvantageM2BoundedAuxiliaryV4, samples: m2_data.M2SamplesV1,
    args: argparse.Namespace, device: torch.device,
) -> float:
    prediction = _predict(model, samples, args, device)
    return m2_data._weighted_auxiliary_mse(samples, prediction.auxiliary_mse_rows)


def _predict(
    model: model_v4.AdvantageM2BoundedAuxiliaryV4, samples: m2_data.M2SamplesV1,
    args: argparse.Namespace, device: torch.device,
) -> PredictionV4:
    probabilities: list[np.ndarray] = []
    auxiliary_rows: list[np.ndarray] = []
    shrink_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in m2_data._loader(samples, args, augment=False, seed=0):
            moved = tuple(value.to(device) for value in batch)
            output = model(*moved[:6])
            probabilities.append(output.raw_probability.cpu().numpy())
            auxiliary_rows.append(m2_data._auxiliary_mse_rows(
                output.auxiliary, moved[8], moved[9],
            ).cpu().numpy())
            shrink_rows.append(output.shrink_fraction.cpu().numpy())
    return PredictionV4(
        np.concatenate(probabilities), np.concatenate(auxiliary_rows),
        np.concatenate(shrink_rows),
    )


def _train_win_epoch(
    model: model_v4.AdvantageM2BoundedAuxiliaryV4, loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    model.train()
    total = denominator = 0.0
    for batch in loader:
        moved = tuple(value.to(device) for value in batch)
        optimizer.zero_grad(set_to_none=True)
        output = model(*moved[:6])
        rows = torch.nn.functional.binary_cross_entropy_with_logits(
            output.logit, moved[6], reduction="none",
        )
        loss = (rows * moved[7]).sum() / moved[7].sum()
        loss.backward()
        optimizer.step()
        weight = float(moved[7].sum().cpu())
        total, denominator = total + float(loss.detach().cpu()) * weight, denominator + weight
    return total / denominator


def _initial_selection(
    model: model_v4.AdvantageM2BoundedAuxiliaryV4, tune: m2_data.M2SamplesV1,
    reference: anchor_train.AnchorReferenceV2, args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    prediction = _predict(model, tune, args, device)
    baseline = anchor_train._predict_anchor(model.anchor, tune, args, device)
    if not np.array_equal(prediction.probability, baseline):
        raise AdvantageM2BoundedTrainingError("epoch 0がM1とbit-identicalではありません")
    calibrated = legacy.calibrate(baseline, reference.slope)
    return frozen._cpu_state(model), legacy.metrics(tune, calibrated)["game_equal_log_loss"]


def _fit_fold(
    samples: m2_data.M2SamplesV1, plan: FoldPlan, variant: str, seed: int,
    reference: anchor_train.AnchorReferenceV2, args: argparse.Namespace,
    device: torch.device,
) -> SelectionV4:
    train = prepare_auxiliary(
        samples.subset(np.isin(samples.folds, plan.train_folds)), variant, seed + 11,
    )
    tune = prepare_auxiliary(
        samples.subset(samples.folds == plan.tune_fold), variant, seed + 12,
    )
    model = _new_model(reference, device, seed)
    auxiliary_mse = _pretrain_auxiliary(model, train, args, seed, device)
    best, best_loss = _initial_selection(model, tune.samples, reference, args, device)
    optimizer = torch.optim.AdamW(
        _win_parameters(model), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    best_epoch, stale = 0, 0
    loader = m2_data._loader(train.samples, args, augment=True, seed=seed)
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_win_epoch(model, loader, optimizer, device)
        raw = _predict(model, tune.samples, args, device).probability
        loss = legacy.metrics(tune.samples, legacy.calibrate(raw, reference.slope))[
            "game_equal_log_loss"
        ]
        print(
            f"[win {variant}] seed={seed} eval={plan.eval_fold} epoch={epoch} "
            f"train={train_loss:.6f} tune={loss:.6f}", flush=True,
        )
        if loss < best_loss:
            best, best_loss, best_epoch, stale = frozen._cpu_state(model), loss, epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best)
    _validate_selected_fallback(model, tune.samples, reference, best_epoch, args, device)
    return SelectionV4(
        model, best_epoch, best_epoch == 0, best_loss, auxiliary_mse,
        train.receipt, tune.receipt,
    )


def _validate_selected_fallback(
    model: model_v4.AdvantageM2BoundedAuxiliaryV4, tune: m2_data.M2SamplesV1,
    reference: anchor_train.AnchorReferenceV2, epoch: int,
    args: argparse.Namespace, device: torch.device,
) -> None:
    if epoch != 0:
        return
    candidate = _predict(model, tune, args, device).probability
    baseline = anchor_train._predict_anchor(model.anchor, tune, args, device)
    if not np.array_equal(candidate, baseline):
        raise AdvantageM2BoundedTrainingError(
            f"M1 fallbackがbit-identicalではありません: seed={reference.seed} fold={reference.fold}"
        )


def _run_fold(
    samples: m2_data.M2SamplesV1, plan: FoldPlan, variant: str, seed: int,
    reference: anchor_train.AnchorReferenceV2, args: argparse.Namespace,
    device: torch.device,
) -> tuple[PredictionV4, np.ndarray, Mapping[str, Any]]:
    fold_seed = seed + plan.eval_fold * 100
    selection = _fit_fold(
        samples, plan, variant, fold_seed, reference, args, device,
    )
    evaluation = prepare_auxiliary(
        samples.subset(samples.folds == plan.eval_fold), variant, fold_seed + 13,
    )
    prediction = _predict(selection.model, evaluation.samples, args, device)
    calibrated = legacy.calibrate(prediction.probability, reference.slope)
    path = _save_model(selection, reference, variant, seed, plan.eval_fold, args.output_root)
    record = _fold_record(
        selection, evaluation, prediction, calibrated, reference,
        variant, seed, plan, path,
    )
    return prediction, calibrated, record


def _save_model(
    selection: SelectionV4, reference: anchor_train.AnchorReferenceV2,
    variant: str, seed: int, fold: int, root: Path,
) -> Path:
    path = root / f"m2_bounded_{variant}__seed_{seed}__fold_{fold}.pt"
    return frozen._save_torch_exclusive(path, {
        "not_production": True, "diagnostic_only": True,
        "variant": variant, "seed": seed, "fold": fold,
        "model_version": model_v4.MODEL_VERSION,
        "fallback_to_m1": selection.fallback_to_m1,
        "max_shrink_fraction": model_v4.MAX_SHRINK_FRACTION,
        "anchor_model_sha256": reference.sha256,
        "state_dict": frozen._cpu_state(selection.model),
    })


def _fold_record(
    selection: SelectionV4, evaluation: PreparedAuxiliary,
    prediction: PredictionV4, calibrated: np.ndarray,
    reference: anchor_train.AnchorReferenceV2, variant: str,
    seed: int, plan: FoldPlan, path: Path,
) -> dict[str, Any]:
    samples = evaluation.samples
    return {
        "variant": variant, "seed": seed, "eval_fold": plan.eval_fold,
        "tune_fold": plan.tune_fold, "train_folds": list(plan.train_folds),
        "best_epoch": selection.best_epoch, "fallback_to_m1": selection.fallback_to_m1,
        "tune_game_equal_log_loss": selection.tune_loss,
        "auxiliary_pretrain_mse": selection.auxiliary_pretrain_mse,
        "train_auxiliary_receipt": dict(selection.train_auxiliary_receipt),
        "tune_auxiliary_receipt": dict(selection.tune_auxiliary_receipt),
        "eval_auxiliary_receipt": dict(evaluation.receipt),
        "fixed_m1_symmetric_platt_slope": reference.slope,
        "metrics_raw": legacy.metrics(samples, prediction.probability),
        "metrics_calibrated": legacy.metrics(samples, calibrated),
        "shrink_nonzero_count": int(np.count_nonzero(prediction.shrink_fraction)),
        "shrink_max": float(prediction.shrink_fraction.max(initial=0.0)),
        "anchor_model": str(reference.path), "anchor_model_sha256": reference.sha256,
        "model": path.name, "model_sha256": receipt.file_sha256(path),
        "model_version": model_v4.MODEL_VERSION,
    }


def _new_accumulator(count: int) -> AccumulatorV4:
    return AccumulatorV4(
        *[np.full(count, np.nan, dtype=np.float32) for _ in range(4)], [],
    )


def _assign_fold(
    accumulator: AccumulatorV4, indices: np.ndarray, prediction: PredictionV4,
    calibrated: np.ndarray, record: Mapping[str, Any],
) -> None:
    if np.isfinite(accumulator.raw[indices]).any():
        raise AdvantageM2BoundedTrainingError("同じOOF行へ重複代入しています")
    arrays = (prediction.probability, calibrated, prediction.auxiliary_mse_rows,
              prediction.shrink_fraction)
    if any(len(value) != len(indices) for value in arrays):
        raise AdvantageM2BoundedTrainingError("OOF fold行数が一致しません")
    accumulator.raw[indices], accumulator.calibrated[indices] = arrays[:2]
    accumulator.auxiliary_mse_rows[indices] = arrays[2]
    accumulator.shrink_fraction[indices] = arrays[3]
    accumulator.records.append(record)


def _finalize(
    samples: m2_data.M2SamplesV1, accumulator: AccumulatorV4, expected: np.ndarray,
) -> dict[str, Any]:
    arrays = (
        accumulator.raw, accumulator.calibrated,
        accumulator.auxiliary_mse_rows, accumulator.shrink_fraction,
    )
    if any(not np.array_equal(np.isfinite(value), expected) for value in arrays):
        raise AdvantageM2BoundedTrainingError("requested fold OOF coverageが一致しません")
    selected = samples.subset(expected)
    return {
        "raw": accumulator.raw, "calibrated": accumulator.calibrated,
        "auxiliary_mse_rows": accumulator.auxiliary_mse_rows,
        "shrink_fraction": accumulator.shrink_fraction,
        "records": accumulator.records,
        "metrics_raw": legacy.metrics(selected, accumulator.raw[expected]),
        "metrics_calibrated": legacy.metrics(selected, accumulator.calibrated[expected]),
    }


def _run_seed(
    samples: m2_data.M2SamplesV1, plans: Mapping[int, FoldPlan], seed: int,
    registry: Mapping[tuple[int, int], anchor_train.AnchorReferenceV2],
    args: argparse.Namespace, device: torch.device,
) -> dict[str, dict[str, Any]]:
    accumulators = {variant: _new_accumulator(len(samples.labels)) for variant in args.variants}
    for fold in args.folds:
        for variant in args.variants:
            output = _run_fold(
                samples, plans[fold], variant, seed, registry[(seed, fold)], args, device,
            )
            _assign_fold(accumulators[variant], np.flatnonzero(samples.folds == fold), *output)
    expected = np.isin(samples.folds, args.folds)
    return {
        f"m2_bounded_{name}__seed_{seed}": _finalize(samples, value, expected)
        for name, value in accumulators.items()
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    """V4 OOFを新規rootへ排他的に保存する。"""

    if args.output_root.exists():
        raise AdvantageM2BoundedTrainingError(f"出力先は新規必須です: {args.output_root}")
    samples, manifest = load_family_dataset(args.dataset_root)
    registry, anchors = anchor_train.load_anchor_registry(args.anchor_roots)
    anchor_train._required_anchors(registry, args.seeds, args.folds)
    plans = {plan.eval_fold: plan for plan in fixed_fold_plan(samples.folds)}
    args.output_root.mkdir(parents=True, exist_ok=False)
    plan = _plan(args, samples, manifest, anchors)
    plan_path = train_base._write_json_exclusive(args.output_root / "PLAN.json", plan)
    device = train_base._device(args.device)
    results: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        results.update(_run_seed(samples, plans, seed, registry, args, device))
    return _write_results(args, samples, plan, plan_path, results)


def _plan(
    args: argparse.Namespace, samples: m2_data.M2SamplesV1,
    manifest: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": PLAN_VERSION, "not_production": True, "diagnostic_only": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_sha256": manifest["dataset"]["sha256"],
        "state_count": len(samples.labels), "source_count": len(np.unique(samples.source_groups)),
        "game_count": len(np.unique(samples.game_keys)), "variants": list(args.variants),
        "seeds": list(args.seeds), "folds": list(args.folds),
        "epochs": args.epochs, "patience": args.patience,
        "auxiliary_epochs": args.auxiliary_epochs,
        "auxiliary_learning_rate": args.auxiliary_learning_rate,
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "weight_decay": args.weight_decay, "device": args.device,
        "anchor_receipts": list(anchors), "auxiliary_coefficient": AUXILIARY_COEFFICIENT,
        "win_residual": "sign_preserving_symmetric_shrink_toward_0p5",
        "max_shrink_fraction": model_v4.MAX_SHRINK_FRACTION,
        "epoch0_contract": "bit_identical_fixed_fold_m1_after_auxiliary_pretraining",
        "unsupported_contract": "bit_identical_m1",
        "integrity_fault_contract": "HOLD/FAULT_never_fallback",
        "selection_metric": "fixed_m1_slope_tune_game_equal_log_loss",
        "attack_difference_ten_percent_correction": False,
        "formal100_used": False, "hidden_reserve_used": False,
        "production_config_changed": False, "code_sha256": _code_hashes(),
    }


def _code_hashes() -> dict[str, str]:
    return {
        "trainer": receipt.file_sha256(Path(__file__)),
        "model": receipt.file_sha256(REPO_ROOT / "src/advantage_m2_bounded_auxiliary_v4.py"),
        "dataset_builder": receipt.file_sha256(
            REPO_ROOT / "scripts/build_advantage_m2_auxiliary_family_dataset_v4.py"
        ),
        "production_config": receipt.file_sha256(REPO_ROOT / "src/production_config.py"),
    }


def _write_results(
    args: argparse.Namespace, samples: m2_data.M2SamplesV1,
    plan: Mapping[str, Any], plan_path: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    predictions: dict[str, np.ndarray] = {
        "labels": samples.labels, "weights": samples.weights, "folds": samples.folds,
        "state_ids": samples.state_ids, "source_groups": samples.source_groups,
        "game_keys": samples.game_keys,
    }
    serializable: dict[str, Any] = {}
    for name, result in results.items():
        for field in ("raw", "calibrated", "auxiliary_mse_rows", "shrink_fraction"):
            predictions[f"{name}__{field}"] = result[field]
        serializable[name] = {
            key: value for key, value in result.items()
            if key not in {"raw", "calibrated", "auxiliary_mse_rows", "shrink_fraction"}
        }
    prediction_path = args.output_root / "oof_predictions.npz"
    legacy._save_npz_exclusive(prediction_path, predictions)
    report = {"format_version": TRAINING_VERSION, "not_production": True,
              "plan": dict(plan), "results": serializable}
    result_path = train_base._write_json_exclusive(args.output_root / "results.json", report)
    complete = {
        "format_version": COMPLETE_VERSION, "not_production": True,
        "plan_sha256": receipt.file_sha256(plan_path),
        "results_sha256": receipt.file_sha256(result_path),
        "predictions_sha256": receipt.file_sha256(prediction_path),
        "production_config_changed": False,
    }
    train_base._write_json_exclusive(args.output_root / "COMPLETE", complete)
    return report


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--anchor-roots", type=Path, nargs="+", default=list(anchor_train.DEFAULT_ANCHOR_ROOTS))
    parser.add_argument("--variants", type=_csv_strings, default=AUXILIARY_VARIANTS)
    parser.add_argument("--seeds", type=m2_data._csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--folds", type=m2_data._csv_ints, default=DEFAULT_FOLDS)
    parser.add_argument("--epochs", type=int, default=m2_data.DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=m2_data.DEFAULT_PATIENCE)
    parser.add_argument("--auxiliary-epochs", type=int, default=DEFAULT_AUXILIARY_EPOCHS)
    parser.add_argument("--auxiliary-learning-rate", type=float, default=DEFAULT_AUXILIARY_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=m2_data.DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=m2_data.DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=m2_data.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.variants or any(value not in AUXILIARY_VARIANTS for value in args.variants):
        parser.error("未対応または空のvariantです")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variantが重複しています")
    if not args.seeds or not args.folds or any(value not in DEFAULT_FOLDS for value in args.folds):
        parser.error("seedとfoldが不正です")
    positive = (args.epochs, args.patience, args.auxiliary_epochs, args.batch_size)
    if min(positive) <= 0:
        parser.error("epoch/patience/batch-sizeは正数必須です")
    floats = (args.auxiliary_learning_rate, args.learning_rate)
    if any(not math.isfinite(value) or value <= 0 for value in floats):
        parser.error("learning rateは有限正数必須です")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decayは有限非負必須です")


def main(argv: Sequence[str] | None = None) -> int:
    report = train(parse_args(argv))
    print(json.dumps(report["results"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
