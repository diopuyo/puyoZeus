"""同一foldのM0を固定し、causal-ledger残差だけをOOF学習する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts import train_advantage_m0_current_cnn_v1 as base
from scripts import train_advantage_m1_causal_ledger_v1 as legacy
from src.advantage_m0_current_cnn_v1 import AdvantageM0CurrentCNNV2
from src.advantage_m1_causal_ledger_v1 import (
    LEDGER_MODE,
    AdvantageM1FrozenResidualCNNV1,
)
from src.event_provisional_oof_v1 import FoldPlan, fixed_fold_plan


TRAINING_VERSION = "advantage-m1-frozen-residual-oof-training/v1"
PLAN_VERSION = "advantage-m1-frozen-residual-oof-plan/v1"
COMPLETE_VERSION = "advantage-m1-frozen-residual-oof-complete/v1"
RESIDUAL_VARIANTS = ("values", "masks", "values_and_masks", "random_control")
DEFAULT_SEEDS = legacy.DEFAULT_SEEDS
REPO_ROOT = Path(__file__).resolve().parents[1]


class FrozenResidualTrainingError(RuntimeError):
    """凍結M0比較または成果物の契約違反。"""


@dataclass(frozen=True, slots=True)
class M0Anchor:
    """一つのseed/foldで一度だけ選択したM0。"""

    model: AdvantageM0CurrentCNNV2
    slope: float
    best_epoch: int
    tune_raw: Mapping[str, float]
    tune_calibrated: Mapping[str, float]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class ResidualSelection:
    """epoch 0を含めて選択した残差model。"""

    model: AdvantageM1FrozenResidualCNNV1
    best_epoch: int
    fallback_to_m0: bool
    tune_raw: Mapping[str, float]
    tune_calibrated: Mapping[str, float]
    m0_state_sha256: str


@dataclass(frozen=True, slots=True)
class FoldOutput:
    """一variant・一foldのOOF予測とreceipt。"""

    raw: np.ndarray
    calibrated: np.ndarray
    record: Mapping[str, Any]


@dataclass(slots=True)
class OOFAccumulator:
    """fold別予測を元行順へ戻す作業領域。"""

    raw: np.ndarray
    calibrated: np.ndarray
    records: list[Mapping[str, Any]]


def model_state_sha256(model: torch.nn.Module) -> str:
    """tensor名・型・shape・値を固定順でhash化する。"""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _fit_m0_anchor(
    samples: legacy.CanonicalSamples, plan: FoldPlan, seed: int,
    args: argparse.Namespace, device: torch.device,
) -> M0Anchor:
    fitted = legacy._fit_fold(samples, plan, "m0", seed, args, device)
    model, slope, epoch, tune_raw, tune_calibrated = fitted
    if not isinstance(model, AdvantageM0CurrentCNNV2):
        raise FrozenResidualTrainingError("M0 anchorのmodel型が不正です")
    model.eval()
    return M0Anchor(
        model, slope, epoch, tune_raw, tune_calibrated,
        model_state_sha256(model),
    )


def _ledger_mode(variant: str) -> LEDGER_MODE:
    if variant in {"values", "masks", "values_and_masks"}:
        return variant
    if variant == "random_control":
        return "values_and_masks"
    raise FrozenResidualTrainingError(f"未対応residual variantです: {variant}")


def _new_residual_model(
    anchor: M0Anchor, variant: str, seed: int, device: torch.device,
) -> AdvantageM1FrozenResidualCNNV1:
    base._set_seed(seed)
    cloned_m0 = AdvantageM0CurrentCNNV2().to(device)
    cloned_m0.load_state_dict(anchor.model.state_dict())
    model = AdvantageM1FrozenResidualCNNV1(cloned_m0, _ledger_mode(variant)).to(device)
    _assert_m0_unchanged(model, anchor.state_sha256)
    return model


def _trainable_parameters(
    model: AdvantageM1FrozenResidualCNNV1,
) -> list[torch.nn.Parameter]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    m0_ids = {id(parameter) for parameter in model.m0.parameters()}
    if not trainable or any(id(parameter) in m0_ids for parameter in trainable):
        raise FrozenResidualTrainingError("optimizerへ凍結M0 parameterが混入しています")
    return trainable


def _assert_m0_unchanged(
    model: AdvantageM1FrozenResidualCNNV1, expected_sha256: str,
) -> None:
    if model_state_sha256(model.m0) != expected_sha256:
        raise FrozenResidualTrainingError("学習中に凍結M0 parameterが変化しました")


def _variant_samples(
    samples: legacy.CanonicalSamples, variant: str, seed: int,
) -> legacy.CanonicalSamples:
    if variant != "random_control":
        return samples
    return randomized_ledger_values_within_availability(samples, seed)


def randomized_ledger_values_within_availability(
    samples: legacy.CanonicalSamples, seed: int,
) -> legacy.CanonicalSamples:
    """availabilityを固定し、同一欠測パターン内で値だけを並べ替える。"""

    count = len(samples.labels)
    if count < 2:
        raise FrozenResidualTrainingError("random controlには2行以上必要です")
    patterns = samples.ledger_availability.reshape(count, -1)
    _, strata = np.unique(patterns, axis=0, return_inverse=True)
    randomized = np.array(samples.ledger_values, copy=True)
    rng = np.random.default_rng(seed)
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        if len(indices) < 2:
            continue
        permutation = rng.permutation(indices)
        if np.array_equal(permutation, indices):
            permutation = np.roll(permutation, 1)
        randomized[indices] = samples.ledger_values[permutation]
    return replace(samples, ledger_values=np.ascontiguousarray(randomized))


def _predict_m0(
    model: AdvantageM0CurrentCNNV2, samples: legacy.CanonicalSamples,
    args: argparse.Namespace, device: torch.device,
) -> np.ndarray:
    return legacy._predict(model, samples, args.batch_size, device, "m0")


def _predict_residual(
    model: AdvantageM1FrozenResidualCNNV1, samples: legacy.CanonicalSamples,
    args: argparse.Namespace, device: torch.device,
) -> np.ndarray:
    return legacy._predict(model, samples, args.batch_size, device, "m1_frozen")


def _residual_loader(
    samples: legacy.CanonicalSamples, args: argparse.Namespace, seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        legacy.CanonicalDataset(samples, augment=True, seed=seed),
        batch_size=args.batch_size, shuffle=True, generator=generator,
    )


def _initial_residual_selection(
    model: AdvantageM1FrozenResidualCNNV1, tune: legacy.CanonicalSamples,
    anchor: M0Anchor, args: argparse.Namespace, device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    baseline = _predict_m0(anchor.model, tune, args, device)
    candidate = _predict_residual(model, tune, args, device)
    if not np.array_equal(candidate, baseline):
        raise FrozenResidualTrainingError("epoch 0がM0とbit-identicalではありません")
    calibrated = legacy.calibrate(baseline, anchor.slope)
    loss = legacy.metrics(tune, calibrated)["game_equal_log_loss"]
    return _cpu_state(model), loss


def _fit_residual(
    samples: legacy.CanonicalSamples, plan: FoldPlan, variant: str, seed: int,
    anchor: M0Anchor, args: argparse.Namespace, device: torch.device,
) -> ResidualSelection:
    train = samples.subset(np.isin(samples.folds, plan.train_folds))
    tune = samples.subset(samples.folds == plan.tune_fold)
    train = _variant_samples(train, variant, seed + 11)
    tune = _variant_samples(tune, variant, seed + 12)
    model = _new_residual_model(anchor, variant, seed, device)
    best_state, best_loss = _initial_residual_selection(model, tune, anchor, args, device)
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loader = _residual_loader(train, args, seed)
    best_epoch, stale = 0, 0
    for epoch in range(1, args.epochs + 1):
        legacy._train_epoch(model, loader, optimizer, device, "m1_frozen")
        _assert_m0_unchanged(model, anchor.state_sha256)
        probability = _predict_residual(model, tune, args, device)
        loss = legacy.metrics(tune, legacy.calibrate(probability, anchor.slope))[
            "game_equal_log_loss"
        ]
        if loss < best_loss:
            best_state, best_loss, best_epoch, stale = _cpu_state(model), loss, epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    return _restore_selection(model, best_state, best_epoch, tune, anchor, args, device)


def _restore_selection(
    model: AdvantageM1FrozenResidualCNNV1, state: Mapping[str, torch.Tensor],
    epoch: int, tune: legacy.CanonicalSamples, anchor: M0Anchor,
    args: argparse.Namespace, device: torch.device,
) -> ResidualSelection:
    model.load_state_dict(state)
    _assert_m0_unchanged(model, anchor.state_sha256)
    raw = _predict_residual(model, tune, args, device)
    calibrated = legacy.calibrate(raw, anchor.slope)
    if epoch == 0:
        baseline = _predict_m0(anchor.model, tune, args, device)
        if not np.array_equal(raw, baseline):
            raise FrozenResidualTrainingError("M0 fallback予測がbit-identicalではありません")
    return ResidualSelection(
        model=model, best_epoch=epoch, fallback_to_m0=epoch == 0,
        tune_raw=legacy.metrics(tune, raw),
        tune_calibrated=legacy.metrics(tune, calibrated),
        m0_state_sha256=model_state_sha256(model.m0),
    )


def _save_torch_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    with path.open("xb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
    return path


def _base_fold_output(
    anchor: M0Anchor, evaluation: legacy.CanonicalSamples, seed: int,
    plan: FoldPlan, args: argparse.Namespace, device: torch.device,
) -> tuple[FoldOutput, str, str]:
    raw = _predict_m0(anchor.model, evaluation, args, device)
    calibrated = legacy.calibrate(raw, anchor.slope)
    name = f"m0__seed_{seed}__fold_{plan.eval_fold}.pt"
    path = _save_torch_exclusive(args.output_root / name, {
        "variant": "m0", "seed": seed, "fold": plan.eval_fold,
        "model_version": anchor.model.model_version,
        "m0_state_sha256": anchor.state_sha256,
        "state_dict": _cpu_state(anchor.model), "symmetric_platt_slope": anchor.slope,
    })
    file_hash = base.file_sha256(path)
    record = _base_record(anchor, evaluation, raw, name, file_hash, plan)
    return FoldOutput(raw, calibrated, record), name, file_hash


def _base_record(
    anchor: M0Anchor, evaluation: legacy.CanonicalSamples, raw: np.ndarray,
    model_name: str, model_sha256: str, plan: FoldPlan,
) -> dict[str, Any]:
    return {
        "eval_fold": plan.eval_fold, "tune_fold": plan.tune_fold,
        "train_folds": list(plan.train_folds), "best_epoch": anchor.best_epoch,
        "symmetric_platt_slope": anchor.slope,
        "tune_raw": dict(anchor.tune_raw),
        "tune_calibrated": dict(anchor.tune_calibrated),
        "eval_raw": legacy.metrics(evaluation, raw),
        "eval_calibrated": legacy.metrics(evaluation, legacy.calibrate(raw, anchor.slope)),
        "model": model_name, "model_sha256": model_sha256,
        "m0_state_sha256": anchor.state_sha256,
    }


def _residual_fold_output(
    selection: ResidualSelection, anchor: M0Anchor,
    evaluation: legacy.CanonicalSamples, baseline: FoldOutput, variant: str,
    seed: int, plan: FoldPlan, args: argparse.Namespace, device: torch.device,
    m0_model_name: str, m0_model_sha256: str,
) -> FoldOutput:
    if selection.fallback_to_m0:
        raw, calibrated = baseline.raw.copy(), baseline.calibrated.copy()
    else:
        raw = _predict_residual(selection.model, evaluation, args, device)
        calibrated = legacy.calibrate(raw, anchor.slope)
    name = f"m1_frozen_{variant}__seed_{seed}__fold_{plan.eval_fold}.pt"
    path = _save_torch_exclusive(args.output_root / name, {
        "variant": variant, "seed": seed, "fold": plan.eval_fold,
        "model_version": selection.model.model_version,
        "ledger_mode": _ledger_mode(variant),
        "fixed_m0_symmetric_platt_slope": anchor.slope,
        "fallback_to_m0": selection.fallback_to_m0,
        "m0_state_sha256": anchor.state_sha256,
        "state_dict": _cpu_state(selection.model),
    })
    record = _residual_record(
        selection, anchor, evaluation, raw, name, base.file_sha256(path), plan,
        m0_model_name, m0_model_sha256,
    )
    return FoldOutput(raw, calibrated, record)


def _residual_record(
    selection: ResidualSelection, anchor: M0Anchor,
    evaluation: legacy.CanonicalSamples, raw: np.ndarray,
    model_name: str, model_sha256: str, plan: FoldPlan,
    m0_model_name: str, m0_model_sha256: str,
) -> dict[str, Any]:
    return {
        "eval_fold": plan.eval_fold, "tune_fold": plan.tune_fold,
        "train_folds": list(plan.train_folds), "best_epoch": selection.best_epoch,
        "fallback_to_m0": selection.fallback_to_m0,
        "fixed_m0_symmetric_platt_slope": anchor.slope,
        "tune_raw": dict(selection.tune_raw),
        "tune_calibrated": dict(selection.tune_calibrated),
        "eval_raw": legacy.metrics(evaluation, raw),
        "eval_calibrated": legacy.metrics(evaluation, legacy.calibrate(raw, anchor.slope)),
        "model": model_name, "model_sha256": model_sha256,
        "m0_model": m0_model_name, "m0_model_sha256": m0_model_sha256,
        "m0_state_sha256_before": anchor.state_sha256,
        "m0_state_sha256_after": selection.m0_state_sha256,
    }


def _run_fold(
    samples: legacy.CanonicalSamples, plan: FoldPlan, seed: int,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, FoldOutput]:
    fold_seed = seed + plan.eval_fold * 100
    anchor = _fit_m0_anchor(samples, plan, fold_seed, args, device)
    evaluation = samples.subset(samples.folds == plan.eval_fold)
    baseline, m0_name, m0_hash = _base_fold_output(
        anchor, evaluation, seed, plan, args, device,
    )
    output = {"m0": baseline}
    for variant in args.variants:
        selection = _fit_residual(
            samples, plan, variant, fold_seed, anchor, args, device,
        )
        candidate_eval = _variant_samples(evaluation, variant, fold_seed + 13)
        output[_result_variant(variant)] = _residual_fold_output(
            selection, anchor, candidate_eval, baseline, variant, seed, plan,
            args, device, m0_name, m0_hash,
        )
    return output


def _result_variant(variant: str) -> str:
    return f"m1_frozen_{variant}"


def _new_accumulators(
    row_count: int, variants: Sequence[str],
) -> dict[str, OOFAccumulator]:
    names = ("m0", *(_result_variant(value) for value in variants))
    return {
        name: OOFAccumulator(
            np.full(row_count, np.nan, dtype=np.float32),
            np.full(row_count, np.nan, dtype=np.float32), [],
        )
        for name in names
    }


def _assign_fold(
    accumulator: OOFAccumulator, indices: np.ndarray, output: FoldOutput,
) -> None:
    if np.isfinite(accumulator.raw[indices]).any():
        raise FrozenResidualTrainingError("同じOOF行へ複数回書き込もうとしました")
    accumulator.raw[indices] = output.raw
    accumulator.calibrated[indices] = output.calibrated
    accumulator.records.append(output.record)


def _finalize_accumulator(
    samples: legacy.CanonicalSamples, accumulator: OOFAccumulator,
) -> dict[str, Any]:
    mask = np.isfinite(accumulator.raw) & np.isfinite(accumulator.calibrated)
    if not mask.any():
        raise FrozenResidualTrainingError("完成したOOF予測がありません")
    selected = samples.subset(mask)
    return {
        "raw": accumulator.raw, "calibrated": accumulator.calibrated,
        "records": accumulator.records,
        "metrics_raw": legacy.metrics(selected, accumulator.raw[mask]),
        "metrics_calibrated": legacy.metrics(selected, accumulator.calibrated[mask]),
    }


def _run_seed(
    samples: legacy.CanonicalSamples, plans: Mapping[int, FoldPlan], seed: int,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, dict[str, Any]]:
    accumulators = _new_accumulators(len(samples.labels), args.variants)
    for fold in args.folds:
        outputs = _run_fold(samples, plans[fold], seed, args, device)
        indices = np.flatnonzero(samples.folds == fold)
        for name, output in outputs.items():
            _assign_fold(accumulators[name], indices, output)
    return {
        f"{name}__seed_{seed}": _finalize_accumulator(samples, accumulator)
        for name, accumulator in accumulators.items()
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    """新規rootだけへ凍結M0と残差OOFを排他的に保存する。"""

    samples, manifest = legacy.load_canonical_dataset(args.dataset_root)
    plans = {plan.eval_fold: plan for plan in fixed_fold_plan(samples.folds)}
    args.output_root.mkdir(parents=True, exist_ok=False)
    plan = _plan_receipt(args, samples, manifest)
    plan_path = base._write_json_exclusive(args.output_root / "PLAN.json", plan)
    device = base._device(args.device)
    results: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        results.update(_run_seed(samples, plans, seed, args, device))
    return _write_results(args, samples, manifest, plan, plan_path, results)


def _plan_receipt(
    args: argparse.Namespace, samples: legacy.CanonicalSamples,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": PLAN_VERSION, "not_production": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_sha256": manifest["dataset"]["sha256"],
        "state_count": len(samples.labels),
        "game_count": len(np.unique(samples.game_keys)),
        "source_count": len(np.unique(samples.source_groups)),
        "variants": list(args.variants), "seeds": list(args.seeds),
        "folds": list(args.folds), "epochs": args.epochs,
        "patience": args.patience, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "fixed_outer_folds": 6,
        "fold_policy": "eval k, tune k+1, train remaining four",
        "m0_anchor_policy": "fit once per seed/fold then freeze and distribute",
        "optimizer_policy": "requires_grad parameters only; M0 hash remains exact",
        "selection_policy": "epoch0 M0 fallback unless fixed-slope tune logloss improves",
        "calibration_policy": "all residual variants reuse the paired M0 slope",
        "code_sha256": _code_hashes(),
    }


def _code_hashes() -> dict[str, str]:
    return {
        "trainer": base.file_sha256(Path(__file__)),
        "legacy_trainer": base.file_sha256(
            REPO_ROOT / "scripts/train_advantage_m1_causal_ledger_v1.py"
        ),
        "m0": base.file_sha256(REPO_ROOT / "src/advantage_m0_current_cnn_v1.py"),
        "m1": base.file_sha256(REPO_ROOT / "src/advantage_m1_causal_ledger_v1.py"),
        "production_config": base.file_sha256(REPO_ROOT / "src/production_config.py"),
    }


def _write_results(
    args: argparse.Namespace, samples: legacy.CanonicalSamples,
    manifest: Mapping[str, Any], plan: Mapping[str, Any], plan_path: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    predictions: dict[str, np.ndarray] = {"labels": samples.labels, "folds": samples.folds}
    serializable: dict[str, Any] = {}
    for key, result in results.items():
        predictions[f"{key}__raw"] = result["raw"]
        predictions[f"{key}__calibrated"] = result["calibrated"]
        serializable[key] = {
            name: value for name, value in result.items()
            if name not in {"raw", "calibrated"}
        }
    prediction_path = args.output_root / "oof_predictions.npz"
    legacy._save_npz_exclusive(prediction_path, predictions)
    report = _report(manifest, plan, serializable)
    report_path = base._write_json_exclusive(args.output_root / "results.json", report)
    _write_complete(args.output_root, plan_path, report_path, prediction_path)
    return report


def _report(
    manifest: Mapping[str, Any], plan: Mapping[str, Any],
    results: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": TRAINING_VERSION, "not_production": True,
        "dataset_source_count": manifest["source_count"],
        "plan": dict(plan), "results": dict(results),
    }


def _write_complete(
    root: Path, plan_path: Path, report_path: Path, prediction_path: Path,
) -> None:
    base._write_json_exclusive(root / "COMPLETE", {
        "format_version": COMPLETE_VERSION,
        "plan_sha256": base.file_sha256(plan_path),
        "results_sha256": base.file_sha256(report_path),
        "predictions_sha256": base.file_sha256(prediction_path),
    })


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _csv_strings(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("整数のカンマ区切りが必要です") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", type=_csv_strings, default=RESIDUAL_VARIANTS)
    parser.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--folds", type=_csv_ints, default=tuple(range(1, 7)))
    parser.add_argument("--epochs", type=int, default=legacy.DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=legacy.DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=legacy.DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=legacy.DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=legacy.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.variants or len(set(args.variants)) != len(args.variants):
        parser.error("variantsは重複なしの非空指定が必要です")
    if any(value not in RESIDUAL_VARIANTS for value in args.variants):
        parser.error(f"variantsは{RESIDUAL_VARIANTS}から指定してください")
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or not args.folds or len(set(args.folds)) != len(args.folds)):
        parser.error("seeds/foldsは重複なしの非空指定が必要です")
    if any(value not in range(1, 7) for value in args.folds):
        parser.error("foldsは1..6で指定してください")
    if any(value <= 0 for value in (args.epochs, args.patience, args.batch_size)):
        parser.error("epochs/patience/batch-sizeは正整数必須です")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rateは正、weight-decayは非負必須です")


def main(argv: Sequence[str] | None = None) -> int:
    train(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
