"""凍結M1をepoch 0に保持してM2補助共有encoder残差をpaired OOF学習する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import train_advantage_m0_current_cnn_v1 as base
from scripts import train_advantage_m1_causal_ledger_v1 as legacy
from scripts import train_advantage_m1_frozen_residual_v1 as frozen
from scripts import train_advantage_m1_zero_counterfactual_v3 as m1_trainer
from scripts import train_advantage_m2_auxiliary_v1 as v1
from src import advantage_m2_auxiliary_cnn_v1 as m2
from src.advantage_m0_current_cnn_v1 import AdvantageM0CurrentCNNV2
from src.advantage_m1_zero_counterfactual_v3 import AdvantageM1ZeroCounterfactualV3
from src.event_provisional_oof_v1 import FoldPlan, fixed_fold_plan


TRAINING_VERSION = "advantage-m2-anchored-auxiliary-fixed-oof-training/v2"
PLAN_VERSION = "advantage-m2-anchored-auxiliary-fixed-oof-plan/v2"
COMPLETE_VERSION = "advantage-m2-anchored-auxiliary-fixed-oof-complete/v2"
ANCHOR_RESULT_VARIANT = "m1_zero_values_and_masks"
ANCHOR_MODEL_VARIANT = "values_and_masks"
DEFAULT_ANCHOR_ROOTS = (
    Path("data/verify/advantage_m1_zero_counterfactual_46v_allfolds_seed20260904_2026-09-05_v3_finalized_primary6_merged"),
    Path("data/verify/advantage_m1_zero_counterfactual_46v_allfolds_seeds20260905_20260906_2026-09-05_v3_finalized_primary6_merged"),
)
DEFAULT_LEARNING_RATE = 1e-4
REPO_ROOT = Path(__file__).resolve().parents[1]


class AdvantageM2AnchoredTrainingError(RuntimeError):
    """M1 anchor、epoch 0 fallback、paired OOF契約に違反した。"""


@dataclass(frozen=True, slots=True)
class AnchorReferenceV2:
    seed: int
    fold: int
    path: Path
    sha256: str
    slope: float
    result_root: Path
    result_sha256: str


@dataclass(frozen=True, slots=True)
class SelectionV2:
    model: m2.AdvantageM2AnchoredAuxiliaryCNNV2
    best_epoch: int
    fallback_to_m1: bool
    initial_state_sha256: str
    tune_raw: Mapping[str, float]
    tune_calibrated: Mapping[str, float]
    tune_auxiliary_mse: float


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM2AnchoredTrainingError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise AdvantageM2AnchoredTrainingError(f"JSON objectではありません: {path}")
    return value


def load_anchor_registry(
    roots: Sequence[Path],
) -> tuple[dict[tuple[int, int], AnchorReferenceV2], list[dict[str, Any]]]:
    """merged M1結果からseed/fold別の検証済みcheckpoint台帳を作る。"""

    registry: dict[tuple[int, int], AnchorReferenceV2] = {}
    receipts: list[dict[str, Any]] = []
    for root in roots:
        result_path = root / "results.json"
        complete = _read_json(root / "COMPLETE")
        if complete.get("results_sha256") != base.file_sha256(result_path):
            raise AdvantageM2AnchoredTrainingError(f"M1 results hashが不一致です: {root}")
        results = _read_json(result_path)
        _add_root_records(registry, results, root, complete["results_sha256"])
        receipts.append({
            "root": str(root.resolve()), "results_sha256": complete["results_sha256"],
            "complete_sha256": base.file_sha256(root / "COMPLETE"),
        })
    return registry, receipts


def _add_root_records(
    registry: dict[tuple[int, int], AnchorReferenceV2], results: Mapping[str, Any],
    root: Path, result_sha256: str,
) -> None:
    for name, value in results.get("results", {}).items():
        if not str(name).startswith(f"{ANCHOR_RESULT_VARIANT}__seed_"):
            continue
        if not isinstance(value, Mapping):
            raise AdvantageM2AnchoredTrainingError("M1 variant結果がobjectではありません")
        seed = int(str(name).rsplit("_", 1)[-1])
        for record in value.get("records", ()):
            reference = _anchor_reference(record, seed, root, result_sha256)
            key = (reference.seed, reference.fold)
            if key in registry:
                raise AdvantageM2AnchoredTrainingError(f"M1 anchorが重複しています: {key}")
            registry[key] = reference


def _anchor_reference(
    record: Mapping[str, Any], seed: int, root: Path, result_sha256: str,
) -> AnchorReferenceV2:
    checkpoint = record.get("checkpoint_reference")
    if not isinstance(checkpoint, Mapping):
        raise AdvantageM2AnchoredTrainingError("M1 checkpoint referenceがありません")
    path = Path(str(checkpoint.get("path", "")))
    expected = str(checkpoint.get("sha256", ""))
    if not path.is_file() or base.file_sha256(path) != expected:
        raise AdvantageM2AnchoredTrainingError(f"M1 checkpoint hashが不一致です: {path}")
    return AnchorReferenceV2(
        seed=seed, fold=int(record["eval_fold"]), path=path,
        sha256=expected, slope=float(record["fixed_m0_symmetric_platt_slope"]),
        result_root=root, result_sha256=result_sha256,
    )


def _required_anchors(
    registry: Mapping[tuple[int, int], AnchorReferenceV2],
    seeds: Sequence[int], folds: Sequence[int],
) -> None:
    required = {(seed, fold) for seed in seeds for fold in folds}
    missing = sorted(required - set(registry))
    if missing:
        raise AdvantageM2AnchoredTrainingError(f"必要なM1 anchorがありません: {missing}")


def _load_anchor(
    reference: AnchorReferenceV2, device: torch.device,
) -> AdvantageM1ZeroCounterfactualV3:
    checkpoint = torch.load(reference.path, map_location=device, weights_only=False)
    _validate_checkpoint(checkpoint, reference)
    model = AdvantageM1ZeroCounterfactualV3(
        AdvantageM0CurrentCNNV2(), ANCHOR_MODEL_VARIANT,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _validate_checkpoint(
    checkpoint: Mapping[str, Any], reference: AnchorReferenceV2,
) -> None:
    expected = (
        checkpoint.get("not_production") is True,
        checkpoint.get("model_version") == m1_trainer.ZERO_COUNTERFACTUAL_MODEL_VERSION,
        checkpoint.get("input_schema_version") == m1_trainer.M1_INPUT_SCHEMA_VERSION,
        checkpoint.get("variant") == ANCHOR_MODEL_VARIANT,
        int(checkpoint.get("seed", -1)) == reference.seed,
        int(checkpoint.get("fold", -1)) == reference.fold,
    )
    if not all(expected) or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise AdvantageM2AnchoredTrainingError("M1 checkpoint metadataが不正です")


def _trainable_parameters(
    model: m2.AdvantageM2AnchoredAuxiliaryCNNV2,
) -> list[torch.nn.Parameter]:
    parameters = [value for value in model.parameters() if value.requires_grad]
    anchor_ids = {id(value) for value in model.anchor.parameters()}
    if not parameters or any(id(value) in anchor_ids for value in parameters):
        raise AdvantageM2AnchoredTrainingError("optimizerへ凍結M1が混入しています")
    return parameters


def _predict(
    model: m2.AdvantageM2AnchoredAuxiliaryCNNV2, samples: v1.M2SamplesV1,
    args: argparse.Namespace, device: torch.device,
) -> v1.PredictionV1:
    return v1._predict(model, samples, args, device)


def _initial_selection(
    model: m2.AdvantageM2AnchoredAuxiliaryCNNV2,
    tune: v1.M2SamplesV1, reference: AnchorReferenceV2,
    args: argparse.Namespace, device: torch.device,
) -> tuple[dict[str, torch.Tensor], float, str]:
    prediction = _predict(model, tune, args, device)
    baseline = _predict_anchor(model.anchor, tune, args, device)
    if not np.array_equal(prediction.probability, baseline):
        raise AdvantageM2AnchoredTrainingError("epoch 0がM1とbit-identicalではありません")
    calibrated = legacy.calibrate(baseline, reference.slope)
    loss = legacy.metrics(tune, calibrated)["game_equal_log_loss"]
    return frozen._cpu_state(model), loss, frozen.model_state_sha256(model)


def _predict_anchor(
    anchor: AdvantageM1ZeroCounterfactualV3, samples: v1.M2SamplesV1,
    args: argparse.Namespace, device: torch.device,
) -> np.ndarray:
    output: list[np.ndarray] = []
    anchor.eval()
    with torch.no_grad():
        for batch in v1._loader(samples, args, augment=False, seed=0):
            moved = tuple(value.to(device) for value in batch)
            result = anchor(moved[0], moved[1], moved[4], moved[5])
            output.append(result.raw_probability.cpu().numpy())
    return np.concatenate(output)


def _fit_fold(
    samples: v1.M2SamplesV1, plan: FoldPlan, coefficient: float, seed: int,
    reference: AnchorReferenceV2, args: argparse.Namespace, device: torch.device,
) -> SelectionV2:
    train = samples.subset(np.isin(samples.folds, plan.train_folds))
    tune = samples.subset(samples.folds == plan.tune_fold)
    base._set_seed(seed)
    model = m2.AdvantageM2AnchoredAuxiliaryCNNV2(
        _load_anchor(reference, device),
    ).to(device)
    best, best_loss, initial_sha = _initial_selection(model, tune, reference, args, device)
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    best_epoch, stale = 0, 0
    loader = v1._loader(train, args, augment=True, seed=seed)
    for epoch in range(1, args.epochs + 1):
        losses = v1._train_epoch(model, loader, optimizer, device, coefficient)
        prediction = _predict(model, tune, args, device)
        calibrated = legacy.calibrate(prediction.probability, reference.slope)
        tune_loss = legacy.metrics(tune, calibrated)["game_equal_log_loss"]
        v1._print_epoch(coefficient, seed, plan.eval_fold, epoch, losses, tune_loss)
        if tune_loss < best_loss:
            best, best_loss, best_epoch, stale = frozen._cpu_state(model), tune_loss, epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    return _restore_selection(model, best, best_epoch, initial_sha, tune, reference, args, device)


def _restore_selection(
    model: m2.AdvantageM2AnchoredAuxiliaryCNNV2,
    state: Mapping[str, torch.Tensor], epoch: int, initial_sha: str,
    tune: v1.M2SamplesV1, reference: AnchorReferenceV2,
    args: argparse.Namespace, device: torch.device,
) -> SelectionV2:
    model.load_state_dict(state)
    prediction = _predict(model, tune, args, device)
    calibrated = legacy.calibrate(prediction.probability, reference.slope)
    if epoch == 0:
        baseline = _predict_anchor(model.anchor, tune, args, device)
        if not np.array_equal(prediction.probability, baseline):
            raise AdvantageM2AnchoredTrainingError("M1 fallbackがbit-identicalではありません")
    return SelectionV2(
        model, epoch, epoch == 0, initial_sha,
        legacy.metrics(tune, prediction.probability), legacy.metrics(tune, calibrated),
        v1._weighted_auxiliary_mse(tune, prediction.auxiliary_mse_rows),
    )


def _run_fold(
    samples: v1.M2SamplesV1, plan: FoldPlan, coefficient: float, seed: int,
    reference: AnchorReferenceV2, args: argparse.Namespace, device: torch.device,
) -> tuple[v1.PredictionV1, np.ndarray, Mapping[str, Any]]:
    fold_seed = seed + plan.eval_fold * 100
    selection = _fit_fold(
        samples, plan, coefficient, fold_seed, reference, args, device,
    )
    evaluation = samples.subset(samples.folds == plan.eval_fold)
    prediction = _predict(selection.model, evaluation, args, device)
    calibrated = legacy.calibrate(prediction.probability, reference.slope)
    name = f"{v1._variant_name(coefficient)}__seed_{seed}__fold_{plan.eval_fold}.pt"
    path = frozen._save_torch_exclusive(args.output_root / name, {
        "not_production": True, "coefficient": coefficient, "seed": seed,
        "fold": plan.eval_fold, "model_version": m2.M2_ANCHORED_MODEL_VERSION,
        "input_schema_version": m2.M2_INPUT_SCHEMA_VERSION,
        "fixed_m1_symmetric_platt_slope": reference.slope,
        "fallback_to_m1": selection.fallback_to_m1,
        "anchor_model_sha256": reference.sha256,
        "state_dict": frozen._cpu_state(selection.model),
    })
    record = _fold_record(
        selection, evaluation, prediction, calibrated, coefficient,
        seed, plan, reference, name, base.file_sha256(path),
    )
    return prediction, calibrated, record


def _fold_record(
    selection: SelectionV2, evaluation: v1.M2SamplesV1,
    prediction: v1.PredictionV1, calibrated: np.ndarray, coefficient: float,
    seed: int, plan: FoldPlan, reference: AnchorReferenceV2,
    model_name: str, model_sha256: str,
) -> dict[str, Any]:
    return {
        "coefficient": coefficient, "seed": seed,
        "eval_fold": plan.eval_fold, "tune_fold": plan.tune_fold,
        "train_folds": list(plan.train_folds), "best_epoch": selection.best_epoch,
        "fallback_to_m1": selection.fallback_to_m1,
        "initial_state_sha256": selection.initial_state_sha256,
        "fixed_m1_symmetric_platt_slope": reference.slope,
        "tune_raw": dict(selection.tune_raw),
        "tune_calibrated": dict(selection.tune_calibrated),
        "tune_auxiliary_mse": selection.tune_auxiliary_mse,
        "eval_raw": legacy.metrics(evaluation, prediction.probability),
        "eval_calibrated": legacy.metrics(evaluation, calibrated),
        "eval_auxiliary_mse": v1._weighted_auxiliary_mse(
            evaluation, prediction.auxiliary_mse_rows,
        ),
        "anchor_model": str(reference.path), "anchor_model_sha256": reference.sha256,
        "anchor_results_root": str(reference.result_root.resolve()),
        "anchor_results_sha256": reference.result_sha256,
        "model": model_name, "model_sha256": model_sha256,
        "model_version": m2.M2_ANCHORED_MODEL_VERSION,
        "input_schema_version": m2.M2_INPUT_SCHEMA_VERSION,
    }


def _run_seed(
    samples: v1.M2SamplesV1, plans: Mapping[int, FoldPlan], seed: int,
    registry: Mapping[tuple[int, int], AnchorReferenceV2],
    args: argparse.Namespace, device: torch.device,
) -> dict[str, dict[str, Any]]:
    accumulators = v1._new_accumulators(len(samples.labels))
    for fold in args.folds:
        reference = registry[(seed, fold)]
        for coefficient in args.coefficients:
            output = _run_fold(
                samples, plans[fold], coefficient, seed, reference, args, device,
            )
            indices = np.flatnonzero(samples.folds == fold)
            v1._assign_fold(accumulators[v1._variant_name(coefficient)], indices, *output)
    expected = np.isin(samples.folds, args.folds)
    selected_names = {v1._variant_name(value) for value in args.coefficients}
    return {
        f"{name}__seed_{seed}": v1._finalize(samples, value, expected)
        for name, value in accumulators.items() if name in selected_names
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    """M1 anchored M2 paired OOFを新規rootへ排他的に保存する。"""

    if args.output_root.exists():
        raise AdvantageM2AnchoredTrainingError(f"出力先は新規必須です: {args.output_root}")
    samples, manifest = v1.load_m2_dataset(args.dataset_root)
    registry, anchor_receipts = load_anchor_registry(args.anchor_roots)
    _required_anchors(registry, args.seeds, args.folds)
    plans = {plan.eval_fold: plan for plan in fixed_fold_plan(samples.folds)}
    args.output_root.mkdir(parents=True, exist_ok=False)
    plan = _plan(args, samples, manifest, anchor_receipts)
    plan_path = base._write_json_exclusive(args.output_root / "PLAN.json", plan)
    device = base._device(args.device)
    results: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        results.update(_run_seed(samples, plans, seed, registry, args, device))
    return _write_results(args, samples, manifest, plan, plan_path, results)


def _plan(
    args: argparse.Namespace, samples: v1.M2SamplesV1, manifest: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
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
        "anchor_variant": ANCHOR_RESULT_VARIANT, "anchor_receipts": list(anchors),
        "epoch0_contract": "bit_identical_fixed_fold_m1_anchor",
        "fallback_policy": "no_tune_gain_returns_bit_identical_m1_anchor",
        "paired_protocol": "same_rows_folds_seed_anchor_initialization_batch_order",
        "selection_metric": "fixed_anchor_slope_tune_game_equal_log_loss",
        "auxiliary_role": "training_only_shared_encoder_not_win_probability_input",
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
    args: argparse.Namespace, samples: v1.M2SamplesV1, manifest: Mapping[str, Any],
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-roots", type=Path, nargs="+", default=list(DEFAULT_ANCHOR_ROOTS))
    parser.add_argument("--coefficients", type=v1._csv_floats, default=v1.DEFAULT_COEFFICIENTS)
    parser.add_argument("--seeds", type=v1._csv_ints, default=v1.DEFAULT_SEEDS)
    parser.add_argument("--folds", type=v1._csv_ints, default=tuple(range(1, 7)))
    parser.add_argument("--epochs", type=int, default=v1.DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=v1.DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=v1.DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=v1.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    v1._validate_args(parser, args)
    if not args.anchor_roots:
        parser.error("anchor rootを1つ以上指定してください")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = train(parse_args(argv))
    print(json.dumps(report["results"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
