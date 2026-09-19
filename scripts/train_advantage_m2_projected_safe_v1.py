"""M1確率を保証済みprojected局面だけで安全側へ縮める固定OOF学習。"""

from __future__ import annotations

from scripts.production_dependency_contract import (
    dependency_receipt, production_compatible, saved_dependency_compatible,
)

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts import build_advantage_m2_dataset_v1 as receipt
from scripts import build_advantage_m2_projected_safe_dataset_v1 as dataset_builder
from src.event_provisional_oof_v1 import FoldPlan, fixed_fold_plan
from src.projected_set_bounded_shrink_cnn_v2 import (
    MAX_PROBABILITY_SHRINK,
    ProjectedSetBoundedShrinkCNNV2,
)
from src.projected_set_residual_cnn_v1 import ProjectedSetAuxiliaryTargetsV1
from src.projected_set_training_v1 import (
    EPOCH_COUNT,
    LOGICAL_BATCH_SIZE,
    TRAINING_SEEDS,
    configure_deterministic_training_v1,
    deterministic_epoch_order_v1,
    make_projected_optimizer_v1,
    make_projected_scheduler_v1,
    train_projected_logical_batch_v1,
)
from src.projected_state_batch_v1 import ProjectedStateBatchV1
from src.projected_state_tensorizer_v1 import AUXILIARY_MANIFEST_SHA256


FORMAT_VERSION = "advantage-m2-projected-safe-oof/v1"
PLAN_VERSION = "advantage-m2-projected-safe-oof-plan/v1"
COMPLETE_VERSION = "advantage-m2-projected-safe-oof-complete/v1"
PRODUCTION_CONFIG_SHA256 = "3fe3c2578b3196a60cffffcafa594c1f64d2a913bb0487932ac5cd8f6f86d376"
DEFAULT_DATASET_ROOT = Path(
    "data/verify/advantage_m2_projected_safe_dataset_46v_2026-09-06_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/verify/advantage_m2_projected_safe_oof_46v_2026-09-06_v1"
)
DEFAULT_FOLDS = (1, 2, 3, 4, 5, 6)
PROBABILITY_EPSILON = 1e-12
SAFETY_ABS_TOLERANCE = 1e-12
REPO_ROOT = Path(__file__).resolve().parents[1]


class AdvantageM2ProjectedSafeTrainingError(RuntimeError):
    """projected-safe OOFの固定契約違反。"""


@dataclass(frozen=True, slots=True)
class ProjectedSafeSamplesV1:
    """全行評価情報とsupported行だけのprojected tensor。"""

    full: Mapping[str, np.ndarray]
    supported: Mapping[str, np.ndarray]

    @property
    def global_indices(self) -> np.ndarray:
        return self.supported["global_indices"]

    @property
    def supported_folds(self) -> np.ndarray:
        return self.full["folds"][self.global_indices]


@dataclass(frozen=True, slots=True)
class FoldSelectionV1:
    """tuneで選んだmodel状態と評価値。"""

    state: Mapping[str, torch.Tensor]
    epoch: int
    tune_baseline_loss: float
    tune_candidate_loss: float


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM2ProjectedSafeTrainingError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise AdvantageM2ProjectedSafeTrainingError(f"JSON objectではありません: {path}")
    return value


def load_dataset(root: Path) -> tuple[ProjectedSafeSamplesV1, dict[str, Any]]:
    """完了receiptと全arrayを検証して読み込む。"""

    manifest = _read_json(root / "manifest.json")
    complete = _read_json(root / "COMPLETE")
    data_path = root / "dataset.npz"
    _validate_dataset_receipts(root, data_path, manifest, complete)
    with np.load(data_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    samples = _split_arrays(arrays)
    _validate_dataset_arrays(samples, manifest)
    return samples, manifest


def _validate_dataset_receipts(
    root: Path, data_path: Path, manifest: Mapping[str, Any],
    complete: Mapping[str, Any],
) -> None:
    checks = (
        manifest.get("format_version") == dataset_builder.FORMAT_VERSION,
        complete.get("format_version") == dataset_builder.COMPLETE_VERSION,
        manifest.get("not_production") is True,
        manifest.get("production_config_changed") is False,
        complete.get("dataset_sha256") == receipt.file_sha256(data_path),
        complete.get("manifest_sha256") == receipt.file_sha256(root / "manifest.json"),
        production_compatible(REPO_ROOT),
    )
    if not all(checks):
        raise AdvantageM2ProjectedSafeTrainingError("dataset receiptまたは本番設定が不正です")


def _split_arrays(arrays: Mapping[str, np.ndarray]) -> ProjectedSafeSamplesV1:
    full_names = (
        "labels", "folds", "weights", "game_keys", "source_groups", "state_ids",
        "baseline_raw", "baseline_probability",
    )
    supported_names = (
        "projected_current", "projected_post_chain", "projected_landing",
        "projected_branch_mask", "projected_scalar", "projected_quantity_present_mask",
        "projected_auxiliary_current", "projected_auxiliary_post_chain",
        "projected_auxiliary_landing", "projected_auxiliary_current_mask",
        "projected_auxiliary_post_chain_mask", "projected_auxiliary_landing_mask",
        "projected_input_digests", "projected_tensor_digests",
    )
    missing = [name for name in (*full_names, *supported_names) if name not in arrays]
    if missing or "supported_global_indices" not in arrays:
        raise AdvantageM2ProjectedSafeTrainingError(f"dataset array不足です: {missing[:1]}")
    full = {name: arrays[name] for name in full_names}
    supported = {name: arrays[name] for name in supported_names}
    supported["global_indices"] = arrays["supported_global_indices"]
    return ProjectedSafeSamplesV1(full, supported)


def _validate_dataset_arrays(
    samples: ProjectedSafeSamplesV1, manifest: Mapping[str, Any],
) -> None:
    count = dataset_builder.EXPECTED_STATE_COUNT
    support = dataset_builder.EXPECTED_SUPPORTED_COUNT
    if any(value.shape[0] != count for value in samples.full.values()):
        raise AdvantageM2ProjectedSafeTrainingError("全行array数が固定値と不一致です")
    if any(value.shape[0] != support for value in samples.supported.values()):
        raise AdvantageM2ProjectedSafeTrainingError("supported array数が固定値と不一致です")
    indices = samples.global_indices
    checks = (
        indices.dtype == np.int32 and np.array_equal(indices, np.unique(indices)),
        set(int(value) for value in samples.full["folds"]) == set(DEFAULT_FOLDS),
        np.isfinite(samples.full["baseline_probability"]).all(),
        np.isfinite(samples.full["weights"]).all(),
        np.all(samples.full["weights"] > 0.0),
        manifest.get("projected_supported_count") == support,
    )
    if not all(checks):
        raise AdvantageM2ProjectedSafeTrainingError("datasetのindex・fold・確率・weightが不正です")
    _validate_supported_contract(samples)


def _validate_supported_contract(samples: ProjectedSafeSamplesV1) -> None:
    supported = samples.supported
    if not np.all(supported["projected_branch_mask"].sum(axis=1) > 0):
        raise AdvantageM2ProjectedSafeTrainingError("有効landing branchがない行があります")
    digests = tuple(str(value) for value in supported["projected_input_digests"])
    if len(set(digests)) != len(digests) or any(len(value) != 64 for value in digests):
        raise AdvantageM2ProjectedSafeTrainingError("projected digestが空または重複です")


def _batch(
    samples: ProjectedSafeSamplesV1, local_indices: np.ndarray,
    device: torch.device,
) -> ProjectedStateBatchV1:
    supported = samples.supported
    tensor = lambda name, dtype: torch.as_tensor(
        supported[name][local_indices], dtype=dtype, device=device,
    )
    auxiliary = ProjectedSetAuxiliaryTargetsV1(
        current=tensor("projected_auxiliary_current", torch.float32),
        post_chain=tensor("projected_auxiliary_post_chain", torch.float32),
        landing=tensor("projected_auxiliary_landing", torch.float32),
        current_mask=tensor("projected_auxiliary_current_mask", torch.bool),
        post_chain_mask=tensor("projected_auxiliary_post_chain_mask", torch.bool),
        landing_mask=tensor("projected_auxiliary_landing_mask", torch.bool),
    )
    branch_mask = tensor("projected_branch_mask", torch.bool)
    return ProjectedStateBatchV1(
        current=tensor("projected_current", torch.float32),
        post_chain=tensor("projected_post_chain", torch.float32),
        landing=tensor("projected_landing", torch.float32),
        branch_mask=branch_mask,
        scalar=tensor("projected_scalar", torch.float32),
        quantity_present_mask=tensor("projected_quantity_present_mask", torch.bool),
        candidate_count=branch_mask.sum(1).to(torch.int8), auxiliary_targets=auxiliary,
        observation_input_digests=tuple(
            str(value) for value in supported["projected_input_digests"][local_indices]
        ),
        auxiliary_manifest_sha256=AUXILIARY_MANIFEST_SHA256,
        tensor_content_digests=tuple(
            str(value) for value in supported["projected_tensor_digests"][local_indices]
        ),
    )


def weighted_log_loss(
    labels: np.ndarray, probability: np.ndarray, weights: np.ndarray,
) -> float:
    """固定game weightによるbinary log-loss。"""

    values = np.clip(np.asarray(probability, np.float64), PROBABILITY_EPSILON,
                     1.0 - PROBABILITY_EPSILON)
    truth = np.asarray(labels, np.float64)
    weight = np.asarray(weights, np.float64)
    if truth.shape != values.shape or weight.shape != truth.shape:
        raise AdvantageM2ProjectedSafeTrainingError("log-loss入力shapeが不一致です")
    rows = -(truth * np.log(values) + (1.0 - truth) * np.log(1.0 - values))
    return float(np.sum(rows * weight) / np.sum(weight))


def _full_fold_loss(
    samples: ProjectedSafeSamplesV1, fold: int,
    support_local: np.ndarray, support_probability: np.ndarray,
) -> float:
    full_mask = samples.full["folds"] == fold
    full_probability = samples.full["baseline_probability"].copy()
    full_probability[samples.global_indices[support_local]] = support_probability
    return weighted_log_loss(
        samples.full["labels"][full_mask], full_probability[full_mask],
        samples.full["weights"][full_mask],
    )


def _predict_supported(
    model: ProjectedSetBoundedShrinkCNNV2, samples: ProjectedSafeSamplesV1,
    local_indices: np.ndarray, device: torch.device,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(local_indices), LOGICAL_BATCH_SIZE):
            current = local_indices[start:start + LOGICAL_BATCH_SIZE]
            batch = _batch(samples, current, device)
            baseline = torch.as_tensor(
                samples.full["baseline_probability"][samples.global_indices[current]],
                dtype=torch.float64, device=device,
            )
            output = model(
                batch.current, batch.post_chain, batch.landing, batch.branch_mask,
                batch.scalar, batch.quantity_present_mask, baseline,
            )
            values.append(output.raw_probability.cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, np.float64)


def validate_bounded_probability(
    baseline: np.ndarray, candidate: np.ndarray,
) -> None:
    """非拡大・0.5非越境・最大50%縮約を検証する。"""

    base = np.asarray(baseline, np.float64)
    value = np.asarray(candidate, np.float64)
    lower, upper = np.minimum(base, 0.5), np.maximum(base, 0.5)
    moved = np.abs(value - base)
    allowed = MAX_PROBABILITY_SHRINK * np.abs(0.5 - base)
    tolerance = SAFETY_ABS_TOLERANCE
    valid = (
        np.isfinite(value) & (value >= lower - tolerance)
        & (value <= upper + tolerance) & (moved <= allowed + tolerance)
    )
    if not np.all(valid):
        violation = float(np.max(moved - allowed))
        raise AdvantageM2ProjectedSafeTrainingError(
            f"candidateがbounded safety契約を破りました: max_excess={violation:.3e}"
        )


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _train_epoch(
    model: ProjectedSetBoundedShrinkCNNV2, optimizer: torch.optim.AdamW,
    samples: ProjectedSafeSamplesV1, train_local: np.ndarray,
    seed: int, epoch: int, device: torch.device,
) -> float:
    digests = tuple(str(value) for value in samples.supported["projected_input_digests"][train_local])
    order = deterministic_epoch_order_v1(seed, epoch - 1, digests)
    ordered = train_local[np.asarray(order, np.int32)]
    losses: list[float] = []
    model.train()
    for start in range(0, len(ordered), LOGICAL_BATCH_SIZE):
        local = ordered[start:start + LOGICAL_BATCH_SIZE]
        global_indices = samples.global_indices[local]
        batch = _batch(samples, local, device)
        baseline = torch.as_tensor(samples.full["baseline_probability"][global_indices],
                                   dtype=torch.float64, device=device)
        labels = torch.as_tensor(samples.full["labels"][global_indices],
                                 dtype=torch.float64, device=device)
        weights = torch.as_tensor(samples.full["weights"][global_indices],
                                  dtype=torch.float64, device=device)
        step = train_projected_logical_batch_v1(
            model, optimizer, batch, baseline, labels, weights,
        )
        losses.append(step.total)
    return float(np.mean(losses))


def _initial_selection(
    model: ProjectedSetBoundedShrinkCNNV2, samples: ProjectedSafeSamplesV1,
    tune_local: np.ndarray, tune_fold: int, device: torch.device,
) -> FoldSelectionV1:
    baseline = samples.full["baseline_probability"][samples.global_indices[tune_local]]
    candidate = _predict_supported(model, samples, tune_local, device)
    if not np.array_equal(candidate, baseline):
        raise AdvantageM2ProjectedSafeTrainingError("epoch 0がM1 baselineとbit-identicalではありません")
    loss = _full_fold_loss(samples, tune_fold, tune_local, candidate)
    return FoldSelectionV1(_cpu_state(model), 0, loss, loss)


def _fit_fold(
    samples: ProjectedSafeSamplesV1, plan: FoldPlan, seed: int,
    args: argparse.Namespace, device: torch.device,
) -> FoldSelectionV1:
    configure_deterministic_training_v1(seed)
    model = ProjectedSetBoundedShrinkCNNV2().to(device)
    train_local = np.flatnonzero(np.isin(samples.supported_folds, plan.train_folds))
    tune_local = np.flatnonzero(samples.supported_folds == plan.tune_fold)
    best = _initial_selection(model, samples, tune_local, plan.tune_fold, device)
    optimizer = make_projected_optimizer_v1(model)
    scheduler = make_projected_scheduler_v1(optimizer)
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, optimizer, samples, train_local, seed, epoch, device)
        candidate = _predict_supported(model, samples, tune_local, device)
        loss = _full_fold_loss(samples, plan.tune_fold, tune_local, candidate)
        print(f"seed={seed} eval={plan.eval_fold} epoch={epoch} "
              f"train={train_loss:.6f} tune={loss:.6f}", flush=True)
        if loss < best.tune_candidate_loss:
            best = FoldSelectionV1(_cpu_state(model), epoch,
                                   best.tune_baseline_loss, loss)
        scheduler.step()
    model.load_state_dict(best.state)
    return best


def _save_model(
    root: Path, selection: FoldSelectionV1, seed: int, fold: int,
) -> Path:
    path = root / f"m2_projected_safe__seed_{seed}__fold_{fold}.pt"
    payload = {
        "format_version": FORMAT_VERSION, "seed": seed, "eval_fold": fold,
        "selected_epoch": selection.epoch, "state_dict": dict(selection.state),
        "tune_baseline_loss": selection.tune_baseline_loss,
        "tune_candidate_loss": selection.tune_candidate_loss,
    }
    with path.open("xb") as handle:
        torch.save(payload, handle)
    return path


def _run_fold(
    samples: ProjectedSafeSamplesV1, plan: FoldPlan, seed: int,
    args: argparse.Namespace, device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    selection = _fit_fold(samples, plan, seed, args, device)
    model = ProjectedSetBoundedShrinkCNNV2().to(device)
    model.load_state_dict(selection.state)
    eval_local = np.flatnonzero(samples.supported_folds == plan.eval_fold)
    candidate = _predict_supported(model, samples, eval_local, device)
    baseline = samples.full["baseline_probability"][samples.global_indices[eval_local]]
    validate_bounded_probability(baseline, candidate)
    model_path = _save_model(args.output_root, selection, seed, plan.eval_fold)
    record = {
        "seed": seed, "eval_fold": plan.eval_fold, "tune_fold": plan.tune_fold,
        "train_folds": list(plan.train_folds), "selected_epoch": selection.epoch,
        "fallback_to_m1": selection.epoch == 0,
        "tune_baseline_loss": selection.tune_baseline_loss,
        "tune_candidate_loss": selection.tune_candidate_loss,
        "eval_supported_count": len(eval_local),
        "model_path": str(model_path.resolve()),
        "model_sha256": receipt.file_sha256(model_path),
    }
    _write_json_exclusive(args.output_root / f"fold_{seed}_{plan.eval_fold}.json", record)
    return candidate, record


def _run_seed(
    samples: ProjectedSafeSamplesV1, plans: Sequence[FoldPlan], seed: int,
    args: argparse.Namespace, device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probability = samples.full["baseline_probability"].copy()
    records: list[dict[str, Any]] = []
    for plan in plans:
        if plan.eval_fold not in args.folds:
            continue
        candidate, record = _run_fold(samples, plan, seed, args, device)
        local = np.flatnonzero(samples.supported_folds == plan.eval_fold)
        probability[samples.global_indices[local]] = candidate
        records.append(record)
    return probability, records


def _plan(args: argparse.Namespace, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": PLAN_VERSION, "not_production": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_sha256": manifest["dataset"]["sha256"],
        "seeds": list(args.seeds), "folds": list(args.folds),
        "epochs": args.epochs, "logical_batch_size": LOGICAL_BATCH_SIZE,
        "baseline": "M1_V3_calibrated_3seed_equal_ensemble",
        "selection": "full_tune_fold_game_equal_log_loss",
        "supported": "gate_status_guaranteed_exact_state_join",
        "unsupported": "bit_identical_M1",
        "max_probability_shrink": MAX_PROBABILITY_SHRINK,
        "formal100_used": False, "hidden_reserve_used": False,
        "production_config_changed": False,
        "production_dependency_contract": dependency_receipt(REPO_ROOT),
        "code_sha256": {
            "trainer": receipt.file_sha256(Path(__file__)),
            "model": receipt.file_sha256(
                REPO_ROOT / "src/projected_set_bounded_shrink_cnn_v2.py"
            ),
            "dataset_builder": receipt.file_sha256(
                REPO_ROOT / "scripts/build_advantage_m2_projected_safe_dataset_v1.py"
            ),
            "production_config": receipt.file_sha256(REPO_ROOT / "src/production_config.py"),
        },
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return path


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return path


def _write_results(
    args: argparse.Namespace, samples: ProjectedSafeSamplesV1,
    plan: Mapping[str, Any], plan_path: Path,
    predictions: Mapping[int, np.ndarray], records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "labels": samples.full["labels"], "weights": samples.full["weights"],
        "folds": samples.full["folds"], "state_ids": samples.full["state_ids"],
        "source_groups": samples.full["source_groups"],
        "game_keys": samples.full["game_keys"],
        "baseline_probability": samples.full["baseline_probability"],
        "supported_global_indices": samples.global_indices,
    }
    arrays.update({f"candidate__seed_{seed}": value for seed, value in predictions.items()})
    prediction_path = _write_npz_exclusive(args.output_root / "oof_predictions.npz", arrays)
    report = {"format_version": FORMAT_VERSION, "not_production": True,
              "plan": dict(plan), "fold_records": list(records)}
    result_path = _write_json_exclusive(args.output_root / "results.json", report)
    complete = {
        "format_version": COMPLETE_VERSION, "not_production": True,
        "plan_sha256": receipt.file_sha256(plan_path),
        "results_sha256": receipt.file_sha256(result_path),
        "predictions_sha256": receipt.file_sha256(prediction_path),
        "production_config_changed": False,
        "production_dependency_contract": dependency_receipt(REPO_ROOT),
    }
    _write_json_exclusive(args.output_root / "COMPLETE", complete)
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    """固定OOFを新規rootへ保存する。"""

    if args.output_root.exists():
        raise AdvantageM2ProjectedSafeTrainingError(f"出力先は新規必須です: {args.output_root}")
    samples, manifest = load_dataset(args.dataset_root)
    plans = fixed_fold_plan(samples.full["folds"])
    args.output_root.mkdir(parents=True, exist_ok=False)
    plan = _plan(args, manifest)
    plan_path = _write_json_exclusive(args.output_root / "PLAN.json", plan)
    device = _device(args.device)
    predictions: dict[int, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for seed in args.seeds:
        prediction, seed_records = _run_seed(samples, plans, seed, args, device)
        predictions[seed] = prediction
        records.extend(seed_records)
    return _write_results(args, samples, plan, plan_path, predictions, records)


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise AdvantageM2ProjectedSafeTrainingError("CUDAが利用できません")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("整数CSVが必要です") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=_csv_ints, default=TRAINING_SEEDS)
    parser.add_argument("--folds", type=_csv_ints, default=DEFAULT_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCH_COUNT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if not args.seeds or any(seed not in TRAINING_SEEDS for seed in args.seeds):
        parser.error("seedは固定3seedから選んでください")
    if not args.folds or any(fold not in DEFAULT_FOLDS for fold in args.folds):
        parser.error("foldは1..6から選んでください")
    if args.epochs < 1 or args.epochs > EPOCH_COUNT:
        parser.error(f"epochsは1..{EPOCH_COUNT}必須です")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = train(parse_args(argv))
    print(json.dumps({"fold_count": len(report["fold_records"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
