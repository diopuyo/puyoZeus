"""c82増分のfloat32 weight scaleだけを除去してM0 fold 1を再現診断する。"""

from __future__ import annotations

from scripts.production_dependency_contract import (
    dependency_receipt, production_compatible, saved_dependency_compatible,
)

import argparse
import json
import os
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import train_advantage_m0_current_cnn_v1 as base
from scripts import train_advantage_m1_fixed_m0_anchor_v1 as fixed
from scripts import train_advantage_m1_frozen_residual_v1 as frozen
from scripts import train_advantage_m1_zero_counterfactual_v3 as v3
from src.event_provisional_oof_v1 import FoldPlan, fixed_fold_plan
from src import projected_set_training_v1 as runtime


FORMAT_VERSION = "advantage-m1-weight-scale-diagnosis/v1"
TARGET_DATASET_RECEIPT = {
    "dataset_sha256": "017cdc009349322d8eabc17dc9ad1a34cde440fcb97b41e4d4910a6c127dd071",
    "dataset_manifest_sha256": "b746812e07eb80a4e14b6f674e93fe3b112b38d6aca034a7c9c0257e1a9f9b8e",
    "dataset_complete_sha256": "84ba6a0307cdd1bca16cf57abb4c7585ce69b4e0d8b8e24c709a2ec0cac8243c",
}
TARGET_GROUP = "04Lb9BZCpP0"
TARGET_ID = "c82"
PUBLIC_SEED = 20260904
EVAL_FOLD = 1
FOLD_SEED = 20261004
REPO_ROOT = Path(__file__).resolve().parents[1]


class WeightScaleDiagnosisError(RuntimeError):
    """c82 weight scale診断の固定契約に違反した。"""


def _target_receipt(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    receipt = {"dataset_sha256": str(manifest.get("dataset", {}).get("sha256", ""))}
    receipt.update(v3.dataset_identity_receipt(root, manifest))
    return {key: receipt[key] for key in TARGET_DATASET_RECEIPT}


def _validate_target_receipt(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    receipt = _target_receipt(root, manifest)
    if receipt != TARGET_DATASET_RECEIPT:
        raise WeightScaleDiagnosisError("事前登録c82 47v dataset receiptと一致しません")
    if int(manifest.get("source_count", -1)) != 47:
        raise WeightScaleDiagnosisError("c82診断datasetは47 source必須です")
    return receipt


def _validate_sources(
    anchor_manifest: Mapping[str, Any], target_manifest: Mapping[str, Any],
) -> None:
    anchor = fixed._source_rows(anchor_manifest)
    target = fixed._source_rows(target_manifest)
    if set(target) - set(anchor) != {TARGET_GROUP} or set(anchor) - set(target):
        raise WeightScaleDiagnosisError("c82以外のsource増減があります")
    if any(target[group] != row for group, row in anchor.items()):
        raise WeightScaleDiagnosisError("clean46共通source receiptが変化しています")
    added = target[TARGET_GROUP]
    if str(added.get("target_id")) != TARGET_ID or int(added.get("partition_fold", -1)) != 1:
        raise WeightScaleDiagnosisError("c82 source/fold receiptが不正です")


def _relation(
    anchor: v3.CanonicalSamplesV3, anchor_manifest: Mapping[str, Any],
    target: v3.CanonicalSamplesV3, target_manifest: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _validate_sources(anchor_manifest, target_manifest)
    anchor_index, target_indices = fixed._common_indices(anchor, target)
    fixed._validate_exact_common(anchor, target, target_indices)
    scale = fixed._common_weight_scale(anchor, target, target_indices)
    common = np.asarray([str(value) in anchor_index for value in target.state_ids])
    anchor_positions = np.asarray([
        anchor_index[str(value)] for value in target.state_ids[common]
    ], dtype=np.int64)
    if {str(value) for value in target.source_groups[~common]} != {TARGET_GROUP}:
        raise WeightScaleDiagnosisError("追加stateがc82だけではありません")
    if {int(value) for value in target.folds[~common]} != {EVAL_FOLD}:
        raise WeightScaleDiagnosisError("c82追加stateがeval fold 1以外へ漏れています")
    if {str(value) for value in anchor.game_keys} & {str(value) for value in target.game_keys[~common]}:
        raise WeightScaleDiagnosisError("clean46/c82でgame keyが衝突しています")
    receipt = {
        "common_state_count": int(common.sum()), "added_state_count": int((~common).sum()),
        "added_group": TARGET_GROUP, "added_folds": [EVAL_FOLD],
        "train_added_count": 0, "tune_added_count": 0,
        "audited_common_median_scale": scale,
    }
    return common, anchor_positions, receipt


def _stabilize_common_weights(
    target: v3.CanonicalSamplesV3, anchor: v3.CanonicalSamplesV3,
    common: np.ndarray, anchor_positions: np.ndarray, plan: FoldPlan,
) -> tuple[v3.CanonicalSamplesV3, dict[str, Any]]:
    weights = target.weights.copy()
    weights[common] = anchor.weights[anchor_positions]
    if not np.array_equal(weights[common], anchor.weights[anchor_positions]):
        raise WeightScaleDiagnosisError("common weight direct copyがbit-exactではありません")
    train_or_tune = np.isin(target.folds, (*plan.train_folds, plan.tune_fold))
    if np.any(train_or_tune & ~common):
        raise WeightScaleDiagnosisError("c82追加stateがM0 train/tuneへ混入しています")
    stable = replace(target, weights=np.ascontiguousarray(weights))
    return stable, {
        "policy": "common_state_id_direct_copy_clean46_float32_bytes",
        "input_target_weight_sha256": fixed._array_sha256(target.weights),
        "stabilized_weight_sha256": fixed._array_sha256(stable.weights),
        "common_weight_sha256": fixed._array_sha256(stable.weights[common]),
        "common_weight_bytes_exact": True,
        "train_row_count": int(np.isin(target.folds, plan.train_folds).sum()),
        "tune_row_count": int((target.folds == plan.tune_fold).sum()),
        "train_tune_added_count": int((train_or_tune & ~common).sum()),
    }


def _assert_anchor_exact(
    fitted: frozen.M0Anchor, reference: fixed.M0AnchorReference,
) -> None:
    expected = (
        fitted.state_sha256 == reference.state_sha256,
        fitted.best_epoch == reference.best_epoch,
        fitted.slope == reference.slope,
        dict(fitted.tune_raw) == dict(reference.tune_raw),
        dict(fitted.tune_calibrated) == dict(reference.tune_calibrated),
    )
    if not all(expected):
        raise WeightScaleDiagnosisError("weight安定化後M0がclean46正本と一致しません")


def _environment(device: torch.device) -> dict[str, Any]:
    gpu_names, capabilities = runtime._gpu_information()
    return {
        "python_version": platform.python_version(), "platform": platform.platform(),
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda or "none",
        "cuda_driver_version": runtime._cuda_driver_version(),
        "cudnn_version": torch.backends.cudnn.version() or "none",
        "gpu_names": gpu_names, "gpu_capabilities": capabilities,
        "selected_device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
    }


def _code_receipt() -> dict[str, str]:
    paths = {
        "diagnoser": Path(__file__), "fixed_m0_trainer": Path(fixed.__file__),
        "v3_trainer": Path(v3.__file__), "frozen_trainer": Path(frozen.__file__),
        "m0_model": REPO_ROOT / "src/advantage_m0_current_cnn_v1.py",
        "production_config": REPO_ROOT / "src/production_config.py",
    }
    receipt = {name: base.file_sha256(path) for name, path in paths.items()}
    if not production_compatible(REPO_ROOT):
        raise WeightScaleDiagnosisError("production_config SHAが変化しています")
    return receipt


def _input_assets(
    dataset_root: Path, registry: Mapping[tuple[int, int], fixed.M0AnchorReference],
) -> list[dict[str, str]]:
    args = argparse.Namespace(dataset_root=dataset_root)
    champion = fixed.load_champion_reference(registry)
    return fixed._capture_input_assets(fixed._input_asset_paths(args, registry, champion))


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    """ファイルを書かずc82のweight量子化交絡を一fold実学習で確定する。"""

    target, target_manifest = v3.load_canonical_dataset_v2(args.dataset_root)
    target_receipt = _validate_target_receipt(args.dataset_root, target_manifest)
    registry, anchor_roots = fixed.load_anchor_registry(args.anchor_roots)
    anchor, anchor_manifest = fixed._load_anchor_dataset(registry)
    common, positions, relation = _relation(
        anchor, anchor_manifest, target, target_manifest,
    )
    plan = {item.eval_fold: item for item in fixed_fold_plan(target.folds)}[EVAL_FOLD]
    stable, weight_receipt = _stabilize_common_weights(
        target, anchor, common, positions, plan,
    )
    assets = _input_assets(args.dataset_root, registry)
    device = base._device(args.device)
    fitted = frozen._fit_m0_anchor(stable, plan, FOLD_SEED, args, device)
    reference = registry[(PUBLIC_SEED, EVAL_FOLD)]
    _assert_anchor_exact(fitted, reference)
    fixed._assert_input_assets_unchanged(assets)
    return {
        "format_version": FORMAT_VERSION, "not_production": True,
        "read_only": True, "dataset_receipt": target_receipt,
        "anchor_root_receipts": anchor_roots, "relation_receipt": relation,
        "weight_receipt": weight_receipt,
        "fit_contract": {"seed": PUBLIC_SEED, "eval_fold": EVAL_FOLD,
                         "fold_seed": FOLD_SEED},
        "result": {"exact_match": True, "state_sha256": fitted.state_sha256,
                   "best_epoch": fitted.best_epoch, "slope": fitted.slope,
                   "tune_raw": dict(fitted.tune_raw),
                   "tune_calibrated": dict(fitted.tune_calibrated)},
        "code_sha256": _code_receipt(), "environment": _environment(device),
        "input_asset_receipts": assets,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--anchor-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if (args.epochs, args.patience, args.batch_size) != (12, 3, 512):
        parser.error("学習回数・patience・batch sizeは固定です")
    if (args.learning_rate, args.weight_decay) != (0.0003, 0.0001):
        parser.error("learning rate・weight decayは固定です")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(diagnose(parse_args(argv)), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
