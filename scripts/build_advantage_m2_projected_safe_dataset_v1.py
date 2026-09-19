"""M1 V3 OOFへ保証済みprojected-state tensorだけを厳密結合する。"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts import build_advantage_m2_dataset_v1 as receipt
from scripts import train_advantage_m1_zero_counterfactual_v3 as m1_train
from src.projected_state_dataset_v1 import (
    ProjectedStateDatasetReceipt,
    iter_projected_state_dataset,
    validate_projected_state_dataset,
)
from src.projected_state_observation_v1 import ProjectedStateObservationV1
from src.projected_state_tensorizer_v1 import (
    AUXILIARY_MANIFEST_SHA256,
    ProjectedStateTensorV1,
    tensorize_projected_state_observation_v1,
)


FORMAT_VERSION = "advantage-m2-projected-safe-dataset/v1"
COMPLETE_VERSION = "advantage-m2-projected-safe-dataset-complete/v1"
M1_VARIANT = "m1_zero_values_and_masks"
M1_SEEDS = (20260904, 20260905, 20260906)
EXPECTED_SOURCE_COUNT = 46
EXPECTED_GAME_COUNT = 2_426
EXPECTED_STATE_COUNT = 224_011
EXPECTED_SUPPORTED_COUNT = 1_660
OFFICIAL_GAME_NUMBER_OFFSET = 1
DEFAULT_PARENT_ROOT = Path(
    "data/verify/advantage_m1_canonical_dataset_46v_2026-09-05_v3_finalized_primary6"
)
DEFAULT_PROJECTED_ROOT = Path(
    "data/verify/projected_state_dataset_pilot48_2026-09-04_v1"
)
DEFAULT_M1_OOF_ROOTS = (
    Path("data/verify/advantage_m1_zero_counterfactual_46v_allfolds_"
         "seed20260904_2026-09-05_v3_finalized_primary6_merged"),
    Path("data/verify/advantage_m1_zero_counterfactual_46v_allfolds_"
         "seeds20260905_20260906_2026-09-05_v3_finalized_primary6_merged"),
)
REPO_ROOT = Path(__file__).resolve().parents[1]


class AdvantageM2ProjectedDatasetError(RuntimeError):
    """M1 OOFまたはprojected-stateの結合契約に違反した。"""


@dataclass(frozen=True, slots=True)
class SupportedJoin:
    """M1全行indexと保証済みprojected観測の一対一対応。"""

    global_indices: np.ndarray
    observations: tuple[ProjectedStateObservationV1, ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM2ProjectedDatasetError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise AdvantageM2ProjectedDatasetError(f"JSON objectではありません: {path}")
    return value


def _load_parent(
    root: Path,
) -> tuple[m1_train.CanonicalSamplesV3, dict[str, Any]]:
    try:
        samples, manifest = m1_train.load_canonical_dataset_v2(root)
    except (OSError, ValueError, RuntimeError) as error:
        raise AdvantageM2ProjectedDatasetError("親M1 dataset検証に失敗しました") from error
    counts = (manifest["source_count"], manifest["game_count"], len(samples.labels))
    if counts != (EXPECTED_SOURCE_COUNT, EXPECTED_GAME_COUNT, EXPECTED_STATE_COUNT):
        raise AdvantageM2ProjectedDatasetError(f"親M1母集団が固定値と不一致です: {counts}")
    return samples, manifest


def _load_oof_root(root: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    complete = _read_json(root / "COMPLETE")
    prediction_path, result_path = root / "oof_predictions.npz", root / "results.json"
    checks = (
        complete.get("predictions_sha256") == receipt.file_sha256(prediction_path),
        complete.get("results_sha256") == receipt.file_sha256(result_path),
    )
    if not all(checks):
        raise AdvantageM2ProjectedDatasetError(f"M1 OOF receipt不一致です: {root}")
    try:
        with np.load(prediction_path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as error:
        raise AdvantageM2ProjectedDatasetError(f"M1 OOFを読めません: {root}") from error
    return arrays, {
        "root": str(root.resolve()),
        "complete_sha256": receipt.file_sha256(root / "COMPLETE"),
        "predictions_sha256": receipt.file_sha256(prediction_path),
        "results_sha256": receipt.file_sha256(result_path),
    }


def _baseline_ensemble(
    samples: m1_train.CanonicalSamplesV3, roots: Sequence[Path],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    raw_by_seed: dict[int, np.ndarray] = {}
    calibrated_by_seed: dict[int, np.ndarray] = {}
    receipts: list[dict[str, str]] = []
    for root in roots:
        arrays, root_receipt = _load_oof_root(root)
        _validate_oof_identity(samples, arrays, root)
        _collect_seed_arrays(arrays, raw_by_seed, calibrated_by_seed)
        receipts.append(root_receipt)
    if set(raw_by_seed) != set(M1_SEEDS) or set(calibrated_by_seed) != set(M1_SEEDS):
        raise AdvantageM2ProjectedDatasetError("M1 3seed OOFが過不足なく揃っていません")
    raw = np.mean(np.stack([raw_by_seed[s] for s in M1_SEEDS]), axis=0)
    calibrated = np.mean(np.stack([calibrated_by_seed[s] for s in M1_SEEDS]), axis=0)
    _validate_probability(raw, "M1 raw ensemble")
    _validate_probability(calibrated, "M1 calibrated ensemble")
    return raw, calibrated, receipts


def _validate_oof_identity(
    samples: m1_train.CanonicalSamplesV3,
    arrays: Mapping[str, np.ndarray],
    root: Path,
) -> None:
    labels, folds = arrays.get("labels"), arrays.get("folds")
    if labels is None or folds is None:
        raise AdvantageM2ProjectedDatasetError(f"M1 OOF identity列がありません: {root}")
    if not np.array_equal(labels, samples.labels) or not np.array_equal(folds, samples.folds):
        raise AdvantageM2ProjectedDatasetError(f"M1 OOF行同定が親datasetと不一致です: {root}")


def _collect_seed_arrays(
    arrays: Mapping[str, np.ndarray],
    raw_by_seed: dict[int, np.ndarray],
    calibrated_by_seed: dict[int, np.ndarray],
) -> None:
    for seed in M1_SEEDS:
        prefix = f"{M1_VARIANT}__seed_{seed}"
        if f"{prefix}__raw" not in arrays:
            continue
        if seed in raw_by_seed or f"{prefix}__calibrated" not in arrays:
            raise AdvantageM2ProjectedDatasetError(f"M1 seed重複または較正欠損です: {seed}")
        raw_by_seed[seed] = np.asarray(arrays[f"{prefix}__raw"], dtype=np.float64)
        calibrated_by_seed[seed] = np.asarray(
            arrays[f"{prefix}__calibrated"], dtype=np.float64,
        )


def _validate_probability(value: np.ndarray, label: str) -> None:
    if value.shape != (EXPECTED_STATE_COUNT,) or not np.isfinite(value).all():
        raise AdvantageM2ProjectedDatasetError(f"{label}のshapeまたは有限性が不正です")
    if np.any((value < 0.0) | (value > 1.0)):
        raise AdvantageM2ProjectedDatasetError(f"{label}が0..1範囲外です")


def select_supported_observations(
    samples: m1_train.CanonicalSamplesV3,
    observations: Iterable[ProjectedStateObservationV1],
) -> SupportedJoin:
    """保証済み観測だけをM1 state IDへ一対一結合する。"""

    positions = {str(value): index for index, value in enumerate(samples.state_ids)}
    if len(positions) != len(samples.state_ids):
        raise AdvantageM2ProjectedDatasetError("親M1 state IDが重複しています")
    joined: dict[int, ProjectedStateObservationV1] = {}
    for observation in observations:
        position = positions.get(observation.observation_id)
        if observation.gate_status != "guaranteed" or position is None:
            continue
        _validate_joined_observation(samples, position, observation)
        if position in joined:
            raise AdvantageM2ProjectedDatasetError("同じM1 stateへ複数観測がjoinしました")
        joined[position] = observation
    ordered = tuple(sorted(joined))
    return SupportedJoin(
        np.asarray(ordered, dtype=np.int32), tuple(joined[index] for index in ordered),
    )


def _validate_joined_observation(
    samples: m1_train.CanonicalSamplesV3,
    position: int,
    observation: ProjectedStateObservationV1,
) -> None:
    official_number = observation.game_idx + OFFICIAL_GAME_NUMBER_OFFSET
    expected_game = f"{observation.source_video_id}:game-{official_number:04d}"
    values = (
        str(samples.projected_gate_status[position]) == "guaranteed",
        str(samples.projected_input_digest[position]) == observation.input_digest,
        str(samples.game_keys[position]) == expected_game,
    )
    if not all(values):
        raise AdvantageM2ProjectedDatasetError(
            f"projected観測のgate/digest/gameがM1と不一致です: {observation.observation_id}"
        )


def _tensor_arrays(values: Sequence[ProjectedStateObservationV1]) -> dict[str, np.ndarray]:
    tensors = tuple(tensorize_projected_state_observation_v1(value) for value in values)
    if not tensors:
        raise AdvantageM2ProjectedDatasetError("保証済みprojected観測が0件です")
    names = (
        "current", "post_chain", "landing", "branch_mask", "scalar",
        "quantity_present_mask", "auxiliary_current", "auxiliary_post_chain",
        "auxiliary_landing", "auxiliary_current_mask", "auxiliary_post_chain_mask",
        "auxiliary_landing_mask",
    )
    arrays = {f"projected_{name}": _stack_tensor_field(tensors, name) for name in names}
    arrays["projected_input_digests"] = np.asarray(
        [value.observation_input_digest for value in tensors],
    )
    arrays["projected_tensor_digests"] = np.asarray(
        [value.tensor_content_digest for value in tensors],
    )
    return arrays


def _stack_tensor_field(
    values: Sequence[ProjectedStateTensorV1], name: str,
) -> np.ndarray:
    array = np.stack([getattr(value, name) for value in values])
    if name in {"current", "post_chain", "landing"}:
        return array.astype(np.uint8)
    return array


def _base_arrays(
    samples: m1_train.CanonicalSamplesV3,
    baseline_raw: np.ndarray,
    baseline_probability: np.ndarray,
    join: SupportedJoin,
) -> dict[str, np.ndarray]:
    return {
        "labels": samples.labels, "folds": samples.folds,
        "weights": samples.weights, "game_keys": samples.game_keys,
        "source_groups": samples.source_groups, "state_ids": samples.state_ids,
        "ledger_values": samples.ledger_values,
        "ledger_availability": samples.ledger_availability,
        "ledger_usable": samples.ledger_usable,
        "available_ms": samples.available_ms,
        "online_segment_index": samples.online_segment_index,
        "baseline_raw": baseline_raw,
        "baseline_probability": baseline_probability,
        "supported_global_indices": join.global_indices,
        "supported_source_video_ids": np.asarray(
            [value.source_video_id for value in join.observations],
        ),
    }


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _array_receipts(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    important = (
        "state_ids", "baseline_raw", "baseline_probability",
        "supported_global_indices", "projected_input_digests",
        "projected_tensor_digests", "projected_landing",
    )
    return {name: receipt._array_sha256(arrays[name]) for name in important}


def _manifest(
    args: argparse.Namespace,
    arrays: Mapping[str, np.ndarray],
    parent_manifest: Mapping[str, Any],
    projected_receipt: ProjectedStateDatasetReceipt,
    oof_receipts: Sequence[Mapping[str, str]],
    dataset_path: Path,
) -> dict[str, Any]:
    supported = arrays["supported_global_indices"]
    return {
        "format_version": FORMAT_VERSION, "not_production": True,
        "source_count": EXPECTED_SOURCE_COUNT, "game_count": EXPECTED_GAME_COUNT,
        "state_count": EXPECTED_STATE_COUNT, "projected_supported_count": len(supported),
        "projected_supported_source_count": len(np.unique(arrays["supported_source_video_ids"])),
        "projected_supported_game_count": len(np.unique(arrays["game_keys"][supported])),
        "parent_root": str(args.parent_root.resolve()),
        "parent_manifest_sha256": receipt.file_sha256(args.parent_root / "manifest.json"),
        "parent_dataset_sha256": parent_manifest["dataset"]["sha256"],
        "projected_root": str(args.projected_root.resolve()),
        "projected_receipt": {
            "record_count": projected_receipt.record_count,
            "source_count": projected_receipt.source_count,
            "observations_sha256": projected_receipt.observations_sha256,
            "manifest_sha256": projected_receipt.manifest_sha256,
        },
        "m1_oof_receipts": list(oof_receipts), "m1_seeds": list(M1_SEEDS),
        "baseline_policy": "equal_mean_of_three_calibrated_m1_v3_oof_probabilities",
        "support_policy": "exact_state_id_join_and_projected_gate_guaranteed_only",
        "fallback_policy": "all_unsupported_rows_bit_identical_m1_probability",
        "auxiliary_manifest_sha256": AUXILIARY_MANIFEST_SHA256,
        "array_sha256": _array_receipts(arrays),
        "dataset": {"name": dataset_path.name, "sha256": receipt.file_sha256(dataset_path)},
        "formal100_used": False, "hidden_reserve_used": False,
        "production_config_changed": False, "attack_difference_ten_percent_correction": False,
        "code_sha256": {
            "builder": receipt.file_sha256(Path(__file__)),
            "tensorizer": receipt.file_sha256(REPO_ROOT / "src/projected_state_tensorizer_v1.py"),
            "model": receipt.file_sha256(REPO_ROOT / "src/projected_set_bounded_shrink_cnn_v2.py"),
            "production_config": receipt.file_sha256(REPO_ROOT / "src/production_config.py"),
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    """固定46-source全行と保証済みprojected tensorを新規rootへ保存する。"""

    if args.output_root.exists():
        raise AdvantageM2ProjectedDatasetError(f"出力先は新規必須です: {args.output_root}")
    samples, parent_manifest = _load_parent(args.parent_root)
    baseline_raw, baseline_probability, oof_receipts = _baseline_ensemble(
        samples, args.m1_oof_roots,
    )
    projected_receipt = validate_projected_state_dataset(args.projected_root)
    observations = iter_projected_state_dataset(args.projected_root)
    join = select_supported_observations(samples, observations)
    if len(join.global_indices) != EXPECTED_SUPPORTED_COUNT:
        raise AdvantageM2ProjectedDatasetError(
            f"保証済み結合数が固定値と不一致です: {len(join.global_indices)}"
        )
    arrays = _base_arrays(samples, baseline_raw, baseline_probability, join)
    arrays.update(_tensor_arrays(join.observations))
    args.output_root.mkdir(parents=True, exist_ok=False)
    dataset_path = args.output_root / "dataset.npz"
    _write_npz_exclusive(dataset_path, arrays)
    manifest = _manifest(
        args, arrays, parent_manifest, projected_receipt, oof_receipts, dataset_path,
    )
    manifest_path = args.output_root / "manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    _write_json_exclusive(args.output_root / "COMPLETE", {
        "format_version": COMPLETE_VERSION,
        "manifest_sha256": receipt.file_sha256(manifest_path),
        "dataset_sha256": receipt.file_sha256(dataset_path),
    })
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--projected-root", type=Path, default=DEFAULT_PROJECTED_ROOT)
    parser.add_argument("--m1-oof-roots", type=Path, nargs="+", default=DEFAULT_M1_OOF_ROOTS)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build(parse_args(argv))
    print(json.dumps({key: manifest[key] for key in (
        "source_count", "game_count", "state_count", "projected_supported_count",
        "projected_supported_game_count",
    )}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
