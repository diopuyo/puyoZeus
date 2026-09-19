"""確定会計6列のzero-counterfactual V3を固定OOF学習する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import build_advantage_m1_dataset_v2 as builder
from scripts import train_advantage_m0_current_cnn_v1 as base
from scripts import train_advantage_m1_causal_ledger_v1 as legacy
from scripts import train_advantage_m1_frozen_residual_v1 as frozen
from src.advantage_m0_current_cnn_v1 import (
    AdvantageM0CurrentCNNV2,
    equal_game_weighted_bce,
)
from src.advantage_m1_causal_ledger_v3 import (
    AVAILABILITY_COUNT,
    AVAILABILITY_ORDER,
    LEDGER_FIELD_COUNT,
    M1_INPUT_SCHEMA_VERSION,
)
from src.advantage_m1_zero_counterfactual_v3 import (
    ZERO_COUNTERFACTUAL_MODEL_VERSION,
    ZERO_VARIANTS,
    AdvantageM1ZeroCounterfactualV3,
)
from src.canonical_observation_v2 import AvailabilityState
from src.event_provisional_oof_v1 import FoldPlan, fixed_fold_plan


TRAINING_VERSION = "advantage-m1-zero-counterfactual-oof-training/v3"
PLAN_VERSION = "advantage-m1-zero-counterfactual-oof-plan/v3"
COMPLETE_VERSION = "advantage-m1-zero-counterfactual-oof-complete/v3"
DEFAULT_SEEDS = legacy.DEFAULT_SEEDS
REPO_ROOT = Path(__file__).resolve().parents[1]
_KNOWN_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN)
_KNOWN_ZERO_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.KNOWN_ZERO)
_FAULT_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.INTEGRITY_FAULT)


class ZeroCounterfactualTrainingV3Error(RuntimeError):
    """V3学習、入力dataset、random controlの固定契約に違反した。"""


@dataclass(frozen=True, slots=True)
class CanonicalSamplesV3:
    """V2 datasetの学習列と行同定・監査列。"""

    boards: np.ndarray
    queues: np.ndarray
    ledger_values: np.ndarray
    ledger_availability: np.ndarray
    labels: np.ndarray
    source_groups: np.ndarray
    game_keys: np.ndarray
    weights: np.ndarray
    folds: np.ndarray
    state_ids: np.ndarray
    available_ms: np.ndarray
    online_segment_index: np.ndarray
    ledger_usable: np.ndarray
    projected_input_digest: np.ndarray
    projected_gate_status: np.ndarray

    def subset(self, indices: np.ndarray) -> CanonicalSamplesV3:
        return CanonicalSamplesV3(*(value[indices] for value in (
            self.boards, self.queues, self.ledger_values,
            self.ledger_availability, self.labels, self.source_groups,
            self.game_keys, self.weights, self.folds, self.state_ids,
            self.available_ms, self.online_segment_index, self.ledger_usable,
            self.projected_input_digest, self.projected_gate_status,
        )))


@dataclass(frozen=True, slots=True)
class RandomizedLedgerV3:
    samples: CanonicalSamplesV3
    effective_mask: np.ndarray
    structural_deranged_count: int
    singleton_count: int
    unchanged_value_count: int


@dataclass(frozen=True, slots=True)
class PreparedSplitV3:
    samples: CanonicalSamplesV3
    supported: np.ndarray
    receipt: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ResidualSelectionV3:
    model: AdvantageM1ZeroCounterfactualV3
    best_epoch: int
    fallback_to_m0: bool
    tune_raw: Mapping[str, float]
    tune_calibrated: Mapping[str, float]
    m0_state_sha256: str
    train_receipt: Mapping[str, int]
    tune_receipt: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class FoldOutputV3:
    raw: np.ndarray
    calibrated: np.ndarray
    record: Mapping[str, Any]


@dataclass(slots=True)
class OOFAccumulatorV3:
    raw: np.ndarray
    calibrated: np.ndarray
    records: list[Mapping[str, Any]]


class PreparedDatasetV3(Dataset):
    """V2 canonical datasetへ行別residual supportを加える。"""

    def __init__(self, prepared: PreparedSplitV3, *, augment: bool, seed: int) -> None:
        self.inner = legacy.CanonicalDataset(prepared.samples, augment=augment, seed=seed)
        self.supported = prepared.supported

    def __len__(self) -> int:
        return len(self.supported)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (*self.inner[index], torch.tensor(bool(self.supported[index])))


def load_canonical_dataset_v2(
    root: Path,
) -> tuple[CanonicalSamplesV3, dict[str, Any]]:
    """排他的hashとV2 schemaを検査し、6列datasetを読む。"""

    manifest_path, data_path = root / "manifest.json", root / "dataset.npz"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads((root / "COMPLETE").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ZeroCounterfactualTrainingV3Error("dataset receiptを読めません") from error
    _validate_dataset_receipts(manifest, complete, manifest_path, data_path)
    try:
        with np.load(data_path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as error:
        raise ZeroCounterfactualTrainingV3Error("dataset NPZを読めません") from error
    _validate_array_dtypes(arrays)
    samples = _samples_from_arrays(arrays, manifest)
    _validate_samples(samples, manifest, set(arrays))
    return samples, manifest


def _validate_dataset_receipts(
    manifest: Any, complete: Any, manifest_path: Path, data_path: Path,
) -> None:
    if not isinstance(manifest, dict) or not isinstance(complete, dict):
        raise ZeroCounterfactualTrainingV3Error("dataset receiptはobject必須です")
    if manifest.get("format_version") != builder.FORMAT_VERSION:
        raise ZeroCounterfactualTrainingV3Error("dataset format versionが不正です")
    if manifest.get("not_production") is not True:
        raise ZeroCounterfactualTrainingV3Error("datasetはnot_production必須です")
    if complete.get("format_version") != builder.COMPLETE_VERSION:
        raise ZeroCounterfactualTrainingV3Error("dataset COMPLETE versionが不正です")
    if complete.get("manifest_sha256") != base.file_sha256(manifest_path):
        raise ZeroCounterfactualTrainingV3Error("dataset manifest hashが一致しません")
    if complete.get("dataset_sha256") != base.file_sha256(data_path):
        raise ZeroCounterfactualTrainingV3Error("dataset NPZ hashが一致しません")


def _samples_from_arrays(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any],
) -> CanonicalSamplesV3:
    groups = np.asarray(arrays["source_groups"])
    folds = np.asarray([_fold_map(manifest)[str(value)] for value in groups], dtype=np.int8)
    return CanonicalSamplesV3(
        boards=np.asarray(arrays["boards"], dtype=np.int8),
        queues=np.asarray(arrays["queues"], dtype=np.int8),
        ledger_values=np.asarray(arrays["ledger_values"], dtype=np.float32),
        ledger_availability=np.asarray(arrays["ledger_availability"], dtype=np.float32),
        labels=np.asarray(arrays["labels"], dtype=np.float32), source_groups=groups,
        game_keys=np.asarray(arrays["game_keys"]),
        weights=np.asarray(arrays["weights"], dtype=np.float32), folds=folds,
        state_ids=np.asarray(arrays["state_ids"]),
        available_ms=np.asarray(arrays["available_ms"], dtype=np.int64),
        online_segment_index=np.asarray(arrays["online_segment_index"], dtype=np.int32),
        ledger_usable=np.asarray(arrays["ledger_usable"], dtype=np.bool_),
        projected_input_digest=np.asarray(arrays["projected_input_digest"]),
        projected_gate_status=np.asarray(arrays["projected_gate_status"]),
    )


def _validate_array_dtypes(arrays: Mapping[str, np.ndarray]) -> None:
    expected = {
        "boards": np.dtype(np.int8), "queues": np.dtype(np.int8),
        "ledger_values": np.dtype(np.float32),
        "ledger_availability": np.dtype(np.float32),
        "labels": np.dtype(np.float32), "weights": np.dtype(np.float32),
        "available_ms": np.dtype(np.int64),
        "online_segment_index": np.dtype(np.int32),
        "ledger_usable": np.dtype(np.bool_),
    }
    if any(name not in arrays or arrays[name].dtype != dtype for name, dtype in expected.items()):
        raise ZeroCounterfactualTrainingV3Error("dataset array dtypeが一致しません")
    strings = (
        "source_groups", "game_keys", "state_ids",
        "projected_input_digest", "projected_gate_status",
    )
    if any(name not in arrays or arrays[name].dtype.kind not in {"U", "S"} for name in strings):
        raise ZeroCounterfactualTrainingV3Error("dataset文字列列のdtypeが不正です")


def _fold_map(manifest: Mapping[str, Any]) -> dict[str, int]:
    groups = tuple(str(value) for value in manifest.get("source_group_ids", ()))
    sources = tuple(manifest.get("sources", ()))
    if len(groups) != len(sources) or len(set(groups)) != len(groups):
        raise ZeroCounterfactualTrainingV3Error("source group/fold台帳が不正です")
    result = {
        group: int(row["partition_fold"])
        for group, row in zip(groups, sources, strict=True)
    }
    if set(result.values()) != set(range(1, 7)):
        raise ZeroCounterfactualTrainingV3Error("固定fold 1〜6が揃っていません")
    return result


def source_group_fold_mapping_sha256(manifest: Mapping[str, Any]) -> str:
    """source-group→fold台帳を順序非依存のcanonical hashにする。"""
    mapping = dict(sorted(_fold_map(manifest).items()))
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_identity_receipt(
    root: Path, manifest: Mapping[str, Any],
) -> dict[str, str]:
    """foldを含むdataset現物の再照合用receiptを返す。"""
    return {
        "dataset_manifest_sha256": base.file_sha256(root / "manifest.json"),
        "dataset_complete_sha256": base.file_sha256(root / "COMPLETE"),
        "source_group_fold_mapping_sha256": source_group_fold_mapping_sha256(manifest),
    }


def _validate_samples(
    samples: CanonicalSamplesV3, manifest: Mapping[str, Any], names: set[str],
) -> None:
    expected = set(CanonicalSamplesV3.__dataclass_fields__) - {"folds"}
    if names != expected:
        raise ZeroCounterfactualTrainingV3Error("dataset NPZ schemaが一致しません")
    count = len(samples.labels)
    arrays = tuple(getattr(samples, name) for name in CanonicalSamplesV3.__dataclass_fields__)
    if count == 0 or any(len(value) != count for value in arrays):
        raise ZeroCounterfactualTrainingV3Error("dataset行数が不正です")
    if samples.boards.shape[1:] != (2, 13, 6) or samples.queues.shape[1:] != (2, 4):
        raise ZeroCounterfactualTrainingV3Error("盤面またはqueue shapeが不正です")
    if samples.ledger_values.shape[1:] != (2, LEDGER_FIELD_COUNT):
        raise ZeroCounterfactualTrainingV3Error("6列ledger value shapeが不正です")
    _validate_numpy_availability(samples.ledger_availability, count)
    _validate_sample_values(samples, manifest)


def _validate_sample_values(
    samples: CanonicalSamplesV3, manifest: Mapping[str, Any],
) -> None:
    count = len(samples.labels)
    if not np.isfinite(samples.ledger_values).all():
        raise ZeroCounterfactualTrainingV3Error("ledger valuesは有限値必須です")
    if np.any(samples.ledger_values < 0.0) or np.any(samples.ledger_values > 1.0):
        raise ZeroCounterfactualTrainingV3Error("ledger valuesは0..1必須です")
    if not np.isfinite(samples.weights).all() or np.any(samples.weights <= 0):
        raise ZeroCounterfactualTrainingV3Error("weightは有限な正値必須です")
    if not np.logical_or(samples.labels == 0.0, samples.labels == 1.0).all():
        raise ZeroCounterfactualTrainingV3Error("labelは0/1必須です")
    state_ids = tuple(str(value) for value in samples.state_ids)
    if any(not value for value in state_ids) or len(set(state_ids)) != count:
        raise ZeroCounterfactualTrainingV3Error("state_idは一意必須です")
    if int(manifest.get("state_count", -1)) != count:
        raise ZeroCounterfactualTrainingV3Error("manifest state_countが一致しません")
    if manifest.get("m1_input_schema_version") != M1_INPUT_SCHEMA_VERSION:
        raise ZeroCounterfactualTrainingV3Error("M1 input schema versionが不正です")
    if not np.array_equal(samples.ledger_usable, _base_supported_mask(samples)):
        raise ZeroCounterfactualTrainingV3Error("ledger usable receiptがtensorと不一致です")
    if int(manifest.get("ledger_usable_count", -1)) != int(samples.ledger_usable.sum()):
        raise ZeroCounterfactualTrainingV3Error("manifest ledger usable件数が不一致です")
    _validate_projected_join(samples, manifest)


def _validate_projected_join(
    samples: CanonicalSamplesV3, manifest: Mapping[str, Any],
) -> None:
    counts = {str(key): int(value) for key, value in Counter(
        str(item) for item in samples.projected_gate_status
    ).items()}
    if counts != manifest.get("projected_join_counts"):
        raise ZeroCounterfactualTrainingV3Error("projected join件数がmanifestと不一致です")
    for digest, status in zip(
        samples.projected_input_digest, samples.projected_gate_status, strict=True,
    ):
        text, gate = str(digest), str(status)
        if gate == builder.PROJECTED_NOT_JOINED:
            if text:
                raise ZeroCounterfactualTrainingV3Error("未join行にprojected digestがあります")
        elif len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
            raise ZeroCounterfactualTrainingV3Error("join済みprojected digestが不正です")


def randomized_ledger_rows_v3(
    samples: CanonicalSamplesV3, seed: int,
) -> RandomizedLedgerV3:
    """availability stratum内で左右込みledger行を固定点なしに置換する。"""

    count = len(samples.labels)
    if count == 0:
        raise ZeroCounterfactualTrainingV3Error("random control splitが空です")
    base_supported = _base_supported_mask(samples)
    patterns = samples.ledger_availability.reshape(count, -1)
    unique, strata = np.unique(patterns, axis=0, return_inverse=True)
    randomized = np.array(samples.ledger_values, copy=True)
    structural = np.zeros(count, dtype=bool)
    singleton = np.zeros(count, dtype=bool)
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        if len(indices) < 2:
            singleton[indices] = True
            continue
        permutation = _sattolo(indices, _stratum_seed(seed, unique[stratum]))
        randomized[indices], structural[indices] = samples.ledger_values[permutation], True
    changed = np.any(randomized != samples.ledger_values, axis=(1, 2))
    return RandomizedLedgerV3(
        replace(samples, ledger_values=np.ascontiguousarray(randomized)),
        structural & changed & base_supported, int(structural.sum()),
        int(singleton.sum()), int(np.count_nonzero(structural & ~changed)),
    )


def _sattolo(indices: np.ndarray, seed: int) -> np.ndarray:
    permutation = np.array(indices, copy=True)
    rng = np.random.default_rng(seed)
    for cursor in range(len(permutation) - 1, 0, -1):
        other = int(rng.integers(0, cursor))
        permutation[cursor], permutation[other] = permutation[other], permutation[cursor]
    if np.array_equal(permutation, indices):
        raise ZeroCounterfactualTrainingV3Error("derangement生成に失敗しました")
    return permutation


def _stratum_seed(seed: int, pattern: np.ndarray) -> int:
    direct = np.ascontiguousarray(pattern).tobytes()
    swapped = np.ascontiguousarray(pattern.reshape(2, -1)[::-1]).tobytes()
    canonical = min(direct, swapped)
    digest = hashlib.sha256(str(seed).encode("ascii") + b"\0" + canonical).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _base_supported_mask(samples: CanonicalSamplesV3) -> np.ndarray:
    masks = samples.ledger_availability
    _validate_numpy_availability(masks, len(samples.labels))
    if np.any(masks[..., _FAULT_INDEX] > 0.5):
        raise ZeroCounterfactualTrainingV3Error("integrity faultを学習へfallbackできません")
    pending = masks[:, :, 0]
    available = pending[:, :, _KNOWN_INDEX] + pending[:, :, _KNOWN_ZERO_INDEX]
    return np.asarray((available > 0.5).all(axis=1), dtype=bool)


def _validate_numpy_availability(masks: np.ndarray, count: int) -> None:
    expected = (count, 2, LEDGER_FIELD_COUNT, AVAILABILITY_COUNT)
    if masks.dtype != np.float32 or masks.shape != expected:
        raise ZeroCounterfactualTrainingV3Error("availability shape/dtypeが不正です")
    if not np.isfinite(masks).all():
        raise ZeroCounterfactualTrainingV3Error("availabilityは有限値必須です")
    if not np.logical_or(masks == 0.0, masks == 1.0).all():
        raise ZeroCounterfactualTrainingV3Error("availabilityは0/1必須です")
    if not np.array_equal(masks.sum(axis=-1), np.ones(masks.shape[:-1])):
        raise ZeroCounterfactualTrainingV3Error("availabilityは厳密one-hot必須です")


def _prepare_split(
    samples: CanonicalSamplesV3, variant: str, seed: int,
) -> PreparedSplitV3:
    if variant != "random_control":
        supported = _base_supported_mask(samples)
        return PreparedSplitV3(samples, supported, _random_receipt(supported))
    result = randomized_ledger_rows_v3(samples, seed)
    receipt = _random_receipt(
        result.effective_mask, result.structural_deranged_count,
        result.singleton_count, result.unchanged_value_count,
    )
    return PreparedSplitV3(result.samples, result.effective_mask, receipt)


def _random_receipt(
    supported: np.ndarray, structural: int = 0, singleton: int = 0, unchanged: int = 0,
) -> dict[str, int]:
    return {
        "row_count": len(supported), "effective_count": int(supported.sum()),
        "structural_deranged_count": structural, "singleton_count": singleton,
        "unchanged_value_count": unchanged,
    }


def _new_model(
    anchor: frozen.M0Anchor, variant: str, seed: int, device: torch.device,
) -> AdvantageM1ZeroCounterfactualV3:
    base._set_seed(seed)
    m0 = AdvantageM0CurrentCNNV2().to(device)
    m0.load_state_dict(anchor.model.state_dict())
    model = AdvantageM1ZeroCounterfactualV3(m0, variant).to(device)
    _assert_m0_unchanged(model, anchor.state_sha256)
    return model


def _assert_m0_unchanged(
    model: AdvantageM1ZeroCounterfactualV3, expected_sha256: str,
) -> None:
    if frozen.model_state_sha256(model.m0) != expected_sha256:
        raise ZeroCounterfactualTrainingV3Error("学習中に凍結M0 parameterが変化しました")


def _trainable_parameters(
    model: AdvantageM1ZeroCounterfactualV3,
) -> list[torch.nn.Parameter]:
    parameters = [item for item in model.parameters() if item.requires_grad]
    m0_ids = {id(item) for item in model.m0.parameters()}
    if not parameters or any(id(item) in m0_ids for item in parameters):
        raise ZeroCounterfactualTrainingV3Error("optimizerへ凍結M0が混入しています")
    return parameters


def _loader(prepared: PreparedSplitV3, args: argparse.Namespace, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        PreparedDatasetV3(prepared, augment=True, seed=seed),
        batch_size=args.batch_size, shuffle=True, generator=generator,
    )


def _train_epoch(
    model: AdvantageM1ZeroCounterfactualV3, loader: DataLoader,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    model.train()
    total = denominator = 0.0
    for batch in loader:
        moved = tuple(value.to(device) for value in batch)
        optimizer.zero_grad(set_to_none=True)
        output = model(moved[0], moved[1], moved[2], moved[3], moved[6])
        loss = equal_game_weighted_bce(output, moved[4], moved[5])
        loss.backward()
        optimizer.step()
        total += float((loss.detach() * moved[5].sum()).cpu())
        denominator += float(moved[5].sum().cpu())
    return total / denominator


def _predict(
    model: AdvantageM1ZeroCounterfactualV3, prepared: PreparedSplitV3,
    args: argparse.Namespace, device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        PreparedDatasetV3(prepared, augment=False, seed=0), batch_size=args.batch_size,
    )
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = tuple(value.to(device) for value in batch)
            predicted = model(moved[0], moved[1], moved[2], moved[3], moved[6])
            output.append(predicted.raw_probability.cpu().numpy())
    return np.concatenate(output)


def _initial_selection(
    model: AdvantageM1ZeroCounterfactualV3, tune: PreparedSplitV3,
    anchor: frozen.M0Anchor, args: argparse.Namespace, device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    baseline = frozen._predict_m0(anchor.model, tune.samples, args, device)
    candidate = _predict(model, tune, args, device)
    if not np.array_equal(candidate, baseline):
        raise ZeroCounterfactualTrainingV3Error("epoch 0がM0とbit-identicalではありません")
    calibrated = legacy.calibrate(baseline, anchor.slope)
    return frozen._cpu_state(model), legacy.metrics(tune.samples, calibrated)[
        "game_equal_log_loss"
    ]


def _fit_residual(
    samples: CanonicalSamplesV3, plan: FoldPlan, variant: str, seed: int,
    anchor: frozen.M0Anchor, args: argparse.Namespace, device: torch.device,
) -> ResidualSelectionV3:
    train = _prepare_split(
        samples.subset(np.isin(samples.folds, plan.train_folds)), variant, seed + 11,
    )
    tune = _prepare_split(samples.subset(samples.folds == plan.tune_fold), variant, seed + 12)
    model = _new_model(anchor, variant, seed, device)
    best_state, best_loss = _initial_selection(model, tune, anchor, args, device)
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    best, best_epoch = _train_loop(
        model, train, tune, anchor, args, device, optimizer, seed, best_state, best_loss,
    )
    return _restore_selection(
        model, best, best_epoch, tune, anchor, args, device,
        train.receipt, tune.receipt,
    )


def _train_loop(
    model: AdvantageM1ZeroCounterfactualV3, train: PreparedSplitV3,
    tune: PreparedSplitV3, anchor: frozen.M0Anchor, args: argparse.Namespace,
    device: torch.device, optimizer: torch.optim.Optimizer, seed: int,
    best_state: dict[str, torch.Tensor], best_loss: float,
) -> tuple[dict[str, torch.Tensor], int]:
    best_epoch = stale = 0
    loader = _loader(train, args, seed)
    for epoch in range(1, args.epochs + 1):
        _train_epoch(model, loader, optimizer, device)
        _assert_m0_unchanged(model, anchor.state_sha256)
        raw = _predict(model, tune, args, device)
        loss = legacy.metrics(tune.samples, legacy.calibrate(raw, anchor.slope))[
            "game_equal_log_loss"
        ]
        if loss < best_loss:
            best_state, best_loss, best_epoch, stale = frozen._cpu_state(model), loss, epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    return best_state, best_epoch


def _restore_selection(
    model: AdvantageM1ZeroCounterfactualV3, state: Mapping[str, torch.Tensor],
    epoch: int, tune: PreparedSplitV3, anchor: frozen.M0Anchor,
    args: argparse.Namespace, device: torch.device,
    train_receipt: Mapping[str, int], tune_receipt: Mapping[str, int],
) -> ResidualSelectionV3:
    model.load_state_dict(state)
    _assert_m0_unchanged(model, anchor.state_sha256)
    raw = _predict(model, tune, args, device)
    calibrated = legacy.calibrate(raw, anchor.slope)
    if epoch == 0:
        baseline = frozen._predict_m0(anchor.model, tune.samples, args, device)
        if not np.array_equal(raw, baseline):
            raise ZeroCounterfactualTrainingV3Error("M0 fallbackがbit-identicalではありません")
    return ResidualSelectionV3(
        model, epoch, epoch == 0, legacy.metrics(tune.samples, raw),
        legacy.metrics(tune.samples, calibrated),
        frozen.model_state_sha256(model.m0), train_receipt, tune_receipt,
    )


def _run_fold(
    samples: CanonicalSamplesV3, plan: FoldPlan, seed: int,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, FoldOutputV3]:
    fold_seed = seed + plan.eval_fold * 100
    anchor = frozen._fit_m0_anchor(samples, plan, fold_seed, args, device)
    evaluation = samples.subset(samples.folds == plan.eval_fold)
    baseline, m0_name, m0_hash = frozen._base_fold_output(
        anchor, evaluation, seed, plan, args, device,
    )
    output = {"m0": FoldOutputV3(baseline.raw, baseline.calibrated, baseline.record)}
    for variant in args.variants:
        selection = _fit_residual(samples, plan, variant, fold_seed, anchor, args, device)
        prepared = _prepare_split(evaluation, variant, fold_seed + 13)
        output[_result_variant(variant)] = _residual_output(
            selection, prepared, anchor, seed, plan, args, device, m0_name, m0_hash,
        )
    return output


def _residual_output(
    selection: ResidualSelectionV3, evaluation: PreparedSplitV3,
    anchor: frozen.M0Anchor, seed: int, plan: FoldPlan,
    args: argparse.Namespace, device: torch.device,
    m0_name: str, m0_sha256: str,
) -> FoldOutputV3:
    baseline = frozen._predict_m0(anchor.model, evaluation.samples, args, device)
    raw = baseline.copy() if selection.fallback_to_m0 else _predict(
        selection.model, evaluation, args, device,
    )
    calibrated = legacy.calibrate(raw, anchor.slope)
    name = f"m1_zero_{selection.model.variant}__seed_{seed}__fold_{plan.eval_fold}.pt"
    path = frozen._save_torch_exclusive(args.output_root / name, {
        "variant": selection.model.variant, "seed": seed, "fold": plan.eval_fold,
        "not_production": True,
        "model_version": selection.model.model_version,
        "input_schema_version": M1_INPUT_SCHEMA_VERSION,
        "fixed_m0_symmetric_platt_slope": anchor.slope,
        "fallback_to_m0": selection.fallback_to_m0,
        "m0_state_sha256": anchor.state_sha256,
        "state_dict": frozen._cpu_state(selection.model),
    })
    record = _residual_record(
        selection, evaluation, anchor, raw, name, base.file_sha256(path),
        plan, m0_name, m0_sha256,
    )
    return FoldOutputV3(raw, calibrated, record)


def _residual_record(
    selection: ResidualSelectionV3, evaluation: PreparedSplitV3,
    anchor: frozen.M0Anchor, raw: np.ndarray, model_name: str,
    model_sha256: str, plan: FoldPlan, m0_name: str, m0_sha256: str,
) -> dict[str, Any]:
    return {
        "eval_fold": plan.eval_fold, "tune_fold": plan.tune_fold,
        "train_folds": list(plan.train_folds), "best_epoch": selection.best_epoch,
        "fallback_to_m0": selection.fallback_to_m0,
        "fixed_m0_symmetric_platt_slope": anchor.slope,
        "train_random_receipt": dict(selection.train_receipt),
        "tune_random_receipt": dict(selection.tune_receipt),
        "eval_random_receipt": dict(evaluation.receipt),
        "tune_raw": dict(selection.tune_raw),
        "tune_calibrated": dict(selection.tune_calibrated),
        "eval_raw": legacy.metrics(evaluation.samples, raw),
        "eval_calibrated": legacy.metrics(
            evaluation.samples, legacy.calibrate(raw, anchor.slope),
        ),
        "model": model_name, "model_sha256": model_sha256,
        "model_version": ZERO_COUNTERFACTUAL_MODEL_VERSION,
        "input_schema_version": M1_INPUT_SCHEMA_VERSION,
        "m0_model": m0_name, "m0_model_sha256": m0_sha256,
        "m0_state_sha256_before": anchor.state_sha256,
        "m0_state_sha256_after": selection.m0_state_sha256,
    }


def _result_variant(variant: str) -> str:
    return f"m1_zero_{variant}"


def _new_accumulators(
    count: int, variants: Sequence[str],
) -> dict[str, OOFAccumulatorV3]:
    names = ("m0", *(_result_variant(value) for value in variants))
    return {name: OOFAccumulatorV3(
        np.full(count, np.nan, dtype=np.float32),
        np.full(count, np.nan, dtype=np.float32), [],
    ) for name in names}


def _assign(
    accumulator: OOFAccumulatorV3, indices: np.ndarray, output: FoldOutputV3,
) -> None:
    if len(output.raw) != len(indices) or len(output.calibrated) != len(indices):
        raise ZeroCounterfactualTrainingV3Error("OOF予測の行数が一致しません")
    if not np.isfinite(output.raw).all() or not np.isfinite(output.calibrated).all():
        raise ZeroCounterfactualTrainingV3Error("fold OOF予測は全行有限値必須です")
    if (np.isfinite(accumulator.raw[indices]).any()
            or np.isfinite(accumulator.calibrated[indices]).any()):
        raise ZeroCounterfactualTrainingV3Error("同じOOF行へ重複代入しています")
    accumulator.raw[indices], accumulator.calibrated[indices] = output.raw, output.calibrated
    accumulator.records.append(output.record)


def _finalize(
    samples: CanonicalSamplesV3, accumulator: OOFAccumulatorV3,
    expected_mask: np.ndarray,
) -> dict[str, Any]:
    raw_mask, calibrated_mask = np.isfinite(accumulator.raw), np.isfinite(accumulator.calibrated)
    if not np.array_equal(raw_mask, calibrated_mask):
        raise ZeroCounterfactualTrainingV3Error("raw/calibrated finite maskが一致しません")
    if not np.array_equal(raw_mask, expected_mask):
        raise ZeroCounterfactualTrainingV3Error("requested foldsのOOF coverageが一致しません")
    if not raw_mask.any():
        raise ZeroCounterfactualTrainingV3Error("完成したOOF予測がありません")
    subset = samples.subset(raw_mask)
    return {
        "raw": accumulator.raw, "calibrated": accumulator.calibrated,
        "records": accumulator.records,
        "metrics_raw": legacy.metrics(subset, accumulator.raw[raw_mask]),
        "metrics_calibrated": legacy.metrics(subset, accumulator.calibrated[raw_mask]),
    }


def _run_seed(
    samples: CanonicalSamplesV3, plans: Mapping[int, FoldPlan], seed: int,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, dict[str, Any]]:
    accumulators = _new_accumulators(len(samples.labels), args.variants)
    for fold in args.folds:
        outputs = _run_fold(samples, plans[fold], seed, args, device)
        indices = np.flatnonzero(samples.folds == fold)
        for name, output in outputs.items():
            _assign(accumulators[name], indices, output)
    expected = np.isin(samples.folds, args.folds)
    return {
        f"{name}__seed_{seed}": _finalize(samples, accumulator, expected)
        for name, accumulator in accumulators.items()
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    """V3 OOFを既存成果物と別の新規rootへ保存する。"""

    if args.output_root.exists():
        raise ZeroCounterfactualTrainingV3Error(f"出力先は新規必須です: {args.output_root}")
    samples, manifest = load_canonical_dataset_v2(args.dataset_root)
    _base_supported_mask(samples)
    plans = {plan.eval_fold: plan for plan in fixed_fold_plan(samples.folds)}
    args.output_root.mkdir(parents=True, exist_ok=False)
    plan = _plan(args, samples, manifest)
    plan_path = base._write_json_exclusive(args.output_root / "PLAN.json", plan)
    device = base._device(args.device)
    results: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        results.update(_run_seed(samples, plans, seed, args, device))
    return _write_results(args.output_root, samples, manifest, plan, plan_path, results)


def _plan(
    args: argparse.Namespace, samples: CanonicalSamplesV3,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    identity = dataset_identity_receipt(args.dataset_root, manifest)
    return {
        "format_version": PLAN_VERSION, "not_production": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_sha256": manifest["dataset"]["sha256"],
        "dataset_format_version": builder.FORMAT_VERSION,
        "input_schema_version": M1_INPUT_SCHEMA_VERSION,
        "model_version": ZERO_COUNTERFACTUAL_MODEL_VERSION,
        "state_count": len(samples.labels),
        "game_count": len(np.unique(samples.game_keys)),
        "source_count": len(np.unique(samples.source_groups)),
        "variants": list(args.variants), "seeds": list(args.seeds),
        "folds": list(args.folds), "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "fixed_outer_folds": 6,
        "residual_definition": "0.5*((F(board,actual)-F(board,zero))-swapped)",
        "random_policy": "split-local exact-availability-stratum Sattolo atomic-two-side-row",
        "fallback_policy": "unsupported/singleton/unchanged exact M0; integrity HOLD/FAULT",
        "calibration_policy": "paired M0 slope fixed for every residual candidate",
        "projected_inputs_used": False,
        "projected_join_role": manifest["projected_join_role"],
        **identity,
        "code_sha256": _code_hashes(),
    }


def _code_hashes() -> dict[str, str]:
    return {
        "trainer": base.file_sha256(Path(__file__)),
        "model": base.file_sha256(REPO_ROOT / "src/advantage_m1_zero_counterfactual_v3.py"),
        "tensorizer": base.file_sha256(REPO_ROOT / "src/advantage_m1_causal_ledger_v3.py"),
        "frozen_v1_trainer": base.file_sha256(
            REPO_ROOT / "scripts/train_advantage_m1_frozen_residual_v1.py"
        ),
        "m0": base.file_sha256(REPO_ROOT / "src/advantage_m0_current_cnn_v1.py"),
        "production_config": base.file_sha256(REPO_ROOT / "src/production_config.py"),
    }


def _write_results(
    root: Path, samples: CanonicalSamplesV3, manifest: Mapping[str, Any],
    plan: Mapping[str, Any], plan_path: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    predictions: dict[str, np.ndarray] = {"labels": samples.labels, "folds": samples.folds}
    serializable: dict[str, Any] = {}
    for key, result in results.items():
        predictions[f"{key}__raw"] = result["raw"]
        predictions[f"{key}__calibrated"] = result["calibrated"]
        serializable[key] = {
            name: value for name, value in result.items() if name not in {"raw", "calibrated"}
        }
    prediction_path = root / "oof_predictions.npz"
    legacy._save_npz_exclusive(prediction_path, predictions)
    report = {
        "format_version": TRAINING_VERSION, "not_production": True,
        "dataset_source_count": manifest["source_count"],
        "plan": dict(plan), "results": serializable,
    }
    result_path = base._write_json_exclusive(root / "results.json", report)
    base._write_json_exclusive(root / "COMPLETE", {
        "format_version": COMPLETE_VERSION,
        "plan_sha256": base.file_sha256(plan_path),
        "results_sha256": base.file_sha256(result_path),
        "predictions_sha256": base.file_sha256(prediction_path),
    })
    return report


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
    parser.add_argument("--variants", type=_csv_strings, default=ZERO_VARIANTS)
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
    if not args.variants or len(args.variants) != len(set(args.variants)):
        parser.error("variantsは重複なしの非空指定が必要です")
    if any(value not in ZERO_VARIANTS for value in args.variants):
        parser.error(f"variantsは{ZERO_VARIANTS}から指定してください")
    if (not args.seeds or len(args.seeds) != len(set(args.seeds))
            or not args.folds or len(args.folds) != len(set(args.folds))):
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
