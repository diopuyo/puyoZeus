"""clean46のM0を固定し、clean48でV3 residualだけをOOF学習する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import merge_advantage_m1_zero_counterfactual_v3 as merger
from scripts import train_advantage_m0_current_cnn_v1 as base
from scripts import train_advantage_m1_causal_ledger_v1 as legacy
from scripts import train_advantage_m1_frozen_residual_v1 as frozen
from scripts import train_advantage_m1_zero_counterfactual_v3 as v3
from src.advantage_m0_current_cnn_v1 import AdvantageM0CurrentCNNV2
from src.event_provisional_oof_v1 import FoldPlan, fixed_fold_plan
from src import projected_set_training_v1 as runtime


PUBLIC_SEEDS = (20260904, 20260905, 20260906)
DEFAULT_VARIANTS = ("values_and_masks", "random_control")
EXPECTED_ADDED_GROUPS = {"04Lb9BZCpP0": ("c82", 1), "UpnGj22itdA": ("c83", 3)}
EXPECTED_TARGET_RECEIPT = {
    "dataset_sha256": "89c57ffc665fd6c4c0944fe8b3ee0dee694e9edb4a7c1ddb53d9a35559d5f146",
    "dataset_manifest_sha256": "d9507b0c23636a1daeb4dce8504b6c4b9b4837ea1b32c56da49d83c121b27200",
    "dataset_complete_sha256": "816164ef06a61c61e8ffbc0f79cb56f3214878ab886750cbcadec803b56a2a3b",
}
EXPECTED_ANCHOR_DATASET_RECEIPT = {
    "dataset_sha256": "fcbb00988d901187cd23d9373e8bf719ead985f996d87feba3a5e7e7d17c580a",
    "dataset_manifest_sha256": "75e9e726e01b77ea932eecbfff8154536aa5f3b53cd42402e9b8448579f7405d",
    "dataset_complete_sha256": "68775da288277b543d44c968f0bb641cd2de8edf645232f7b3d79aeb618e9826",
    "source_group_fold_mapping_sha256": "e81f6d6e2c6cbb356e7dc206fc67887b1124327ae68523722ce44ca45b908347",
}
EXPECTED_ANCHOR_ROOT_HASHES = frozenset({
    (
        "2928fe7bc6da8e1461f63339a2af7aef5543cb7282dfce517696631bea2a7ada",
        "ee095a8bf713b4fe1917fce715e25e01dcac4890657726596be2fdfd2183245a",
    ),
    (
        "f54db1d9311cd4dbaa7286dc7337ad3f4480cf676917e7a4f490e2b5b488eacb",
        "6adc8dc67c43f41637db878d8f3e2da5dde526d3bf0639901f47f2d53a58cb6f",
    ),
})
EXPECTED_PRODUCTION_CONFIG_SHA256 = (
    "3fe3c2578b3196a60cffffcafa594c1f64d2a913bb0487932ac5cd8f6f86d376"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_WEIGHT_SCALE_RTOL = 2e-7
COMMON_WEIGHT_SCALE_RTOL_RATIONALE = (
    "float32 game-equal weightの量子化で観測したratio spread 1.497e-7を"
    "直上で包含し、非一様な再重み付けは拒否する上限"
)


class FixedM0AnchorTrainingError(RuntimeError):
    """外部M0固定またはclean46→48増分契約に違反した。"""


@dataclass(frozen=True, slots=True)
class M0AnchorReference:
    seed: int
    fold: int
    path: Path
    sha256: str
    state_sha256: str
    slope: float
    best_epoch: int
    tune_raw: Mapping[str, float]
    tune_calibrated: Mapping[str, float]
    result_root: Path
    results_sha256: str
    predictions_path: Path
    predictions_sha256: str


@dataclass(frozen=True, slots=True)
class M1ChampionReference:
    path: Path
    sha256: str
    model_state_sha256: str
    result_root: Path
    predictions_path: Path
    predictions_sha256: str
    record: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DatasetRelation:
    anchor_samples: v3.CanonicalSamplesV3
    anchor_manifest: Mapping[str, Any]
    anchor_index: Mapping[str, int]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ValidatedInputs:
    samples: v3.CanonicalSamplesV3
    manifest: Mapping[str, Any]
    registry: Mapping[tuple[int, int], M0AnchorReference]
    champion: M1ChampionReference
    anchor_receipts: Sequence[Mapping[str, Any]]
    relation: DatasetRelation
    weight_receipt: Mapping[str, Any]
    input_assets: Sequence[Mapping[str, str]]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixedM0AnchorTrainingError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise FixedM0AnchorTrainingError(f"JSON objectではありません: {path}")
    return value


def _validate_root_complete(root: Path, complete: Mapping[str, Any]) -> None:
    if complete.get("format_version") != merger.MERGE_COMPLETE_VERSION:
        raise FixedM0AnchorTrainingError(f"anchor COMPLETE形式が不正です: {root}")
    paths = {
        "plan_sha256": root / "PLAN.json",
        "results_sha256": root / "results.json",
        "predictions_sha256": root / "oof_predictions.npz",
    }
    for key, path in paths.items():
        if not path.is_file() or complete.get(key) != base.file_sha256(path):
            raise FixedM0AnchorTrainingError(f"anchor {key}が一致しません: {root}")


def _validate_anchor_plan(plan: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    expected = (
        ("format_version", merger.MERGE_PLAN_VERSION),
        ("source_plan_format_version", v3.PLAN_VERSION),
        ("source_results_format_version", v3.TRAINING_VERSION),
        ("source_complete_format_version", v3.COMPLETE_VERSION),
        ("dataset_format_version", v3.builder.FORMAT_VERSION),
        ("input_schema_version", v3.M1_INPUT_SCHEMA_VERSION),
        ("model_version", v3.ZERO_COUNTERFACTUAL_MODEL_VERSION),
        ("source_count", 46),
    )
    if any(plan.get(key) != value for key, value in expected):
        raise FixedM0AnchorTrainingError("anchor merged PLAN契約が不正です")
    if report.get("format_version") != merger.MERGE_RESULTS_VERSION:
        raise FixedM0AnchorTrainingError("anchor results形式が不正です")
    if report.get("not_production") is not True or report.get("plan") != plan:
        raise FixedM0AnchorTrainingError("anchor resultsのPLAN receiptが不正です")


def _anchor_dataset_receipt(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(plan.get(key, "")) for key in EXPECTED_ANCHOR_DATASET_RECEIPT
    }


def _record_metrics(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise FixedM0AnchorTrainingError(f"anchor {name}が不正です")
    result = {str(key): float(item) for key, item in value.items()}
    if any(not math.isfinite(item) for item in result.values()):
        raise FixedM0AnchorTrainingError(f"anchor {name}に非有限値があります")
    return result


def _reference_from_record(
    record: Mapping[str, Any], seed: int, root: Path,
    complete: Mapping[str, Any],
) -> M0AnchorReference:
    checkpoint = record.get("checkpoint_reference")
    if not isinstance(checkpoint, Mapping):
        raise FixedM0AnchorTrainingError("M0 checkpoint_referenceがありません")
    path, expected = Path(str(checkpoint.get("path", ""))), str(checkpoint.get("sha256", ""))
    slope = float(record.get("symmetric_platt_slope", math.nan))
    if not path.is_file() or base.file_sha256(path) != expected:
        raise FixedM0AnchorTrainingError(f"M0 checkpoint SHAが一致しません: {path}")
    if record.get("model") != path.name or record.get("model_sha256") != expected:
        raise FixedM0AnchorTrainingError("M0 recordのcheckpoint receiptが不正です")
    if not math.isfinite(slope) or slope <= 0.0:
        raise FixedM0AnchorTrainingError("M0 symmetric Platt slopeが不正です")
    return M0AnchorReference(
        seed, int(record.get("eval_fold", -1)), path, expected,
        str(record.get("m0_state_sha256", "")), slope,
        int(record.get("best_epoch", -1)), _record_metrics(record.get("tune_raw"), "tune_raw"),
        _record_metrics(record.get("tune_calibrated"), "tune_calibrated"), root,
        str(complete["results_sha256"]), root / "oof_predictions.npz",
        str(complete["predictions_sha256"]),
    )


def _add_m0_records(
    registry: dict[tuple[int, int], M0AnchorReference], report: Mapping[str, Any],
    root: Path, complete: Mapping[str, Any],
) -> None:
    results = report.get("results")
    if not isinstance(results, Mapping):
        raise FixedM0AnchorTrainingError("anchor result集合が不正です")
    for key, value in results.items():
        if not str(key).startswith("m0__seed_"):
            continue
        if not isinstance(value, Mapping) or not isinstance(value.get("records"), list):
            raise FixedM0AnchorTrainingError(f"M0 resultが不正です: {key}")
        seed = int(str(key).rsplit("_", 1)[-1])
        for record in value["records"]:
            reference = _reference_from_record(record, seed, root, complete)
            lookup = (reference.seed, reference.fold)
            if lookup in registry:
                raise FixedM0AnchorTrainingError(f"M0 anchorが重複しています: {lookup}")
            registry[lookup] = reference


def _validate_checkpoint_payload(
    checkpoint: Mapping[str, Any], reference: M0AnchorReference,
) -> Mapping[str, torch.Tensor]:
    expected = (
        checkpoint.get("variant") == "m0",
        int(checkpoint.get("seed", -1)) == reference.seed,
        int(checkpoint.get("fold", -1)) == reference.fold,
        checkpoint.get("model_version") == AdvantageM0CurrentCNNV2.model_version,
        checkpoint.get("m0_state_sha256") == reference.state_sha256,
        checkpoint.get("symmetric_platt_slope") == reference.slope,
        isinstance(checkpoint.get("state_dict"), Mapping),
    )
    if not all(expected):
        raise FixedM0AnchorTrainingError("M0 checkpoint metadataが不正です")
    return checkpoint["state_dict"]


def load_m0_anchor(
    reference: M0AnchorReference, device: torch.device,
) -> frozen.M0Anchor:
    """SHAとmetadataを再検査し、seed×eval foldのM0を凍結ロードする。"""

    if base.file_sha256(reference.path) != reference.sha256:
        raise FixedM0AnchorTrainingError(f"M0 checkpointが変更されました: {reference.path}")
    try:
        checkpoint = torch.load(reference.path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise FixedM0AnchorTrainingError("M0 checkpointを読めません") from error
    if not isinstance(checkpoint, Mapping):
        raise FixedM0AnchorTrainingError("M0 checkpointはmapping必須です")
    state = _validate_checkpoint_payload(checkpoint, reference)
    model = AdvantageM0CurrentCNNV2().to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise FixedM0AnchorTrainingError("M0 state/model versionが一致しません") from error
    model.eval()
    if frozen.model_state_sha256(model) != reference.state_sha256:
        raise FixedM0AnchorTrainingError("M0 checkpoint state SHAが一致しません")
    return frozen.M0Anchor(
        model, reference.slope, reference.best_epoch, reference.tune_raw,
        reference.tune_calibrated, reference.state_sha256,
    )


def _root_receipt(root: Path, complete: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root": str(root.resolve()),
        "complete_sha256": base.file_sha256(root / "COMPLETE"),
        "plan_sha256": str(complete["plan_sha256"]),
        "results_sha256": str(complete["results_sha256"]),
        "predictions_sha256": str(complete["predictions_sha256"]),
    }


def load_anchor_registry(
    roots: Sequence[Path], *, enforce_preregistered: bool = True,
) -> tuple[dict[tuple[int, int], M0AnchorReference], list[dict[str, Any]]]:
    """clean46 merged rootから18件の検証済みM0台帳を構築する。"""

    if not roots or len({root.resolve() for root in roots}) != len(roots):
        raise FixedM0AnchorTrainingError("anchor rootは重複なしの非空指定が必要です")
    registry: dict[tuple[int, int], M0AnchorReference] = {}
    receipts: list[dict[str, Any]] = []
    identities: set[tuple[tuple[str, str], ...]] = set()
    for root in roots:
        complete = _read_json(root / "COMPLETE")
        _validate_root_complete(root, complete)
        plan, report = _read_json(root / "PLAN.json"), _read_json(root / "results.json")
        _validate_anchor_plan(plan, report)
        identities.add(tuple(sorted(_anchor_dataset_receipt(plan).items())))
        _add_m0_records(registry, report, root, complete)
        receipts.append(_root_receipt(root, complete))
    _validate_registry(registry, receipts, identities, enforce_preregistered)
    return registry, sorted(receipts, key=lambda item: item["complete_sha256"])


def _validate_registry(
    registry: Mapping[tuple[int, int], M0AnchorReference],
    receipts: Sequence[Mapping[str, Any]], identities: set[tuple[tuple[str, str], ...]],
    enforce_preregistered: bool,
) -> None:
    expected = {(seed, fold) for seed in PUBLIC_SEEDS for fold in range(1, 7)}
    if set(registry) != expected:
        raise FixedM0AnchorTrainingError("M0 anchor 3 seed × 6 fold coverageが不正です")
    if len(identities) != 1:
        raise FixedM0AnchorTrainingError("anchor dataset identityがroot間で不一致です")
    identity = dict(next(iter(identities)))
    if enforce_preregistered and identity != EXPECTED_ANCHOR_DATASET_RECEIPT:
        raise FixedM0AnchorTrainingError("事前登録clean46 dataset receiptと一致しません")
    pairs = {(row["complete_sha256"], row["results_sha256"]) for row in receipts}
    if enforce_preregistered and pairs != EXPECTED_ANCHOR_ROOT_HASHES:
        raise FixedM0AnchorTrainingError("事前登録anchor root receiptと一致しません")
    for reference in registry.values():
        load_m0_anchor(reference, torch.device("cpu"))


def _champion_record(root: Path) -> Mapping[str, Any]:
    report = _read_json(root / "results.json")
    key = "m1_zero_values_and_masks__seed_20260904"
    result = report.get("results", {}).get(key)
    if not isinstance(result, Mapping) or not isinstance(result.get("records"), list):
        raise FixedM0AnchorTrainingError("固定clean46 champion resultがありません")
    records = [row for row in result["records"] if int(row.get("eval_fold", -1)) == 1]
    if len(records) != 1 or not isinstance(records[0], Mapping):
        raise FixedM0AnchorTrainingError("固定clean46 champion fold 1が一意ではありません")
    return records[0]


def _load_champion_model(
    path: Path, record: Mapping[str, Any], m0: M0AnchorReference,
    device: torch.device,
) -> tuple[v3.AdvantageM1ZeroCounterfactualV3, str]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise FixedM0AnchorTrainingError("clean46 champion checkpointを読めません") from error
    if not isinstance(checkpoint, Mapping):
        raise FixedM0AnchorTrainingError("clean46 champion checkpointはmapping必須です")
    expected = (
        checkpoint.get("variant") == "values_and_masks",
        checkpoint.get("seed") == 20260904, checkpoint.get("fold") == 1,
        checkpoint.get("not_production") is True,
        checkpoint.get("model_version") == v3.ZERO_COUNTERFACTUAL_MODEL_VERSION,
        checkpoint.get("input_schema_version") == v3.M1_INPUT_SCHEMA_VERSION,
        checkpoint.get("fixed_m0_symmetric_platt_slope") == m0.slope,
        checkpoint.get("m0_state_sha256") == m0.state_sha256,
        checkpoint.get("fallback_to_m0") == record.get("fallback_to_m0"),
        isinstance(checkpoint.get("state_dict"), Mapping),
    )
    if not all(expected):
        raise FixedM0AnchorTrainingError("clean46 champion checkpoint metadataが不正です")
    model = v3.AdvantageM1ZeroCounterfactualV3(
        AdvantageM0CurrentCNNV2(), "values_and_masks",
    ).to(device)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as error:
        raise FixedM0AnchorTrainingError("clean46 champion state/modelが不一致です") from error
    model.eval()
    return model, frozen.model_state_sha256(model)


def load_champion_reference(
    registry: Mapping[tuple[int, int], M0AnchorReference],
) -> M1ChampionReference:
    """事前固定したclean46 M1代表checkpointを検査して返す。"""

    m0 = registry[(20260904, 1)]
    root, record = m0.result_root, _champion_record(m0.result_root)
    checkpoint = record.get("checkpoint_reference")
    if not isinstance(checkpoint, Mapping):
        raise FixedM0AnchorTrainingError("clean46 champion checkpoint receiptがありません")
    path, expected = Path(str(checkpoint.get("path", ""))), str(checkpoint.get("sha256", ""))
    if not path.is_file() or base.file_sha256(path) != expected:
        raise FixedM0AnchorTrainingError("clean46 champion checkpoint SHAが不一致です")
    if record.get("model") != path.name or record.get("model_sha256") != expected:
        raise FixedM0AnchorTrainingError("clean46 champion model receiptが不正です")
    if (record.get("m0_model_sha256") != m0.sha256
            or record.get("m0_state_sha256_before") != m0.state_sha256
            or record.get("m0_state_sha256_after") != m0.state_sha256):
        raise FixedM0AnchorTrainingError("clean46 championのM0 linkが不正です")
    if (record.get("fixed_m0_symmetric_platt_slope") != m0.slope
            or record.get("model_version") != v3.ZERO_COUNTERFACTUAL_MODEL_VERSION
            or record.get("input_schema_version") != v3.M1_INPUT_SCHEMA_VERSION
            or int(record.get("best_epoch", -1)) < 0):
        raise FixedM0AnchorTrainingError("clean46 champion record metadataが不正です")
    _, state_sha = _load_champion_model(path, record, m0, torch.device("cpu"))
    return M1ChampionReference(
        path, expected, state_sha, root, root / "oof_predictions.npz",
        m0.predictions_sha256, record,
    )


def _source_rows(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    groups, sources = manifest.get("source_group_ids"), manifest.get("sources")
    if not isinstance(groups, list) or not isinstance(sources, list) or len(groups) != len(sources):
        raise FixedM0AnchorTrainingError("dataset source台帳が不正です")
    if len(set(str(value) for value in groups)) != len(groups):
        raise FixedM0AnchorTrainingError("dataset source groupが重複しています")
    if any(not isinstance(row, Mapping) for row in sources):
        raise FixedM0AnchorTrainingError("dataset source recordが不正です")
    return {str(group): row for group, row in zip(groups, sources, strict=True)}


def _validate_source_relation(
    anchor_manifest: Mapping[str, Any], target_manifest: Mapping[str, Any],
) -> tuple[set[str], dict[str, Mapping[str, Any]]]:
    anchor, target = _source_rows(anchor_manifest), _source_rows(target_manifest)
    added = set(target) - set(anchor)
    if set(anchor) - set(target) or added != set(EXPECTED_ADDED_GROUPS):
        raise FixedM0AnchorTrainingError("追加sourceはc82/c83の2 groupだけに限ります")
    if any(target[group] != row for group, row in anchor.items()):
        raise FixedM0AnchorTrainingError("clean46共通source receiptが変更されています")
    for group, (target_id, fold) in EXPECTED_ADDED_GROUPS.items():
        row = target[group]
        if str(row.get("target_id")) != target_id or int(row.get("partition_fold", -1)) != fold:
            raise FixedM0AnchorTrainingError(f"追加source fold契約が不正です: {group}")
    return added, target


def _validate_game_sources(samples: v3.CanonicalSamplesV3) -> None:
    observed: dict[str, str] = {}
    for game, group in zip(samples.game_keys, samples.source_groups, strict=True):
        key, source = str(game), str(group)
        if key in observed and observed[key] != source:
            raise FixedM0AnchorTrainingError(f"gameが複数sourceへ衝突しています: {key}")
        observed[key] = source


def _common_indices(
    anchor: v3.CanonicalSamplesV3, target: v3.CanonicalSamplesV3,
) -> tuple[dict[str, int], np.ndarray]:
    target_index = {str(value): index for index, value in enumerate(target.state_ids)}
    try:
        indices = np.asarray([target_index[str(value)] for value in anchor.state_ids])
    except KeyError as error:
        raise FixedM0AnchorTrainingError(f"clean46 stateがclean48にありません: {error}") from error
    return {str(value): index for index, value in enumerate(anchor.state_ids)}, indices


def _validate_exact_common(
    anchor: v3.CanonicalSamplesV3, target: v3.CanonicalSamplesV3,
    target_indices: np.ndarray,
) -> None:
    state_fields = set(v3.CanonicalSamplesV3.__dataclass_fields__) - {"weights"}
    for name in state_fields:
        left, right = getattr(anchor, name), getattr(target, name)[target_indices]
        if not np.array_equal(left, right):
            raise FixedM0AnchorTrainingError(f"clean46共通state列が変化しています: {name}")


def _common_weight_scale(
    anchor: v3.CanonicalSamplesV3, target: v3.CanonicalSamplesV3,
    target_indices: np.ndarray,
) -> float:
    old = anchor.weights.astype(np.float64)
    new = target.weights[target_indices].astype(np.float64)
    ratio = new / old
    scale = float(np.median(ratio))
    if not math.isfinite(scale) or scale <= 0.0:
        raise FixedM0AnchorTrainingError("共通state weight scaleが不正です")
    if not np.allclose(ratio, scale, rtol=COMMON_WEIGHT_SCALE_RTOL, atol=0.0):
        raise FixedM0AnchorTrainingError("共通state weightが一様な再正規化ではありません")
    return scale


def validate_dataset_relation(
    anchor: v3.CanonicalSamplesV3, anchor_manifest: Mapping[str, Any],
    target: v3.CanonicalSamplesV3, target_manifest: Mapping[str, Any],
) -> DatasetRelation:
    """clean48がclean46を厳密包含し、増分がc82/c83だけか検査する。"""

    if int(anchor_manifest.get("source_count", -1)) != 46:
        raise FixedM0AnchorTrainingError("anchor source_countは46必須です")
    if int(target_manifest.get("source_count", -1)) != 48:
        raise FixedM0AnchorTrainingError("target source_countは48必須です")
    added, _ = _validate_source_relation(anchor_manifest, target_manifest)
    anchor_index, target_indices = _common_indices(anchor, target)
    _validate_exact_common(anchor, target, target_indices)
    weight_scale = _common_weight_scale(anchor, target, target_indices)
    common = set(anchor_index)
    extra_mask = np.asarray([str(value) not in common for value in target.state_ids])
    extra_groups = {str(value) for value in target.source_groups[extra_mask]}
    if extra_groups != added or np.any(np.isin(target.source_groups[~extra_mask], tuple(added))):
        raise FixedM0AnchorTrainingError("追加stateのsource所属が不正です")
    _validate_game_sources(anchor)
    _validate_game_sources(target)
    old_games = {str(value) for value in anchor.game_keys}
    new_games = {str(value) for value in target.game_keys[extra_mask]}
    if old_games & new_games:
        raise FixedM0AnchorTrainingError("clean46と追加sourceでgame keyが衝突しています")
    receipt = {
        "anchor_state_count": len(anchor.labels), "target_state_count": len(target.labels),
        "exact_common_state_count": len(anchor.labels), "added_state_count": int(extra_mask.sum()),
        "added_groups": sorted(added), "game_collision_count": 0,
        "common_weight_policy": "uniform_global_renormalization_only",
        "common_weight_scale": weight_scale,
        "common_weight_scale_rtol": COMMON_WEIGHT_SCALE_RTOL,
        "common_weight_scale_rtol_rationale": COMMON_WEIGHT_SCALE_RTOL_RATIONALE,
    }
    return DatasetRelation(anchor, anchor_manifest, anchor_index, receipt)


def _weight_masks_and_anchor_indices(
    samples: v3.CanonicalSamplesV3, relation: DatasetRelation,
) -> tuple[np.ndarray, np.ndarray]:
    common = np.asarray([
        str(state_id) in relation.anchor_index for state_id in samples.state_ids
    ])
    anchor_indices = np.asarray([
        relation.anchor_index[str(state_id)]
        for state_id in samples.state_ids[common]
    ], dtype=np.int64)
    if int(common.sum()) != len(relation.anchor_samples.labels):
        raise FixedM0AnchorTrainingError("weight安定化のcommon state coverageが不正です")
    return common, anchor_indices


def _game_weight_rows(samples: v3.CanonicalSamplesV3) -> list[dict[str, Any]]:
    grouped: dict[str, list[float | int]] = {}
    for game_value, weight_value in zip(samples.game_keys, samples.weights, strict=True):
        game, weight = str(game_value), float(weight_value)
        current = grouped.get(game)
        if current is None:
            grouped[game] = [1, weight, weight]
            continue
        if np.float32(current[1]) != np.float32(weight_value):
            raise FixedM0AnchorTrainingError(f"game内weightが一定ではありません: {game}")
        current[0], current[2] = int(current[0]) + 1, float(current[2]) + weight
    return [{
        "game_key": game, "row_count": int(value[0]),
        "row_weight": float(value[1]), "total_weight": float(value[2]),
    } for game, value in sorted(grouped.items())]


def _validate_equal_game_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    totals = np.asarray([float(row["total_weight"]) for row in rows])
    expected = float(np.median(totals))
    bounds = []
    for row in rows:
        spacing = float(np.spacing(np.float32(row["row_weight"])))
        bounds.append(abs(spacing) * int(row["row_count"]) + 4 * np.finfo(np.float32).eps * expected)
    tolerance = float(max(bounds))
    spread = float(np.max(np.abs(totals - expected)))
    if not np.isfinite(totals).all() or spread > tolerance:
        raise FixedM0AnchorTrainingError("全game総weightが同一尺度ではありません")
    return {
        "expected_total": expected, "min_total": float(totals.min()),
        "max_total": float(totals.max()), "max_abs_deviation": spread,
        "float32_rounding_tolerance": tolerance,
    }


def stabilize_increment_weights(
    samples: v3.CanonicalSamplesV3, relation: DatasetRelation,
) -> tuple[v3.CanonicalSamplesV3, dict[str, Any]]:
    """clean48 weightをclean46のgame-equal尺度へメモリ上だけで戻す。"""

    common, anchor_indices = _weight_masks_and_anchor_indices(samples, relation)
    scale = float(relation.receipt["common_weight_scale"])
    stabilized = np.empty_like(samples.weights)
    stabilized[common] = relation.anchor_samples.weights[anchor_indices]
    stabilized[~common] = (
        samples.weights[~common].astype(np.float64) / scale
    ).astype(np.float32)
    if not np.array_equal(stabilized[common], relation.anchor_samples.weights[anchor_indices]):
        raise FixedM0AnchorTrainingError("common weightがclean46とbyte-exactではありません")
    output = replace(samples, weights=np.ascontiguousarray(stabilized))
    game_rows = _game_weight_rows(output)
    totals = _validate_equal_game_totals(game_rows)
    added_games = {str(value) for value in samples.game_keys[~common]}
    receipt = {
        "policy": "common_clean46_bytes_added_divide_audited_common_median_scale",
        "audited_common_median_scale": scale,
        "input_target_weight_sha256": _array_sha256(samples.weights),
        "stabilized_weight_sha256": _array_sha256(output.weights),
        "common_weight_sha256": _array_sha256(output.weights[common]),
        "common_weight_bytes_exact": True, "common_state_count": int(common.sum()),
        "added_state_count": int((~common).sum()), "all_game_totals": totals,
        "added_game_totals": [row for row in game_rows if row["game_key"] in added_games],
    }
    return output, receipt


def _validate_target_receipt(root: Path, manifest: Mapping[str, Any]) -> None:
    receipt = {"dataset_sha256": str(manifest.get("dataset", {}).get("sha256", ""))}
    receipt.update(v3.dataset_identity_receipt(root, manifest))
    selected = {key: receipt[key] for key in EXPECTED_TARGET_RECEIPT}
    if selected != EXPECTED_TARGET_RECEIPT:
        raise FixedM0AnchorTrainingError("事前登録clean48 dataset receiptと一致しません")


def _load_anchor_dataset(
    registry: Mapping[tuple[int, int], M0AnchorReference],
) -> tuple[v3.CanonicalSamplesV3, dict[str, Any]]:
    root = next(iter(registry.values())).result_root
    plan = _read_json(root / "PLAN.json")
    dataset_root = Path(str(plan.get("dataset_root", "")))
    samples, manifest = v3.load_canonical_dataset_v2(dataset_root)
    receipt = {"dataset_sha256": str(manifest.get("dataset", {}).get("sha256", ""))}
    receipt.update(v3.dataset_identity_receipt(dataset_root, manifest))
    if {key: receipt[key] for key in EXPECTED_ANCHOR_DATASET_RECEIPT} != EXPECTED_ANCHOR_DATASET_RECEIPT:
        raise FixedM0AnchorTrainingError("clean46 dataset現物が事前登録receiptと一致しません")
    return samples, manifest


def _copy_checkpoint(reference: M0AnchorReference, output_root: Path) -> tuple[str, str]:
    name, destination = reference.path.name, output_root / reference.path.name
    if base.file_sha256(reference.path) != reference.sha256:
        raise FixedM0AnchorTrainingError("copy前にM0 checkpointが変更されました")
    try:
        with reference.path.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
    except OSError as error:
        raise FixedM0AnchorTrainingError(f"M0 checkpointを排他的copyできません: {destination}") from error
    if base.file_sha256(reference.path) != reference.sha256:
        raise FixedM0AnchorTrainingError("copy中にM0 checkpointが変更されました")
    if base.file_sha256(destination) != reference.sha256:
        raise FixedM0AnchorTrainingError("M0 checkpoint copyがbit-identicalではありません")
    return name, reference.sha256


def _load_anchor_predictions(
    reference: M0AnchorReference, cache: dict[Path, Mapping[str, np.ndarray]],
) -> Mapping[str, np.ndarray]:
    cached = cache.get(reference.predictions_path)
    if cached is not None:
        return cached
    if base.file_sha256(reference.predictions_path) != reference.predictions_sha256:
        raise FixedM0AnchorTrainingError("anchor OOF predictions SHAが一致しません")
    try:
        with np.load(reference.predictions_path, allow_pickle=False) as data:
            loaded = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as error:
        raise FixedM0AnchorTrainingError("anchor OOF predictionsを読めません") from error
    cache[reference.predictions_path] = loaded
    return loaded


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _anchor_prediction_pair(
    reference: M0AnchorReference, samples: v3.CanonicalSamplesV3,
    cache: dict[Path, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    arrays = _load_anchor_predictions(reference, cache)
    if not np.array_equal(arrays.get("labels"), samples.labels):
        raise FixedM0AnchorTrainingError("anchor OOF labelsがclean46 datasetと不一致です")
    if not np.array_equal(arrays.get("folds"), samples.folds):
        raise FixedM0AnchorTrainingError("anchor OOF foldsがclean46 datasetと不一致です")
    prefix = f"m0__seed_{reference.seed}"
    raw = arrays.get(f"{prefix}__raw")
    calibrated = arrays.get(f"{prefix}__calibrated")
    if raw is None or calibrated is None or raw.shape != samples.labels.shape:
        raise FixedM0AnchorTrainingError("anchor M0 OOF prediction shapeが不正です")
    if calibrated.shape != raw.shape or not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise FixedM0AnchorTrainingError("anchor M0 OOF prediction coverageが不正です")
    return raw, calibrated


def _oof_validation_receipt(
    reference: M0AnchorReference, raw: np.ndarray, calibrated: np.ndarray,
    stored_raw: np.ndarray, stored_calibrated: np.ndarray,
) -> dict[str, Any]:
    return {
        "seed": reference.seed, "eval_fold": reference.fold,
        "row_count": len(raw), "raw_bit_exact": True, "calibrated_bit_exact": True,
        "raw_sha256": _array_sha256(raw),
        "stored_raw_sha256": _array_sha256(stored_raw),
        "calibrated_sha256": _array_sha256(calibrated),
        "stored_calibrated_sha256": _array_sha256(stored_calibrated),
        "checkpoint_sha256": reference.sha256,
        "state_dict_sha256": reference.state_sha256, "slope": reference.slope,
        "oof_predictions_path": str(reference.predictions_path.resolve()),
        "oof_predictions_sha256": reference.predictions_sha256,
    }


def validate_clean46_anchor_oof(
    registry: Mapping[tuple[int, int], M0AnchorReference],
    samples: v3.CanonicalSamplesV3, args: argparse.Namespace, device: torch.device,
) -> list[dict[str, Any]]:
    """全18 M0をclean46 eval foldへ再推論し、保存OOFとbit-exact照合する。"""

    cache: dict[Path, Mapping[str, np.ndarray]] = {}
    receipts: list[dict[str, Any]] = []
    for key in sorted(registry):
        reference = registry[key]
        anchor = load_m0_anchor(reference, device)
        mask = samples.folds == reference.fold
        evaluation = samples.subset(mask)
        raw = frozen._predict_m0(anchor.model, evaluation, args, device)
        stored_raw, stored_calibrated = _anchor_prediction_pair(reference, samples, cache)
        expected_raw = stored_raw[mask]
        calibrated = legacy.calibrate(raw, reference.slope).astype(
            stored_calibrated.dtype, copy=False,
        )
        expected_calibrated = stored_calibrated[mask]
        if not np.array_equal(raw, expected_raw):
            raise FixedM0AnchorTrainingError(f"clean46 M0 raw OOF不一致: {key}")
        if not np.array_equal(calibrated, expected_calibrated):
            raise FixedM0AnchorTrainingError(f"clean46 M0 calibrated OOF不一致: {key}")
        receipts.append(_oof_validation_receipt(
            reference, raw, calibrated, expected_raw, expected_calibrated,
        ))
    if len(receipts) != len(PUBLIC_SEEDS) * 6:
        raise FixedM0AnchorTrainingError("clean46 M0 OOF照合receiptが18件ありません")
    return receipts


def _champion_prediction_pair(
    reference: M1ChampionReference, samples: v3.CanonicalSamplesV3,
) -> tuple[np.ndarray, np.ndarray]:
    if base.file_sha256(reference.predictions_path) != reference.predictions_sha256:
        raise FixedM0AnchorTrainingError("clean46 champion OOF SHAが不一致です")
    try:
        with np.load(reference.predictions_path, allow_pickle=False) as data:
            if not np.array_equal(data["labels"], samples.labels):
                raise FixedM0AnchorTrainingError("clean46 champion OOF labelsが不一致です")
            if not np.array_equal(data["folds"], samples.folds):
                raise FixedM0AnchorTrainingError("clean46 champion OOF foldsが不一致です")
            raw = np.asarray(data["m1_zero_values_and_masks__seed_20260904__raw"])
            calibrated = np.asarray(
                data["m1_zero_values_and_masks__seed_20260904__calibrated"]
            )
    except (OSError, ValueError, KeyError) as error:
        raise FixedM0AnchorTrainingError("clean46 champion OOFを読めません") from error
    if raw.shape != samples.labels.shape or calibrated.shape != raw.shape:
        raise FixedM0AnchorTrainingError("clean46 champion OOF shapeが不正です")
    return raw, calibrated


def _validate_champion_selection(
    selection: v3.ResidualSelectionV3, reference: M1ChampionReference,
    m0: M0AnchorReference, evaluation: v3.PreparedSplitV3,
) -> None:
    record = reference.record
    expected = (
        selection.best_epoch == int(record.get("best_epoch", -1)),
        selection.fallback_to_m0 == record.get("fallback_to_m0"),
        selection.m0_state_sha256 == m0.state_sha256,
        frozen.model_state_sha256(selection.model) == reference.model_state_sha256,
        dict(selection.tune_raw) == record.get("tune_raw"),
        dict(selection.tune_calibrated) == record.get("tune_calibrated"),
        dict(selection.train_receipt) == record.get("train_random_receipt"),
        dict(selection.tune_receipt) == record.get("tune_random_receipt"),
        dict(evaluation.receipt) == record.get("eval_random_receipt"),
    )
    if not all(expected):
        raise FixedM0AnchorTrainingError("clean46 representative residual selectionが不一致です")


def validate_clean46_representative_residual(
    registry: Mapping[tuple[int, int], M0AnchorReference],
    reference: M1ChampionReference, samples: v3.CanonicalSamplesV3,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, Any]:
    """固定seed/fold/variantを再学習し、既存clean46 championと厳密照合する。"""

    seed, fold, variant = 20260904, 1, "values_and_masks"
    plan = {item.eval_fold: item for item in fixed_fold_plan(samples.folds)}[fold]
    m0_reference = registry[(seed, fold)]
    anchor = load_m0_anchor(m0_reference, device)
    fold_seed = seed + fold * 100
    selection = v3._fit_residual(
        samples, plan, variant, fold_seed, anchor, args, device,
    )
    evaluation = v3._prepare_split(
        samples.subset(samples.folds == fold), variant, fold_seed + 13,
    )
    _validate_champion_selection(selection, reference, m0_reference, evaluation)
    baseline = frozen._predict_m0(anchor.model, evaluation.samples, args, device)
    raw = baseline if selection.fallback_to_m0 else v3._predict(
        selection.model, evaluation, args, device,
    )
    stored_raw, stored_calibrated = _champion_prediction_pair(reference, samples)
    mask = samples.folds == fold
    raw = raw.astype(stored_raw.dtype, copy=False)
    calibrated = legacy.calibrate(raw, anchor.slope).astype(
        stored_calibrated.dtype, copy=False,
    )
    if not np.array_equal(raw, stored_raw[mask]):
        raise FixedM0AnchorTrainingError("clean46 representative residual raw OOF不一致です")
    if not np.array_equal(calibrated, stored_calibrated[mask]):
        raise FixedM0AnchorTrainingError("clean46 representative residual calibrated OOF不一致です")
    return {
        "seed": seed, "eval_fold": fold, "variant": variant,
        "row_count": len(raw), "raw_bit_exact": True, "calibrated_bit_exact": True,
        "best_epoch": selection.best_epoch, "fallback_to_m0": selection.fallback_to_m0,
        "model_state_sha256": frozen.model_state_sha256(selection.model),
        "checkpoint_path": str(reference.path.resolve()),
        "checkpoint_sha256": reference.sha256,
        "raw_sha256": _array_sha256(raw),
        "calibrated_sha256": _array_sha256(calibrated),
        "m0_state_sha256": selection.m0_state_sha256, "slope": anchor.slope,
    }


def _assert_common_prediction_exact(
    raw: np.ndarray, calibrated: np.ndarray, evaluation: v3.CanonicalSamplesV3,
    relation: DatasetRelation, reference: M0AnchorReference,
    cache: dict[Path, Mapping[str, np.ndarray]],
) -> None:
    common_positions, anchor_positions = [], []
    for position, state_id in enumerate(evaluation.state_ids):
        anchor_position = relation.anchor_index.get(str(state_id))
        if anchor_position is not None:
            common_positions.append(position)
            anchor_positions.append(anchor_position)
    expected_raw, expected_calibrated = _anchor_prediction_pair(
        reference, relation.anchor_samples, cache,
    )
    if not common_positions:
        raise FixedM0AnchorTrainingError("anchor M0 OOF prediction coverageが不正です")
    if not np.array_equal(raw[common_positions], expected_raw[anchor_positions]):
        raise FixedM0AnchorTrainingError("clean46共通M0 raw予測がbit-exactではありません")
    cast = calibrated.astype(expected_calibrated.dtype, copy=False)
    if not np.array_equal(cast[common_positions], expected_calibrated[anchor_positions]):
        raise FixedM0AnchorTrainingError("clean46共通M0 calibrated予測がbit-exactではありません")


def _baseline_record(
    anchor: frozen.M0Anchor, reference: M0AnchorReference,
    evaluation: v3.CanonicalSamplesV3, raw: np.ndarray, plan: FoldPlan,
    model_name: str, model_sha256: str,
) -> dict[str, Any]:
    return {
        "eval_fold": plan.eval_fold, "tune_fold": plan.tune_fold,
        "train_folds": list(plan.train_folds), "best_epoch": anchor.best_epoch,
        "symmetric_platt_slope": anchor.slope, "tune_raw": dict(anchor.tune_raw),
        "tune_calibrated": dict(anchor.tune_calibrated),
        "eval_raw": legacy.metrics(evaluation, raw),
        "eval_calibrated": legacy.metrics(evaluation, legacy.calibrate(raw, anchor.slope)),
        "model": model_name, "model_sha256": model_sha256,
        "m0_state_sha256": anchor.state_sha256,
        "external_anchor_root": str(reference.result_root.resolve()),
        "external_anchor_checkpoint": str(reference.path),
    }


def _predict_external_m0(
    anchor: frozen.M0Anchor, evaluation: v3.CanonicalSamplesV3,
    relation: DatasetRelation, args: argparse.Namespace, device: torch.device,
) -> np.ndarray:
    """clean46共通行は元OOFと同じbatch境界、新規行は別batchで推論する。"""

    common = np.asarray([
        str(state_id) in relation.anchor_index for state_id in evaluation.state_ids
    ])
    if not common.any():
        raise FixedM0AnchorTrainingError("eval foldにclean46共通stateがありません")
    common_raw = frozen._predict_m0(
        anchor.model, evaluation.subset(common), args, device,
    )
    raw = np.empty(len(evaluation.labels), dtype=common_raw.dtype)
    raw[common] = common_raw
    if (~common).any():
        raw[~common] = frozen._predict_m0(
            anchor.model, evaluation.subset(~common), args, device,
        )
    return raw


def _run_fold(
    samples: v3.CanonicalSamplesV3, plan: FoldPlan, seed: int,
    registry: Mapping[tuple[int, int], M0AnchorReference], relation: DatasetRelation,
    cache: dict[Path, Mapping[str, np.ndarray]], args: argparse.Namespace,
    device: torch.device,
) -> dict[str, v3.FoldOutputV3]:
    reference = registry[(seed, plan.eval_fold)]
    anchor = load_m0_anchor(reference, device)
    evaluation = samples.subset(samples.folds == plan.eval_fold)
    raw = _predict_external_m0(anchor, evaluation, relation, args, device)
    calibrated = legacy.calibrate(raw, anchor.slope)
    _assert_common_prediction_exact(raw, calibrated, evaluation, relation, reference, cache)
    m0_name, m0_hash = _copy_checkpoint(reference, args.output_root)
    baseline_record = _baseline_record(
        anchor, reference, evaluation, raw, plan, m0_name, m0_hash,
    )
    output = {"m0": v3.FoldOutputV3(raw, calibrated, baseline_record)}
    fold_seed = seed + plan.eval_fold * 100
    for variant in args.variants:
        selection = v3._fit_residual(samples, plan, variant, fold_seed, anchor, args, device)
        prepared = v3._prepare_split(evaluation, variant, fold_seed + 13)
        output[v3._result_variant(variant)] = v3._residual_output(
            selection, prepared, anchor, seed, plan, args, device, m0_name, m0_hash,
        )
    return output


def _run_seed(
    samples: v3.CanonicalSamplesV3, plans: Mapping[int, FoldPlan], seed: int,
    registry: Mapping[tuple[int, int], M0AnchorReference], relation: DatasetRelation,
    cache: dict[Path, Mapping[str, np.ndarray]], args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    accumulators = v3._new_accumulators(len(samples.labels), args.variants)
    for fold in args.folds:
        outputs = _run_fold(
            samples, plans[fold], seed, registry, relation, cache, args, device,
        )
        indices = np.flatnonzero(samples.folds == fold)
        for name, output in outputs.items():
            v3._assign(accumulators[name], indices, output)
    expected = np.isin(samples.folds, args.folds)
    return {
        f"{name}__seed_{seed}": v3._finalize(samples, accumulator, expected)
        for name, accumulator in accumulators.items()
    }


def _code_hashes() -> dict[str, str]:
    hashes = dict(v3._code_hashes())
    hashes["fixed_m0_anchor_trainer"] = base.file_sha256(Path(__file__))
    return hashes


def _environment_fingerprint(device: torch.device) -> dict[str, Any]:
    """再現に必要な限定的な実行環境と決定性設定を返す。"""

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


def _anchor_checkpoint_receipts(
    registry: Mapping[tuple[int, int], M0AnchorReference],
) -> list[dict[str, Any]]:
    return [{
        "seed": reference.seed, "eval_fold": reference.fold,
        "checkpoint_path": str(reference.path.resolve()),
        "checkpoint_file_sha256": reference.sha256,
        "state_dict_sha256": reference.state_sha256,
        "symmetric_platt_slope": reference.slope,
        "results_root": str(reference.result_root.resolve()),
        "results_sha256": reference.results_sha256,
        "oof_predictions_path": str(reference.predictions_path.resolve()),
        "oof_predictions_sha256": reference.predictions_sha256,
    } for _, reference in sorted(registry.items())]


def _anchor_dataset_root(
    registry: Mapping[tuple[int, int], M0AnchorReference],
) -> Path:
    root = next(iter(registry.values())).result_root
    return Path(str(_read_json(root / "PLAN.json").get("dataset_root", "")))


def _input_asset_paths(
    args: argparse.Namespace, registry: Mapping[tuple[int, int], M0AnchorReference],
    champion: M1ChampionReference,
) -> tuple[Path, ...]:
    roots = (args.dataset_root, _anchor_dataset_root(registry))
    paths = {
        *(root / name for root in roots for name in ("manifest.json", "dataset.npz", "COMPLETE")),
        *(reference.path for reference in registry.values()),
        champion.path,
        *(reference.result_root / name for reference in registry.values()
          for name in ("PLAN.json", "results.json", "oof_predictions.npz", "COMPLETE")),
        Path(__file__), Path(v3.__file__), Path(frozen.__file__), Path(legacy.__file__),
        Path(base.__file__), Path(merger.__file__), Path(v3.builder.__file__),
        REPO_ROOT / "src/advantage_m1_zero_counterfactual_v3.py",
        REPO_ROOT / "src/advantage_m1_causal_ledger_v3.py",
        REPO_ROOT / "src/advantage_m0_current_cnn_v1.py",
        REPO_ROOT / "src/production_config.py",
    }
    return tuple(sorted((path.resolve() for path in paths), key=str))


def _capture_input_assets(paths: Sequence[Path]) -> list[dict[str, str]]:
    receipts = []
    for path in paths:
        if not path.is_file():
            raise FixedM0AnchorTrainingError(f"入力資産がありません: {path}")
        receipts.append({"path": str(path), "sha256": base.file_sha256(path)})
    return receipts


def _assert_input_assets_unchanged(receipts: Sequence[Mapping[str, str]]) -> None:
    for receipt in receipts:
        path = Path(receipt["path"])
        if not path.is_file() or base.file_sha256(path) != receipt["sha256"]:
            raise FixedM0AnchorTrainingError(f"実行中に入力資産が変化しました: {path}")


def _validate_evidence_coverage(
    checkpoints: Sequence[Mapping[str, Any]], oof: Sequence[Mapping[str, Any]],
) -> None:
    expected = {(seed, fold) for seed in PUBLIC_SEEDS for fold in range(1, 7)}
    checkpoint_keys = {(int(row["seed"]), int(row["eval_fold"])) for row in checkpoints}
    oof_keys = {(int(row["seed"]), int(row["eval_fold"])) for row in oof}
    if checkpoint_keys != expected or len(checkpoints) != len(expected):
        raise FixedM0AnchorTrainingError("PLAN用M0 checkpoint receiptが18件一意ではありません")
    if oof_keys != expected or len(oof) != len(expected):
        raise FixedM0AnchorTrainingError("PLAN用clean46 OOF receiptが18件一意ではありません")


def _build_plan(
    args: argparse.Namespace, samples: v3.CanonicalSamplesV3,
    manifest: Mapping[str, Any], anchor_receipts: Sequence[Mapping[str, Any]],
    relation: DatasetRelation, registry: Mapping[tuple[int, int], M0AnchorReference],
    oof_receipts: Sequence[Mapping[str, Any]],
    representative_receipt: Mapping[str, Any], weight_receipt: Mapping[str, Any],
    input_assets: Sequence[Mapping[str, str]],
    device: torch.device | None = None,
) -> dict[str, Any]:
    checkpoint_receipts = _anchor_checkpoint_receipts(registry)
    _validate_evidence_coverage(checkpoint_receipts, oof_receipts)
    plan = v3._plan(args, samples, manifest)
    plan.update({
        "m0_anchor_policy": "external_clean46_fixed",
        "external_m0_anchor_policy": "clean46_seed_by_eval_fold_fixed_no_refit",
        "external_m0_anchor_lookup": "public_seed_and_eval_fold_not_fold_seed",
        "external_m0_anchor_coverage": "3_seed_x_6_fold_exact",
        "external_m0_anchor_receipts": list(anchor_receipts),
        "m0_anchor_checkpoint_receipts": checkpoint_receipts,
        "clean46_m0_oof_validation_receipts": list(oof_receipts),
        "clean46_representative_residual_validation_receipt": dict(
            representative_receipt
        ),
        "external_m0_anchor_dataset_receipt": dict(EXPECTED_ANCHOR_DATASET_RECEIPT),
        "increment_dataset_relation": dict(relation.receipt),
        "increment_weight_stabilization_receipt": dict(weight_receipt),
        "m0_checkpoint_copy_policy": "exclusive_byte_copy_sha_identical",
        "calibration_policy": "clean46 external M0 symmetric Platt slope fixed",
        "input_asset_receipts": list(input_assets),
        "code_sha256": _code_hashes(),
        "environment_fingerprint": _environment_fingerprint(
            device if device is not None else torch.device("cpu")
        ),
    })
    if plan["code_sha256"]["production_config"] != EXPECTED_PRODUCTION_CONFIG_SHA256:
        raise FixedM0AnchorTrainingError("production_config SHAが事前登録時から変化しました")
    return plan


def _prepare_inputs(args: argparse.Namespace) -> ValidatedInputs:
    samples, manifest = v3.load_canonical_dataset_v2(args.dataset_root)
    _validate_target_receipt(args.dataset_root, manifest)
    registry, anchor_receipts = load_anchor_registry(args.anchor_roots)
    champion = load_champion_reference(registry)
    anchor_samples, anchor_manifest = _load_anchor_dataset(registry)
    relation = validate_dataset_relation(anchor_samples, anchor_manifest, samples, manifest)
    samples, weight_receipt = stabilize_increment_weights(samples, relation)
    assets = _capture_input_assets(_input_asset_paths(args, registry, champion))
    return ValidatedInputs(
        samples, manifest, registry, champion, anchor_receipts,
        relation, weight_receipt, assets,
    )


def _serialize_results(
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    predictions: dict[str, np.ndarray] = {}
    serializable: dict[str, Any] = {}
    for key, result in results.items():
        predictions[f"{key}__raw"] = result["raw"]
        predictions[f"{key}__calibrated"] = result["calibrated"]
        serializable[key] = {
            name: value for name, value in result.items() if name not in {"raw", "calibrated"}
        }
    return predictions, serializable


def _write_results(
    root: Path, inputs: ValidatedInputs, plan: Mapping[str, Any], plan_path: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    predictions, serializable = _serialize_results(results)
    predictions.update({"labels": inputs.samples.labels, "folds": inputs.samples.folds})
    prediction_path = root / "oof_predictions.npz"
    legacy._save_npz_exclusive(prediction_path, predictions)
    report = {
        "format_version": v3.TRAINING_VERSION, "not_production": True,
        "dataset_source_count": inputs.manifest["source_count"],
        "plan": dict(plan), "results": serializable,
    }
    result_path = base._write_json_exclusive(root / "results.json", report)
    _assert_input_assets_unchanged(inputs.input_assets)
    base._write_json_exclusive(root / "COMPLETE", {
        "format_version": v3.COMPLETE_VERSION,
        "plan_sha256": base.file_sha256(plan_path),
        "results_sha256": base.file_sha256(result_path),
        "predictions_sha256": base.file_sha256(prediction_path),
    })
    return report


def check_only(args: argparse.Namespace) -> dict[str, Any]:
    """全固定入力と18 anchor OOFを検査し、ファイルを書かずreceiptを返す。"""

    inputs = _prepare_inputs(args)
    device = base._device(args.device)
    oof_receipts = validate_clean46_anchor_oof(
        inputs.registry, inputs.relation.anchor_samples, args, device,
    )
    representative = validate_clean46_representative_residual(
        inputs.registry, inputs.champion, inputs.relation.anchor_samples, args, device,
    )
    _assert_input_assets_unchanged(inputs.input_assets)
    return {
        "check_only": True, "m0_anchor_policy": "external_clean46_fixed",
        "anchor_count": len(inputs.registry),
        "clean46_m0_oof_validation_receipts": oof_receipts,
        "clean46_representative_residual_validation_receipt": representative,
        "increment_dataset_relation": dict(inputs.relation.receipt),
        "increment_weight_stabilization_receipt": dict(inputs.weight_receipt),
        "input_asset_receipts": list(inputs.input_assets),
        "code_sha256": _code_hashes(),
        "environment_fingerprint": _environment_fingerprint(device),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    """事前登録済みclean46 M0固定・clean48 residual OOFを新規rootへ保存する。"""

    if args.output_root is None or args.output_root.exists():
        raise FixedM0AnchorTrainingError(f"出力先は新規必須です: {args.output_root}")
    inputs = _prepare_inputs(args)
    device = base._device(args.device)
    oof_receipts = validate_clean46_anchor_oof(
        inputs.registry, inputs.relation.anchor_samples, args, device,
    )
    representative = validate_clean46_representative_residual(
        inputs.registry, inputs.champion, inputs.relation.anchor_samples, args, device,
    )
    plans = {plan.eval_fold: plan for plan in fixed_fold_plan(inputs.samples.folds)}
    plan = _build_plan(
        args, inputs.samples, inputs.manifest, inputs.anchor_receipts,
        inputs.relation, inputs.registry, oof_receipts,
        representative, inputs.weight_receipt, inputs.input_assets,
        device,
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    plan_path = base._write_json_exclusive(args.output_root / "PLAN.json", plan)
    cache: dict[Path, Mapping[str, np.ndarray]] = {}
    results: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        results.update(_run_seed(
            inputs.samples, plans, seed, inputs.registry,
            inputs.relation, cache, args, device,
        ))
    return _write_results(args.output_root, inputs, plan, plan_path, results)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--anchor-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--variants", type=v3._csv_strings, default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", type=v3._csv_ints, default=PUBLIC_SEEDS)
    parser.add_argument("--folds", type=v3._csv_ints, default=tuple(range(1, 7)))
    parser.add_argument("--epochs", type=int, default=legacy.DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=legacy.DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=legacy.DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=legacy.DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=legacy.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-output", type=Path)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    v3._validate_args(parser, args)
    fixed = (
        tuple(args.variants) == DEFAULT_VARIANTS,
        tuple(args.seeds) == PUBLIC_SEEDS,
        tuple(args.folds) == tuple(range(1, 7)),
        args.epochs == legacy.DEFAULT_EPOCHS,
        args.patience == legacy.DEFAULT_PATIENCE,
        args.batch_size == legacy.DEFAULT_BATCH_SIZE,
        args.learning_rate == legacy.DEFAULT_LEARNING_RATE,
        args.weight_decay == legacy.DEFAULT_WEIGHT_DECAY,
    )
    if not all(fixed):
        parser.error("variants/seeds/folds/学習hyperparameterは事前登録値から変更できません")
    if not args.check_only and args.output_root is None:
        parser.error("学習時は--output-rootが必須です")
    if args.check_output is not None and not args.check_only:
        parser.error("--check-outputは--check-only専用です")
    if args.check_output is not None and args.check_output.exists():
        parser.error("--check-outputは新規path必須です")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_only:
        receipt = check_only(args)
        if args.check_output is not None:
            base._write_json_exclusive(args.check_output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
