"""old308 anchorと4種ledger残差の3seed OOFを診断監査する。"""

from __future__ import annotations

from scripts.production_dependency_contract import (
    dependency_receipt, production_compatible, saved_dependency_compatible,
)

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import analyze_advantage_m1_fixed_oof_v1 as strata_analysis
from scripts import build_advantage_m2_dataset_v1 as base
from scripts import train_advantage_m2_auxiliary_v1 as m2_trainer
from scripts import train_advantage_m2_old308_anchor_ledger_v1 as trainer
from src.advantage_m1_causal_ledger_v3 import AVAILABILITY_ORDER
from src.canonical_observation_v3 import AvailabilityState
from src.event_provisional_oof_v1 import (
    apply_platt_calibration,
    symmetric_predict_probability,
    weighted_probability_metrics,
)


FORMAT_VERSION = "advantage-m2-old308-anchor-ledger-analysis/v1"
COMPLETE_VERSION = "advantage-m2-old308-anchor-ledger-analysis-complete/v1"
DEFAULT_OUTPUT_ROOT = Path(
    "data/verify/advantage_m2_old308_anchor_ledger_audit_46v_2026-09-05_v1"
)
BOOTSTRAP_REPEATS = 10_000
THRESHOLDS = (0.10, 0.20)
EXAMPLE_LIMIT = 20
PRODUCTION_CONFIG_SHA256 = "3fe3c2578b3196a60cffffcafa594c1f64d2a913bb0487932ac5cd8f6f86d376"
BASELINE = "old308_anchor"
VARIANTS = tuple(f"m2_old308_{name}" for name in trainer.LEDGER_VARIANTS)
PREDICTION_KINDS = ("raw", "calibrated")
_FAULT_INDEX = AVAILABILITY_ORDER.index(AvailabilityState.INTEGRITY_FAULT)
AGGREGATION_SUFFIXES = (
    "_p1_missing", "_p2_missing", "_diff", "_max", "_p1", "_p2", "_sum",
)
FEATURE_FAMILY_BASES = {
    "shortcut_score_progress": ("a_score_log", "a_tsumo_log"),
    "shortcut_recognition_quality": ("a_stable_confidence", "a_unknown_ratio"),
    "shortcut_mechanism": (
        "a_mechanism_baseline", "a_mechanism_landing", "a_mechanism_other",
    ),
    "all_clear": ("a_all_clear_pending",),
    "queue_color_fit": (
        "a_color_ratio", "a_next_first_match_ratio", "a_next_pair_same",
        "a_next_second_match_ratio", "a_double_next_first_match_ratio",
        "a_double_next_pair_same", "a_double_next_second_match_ratio",
    ),
    "material_and_height": (
        "a_garbage_ratio", "a_indicator_board_color_puyo_total",
        "a_indicator_board_ojama_count", "a_indicator_board_puyo_total",
        "a_indicator_max_column_height", "a_occupied_ratio",
    ),
    "resilience_and_room": (
        "a_indicator_buried_hole_count", "a_indicator_center_bulge_color",
        "a_indicator_center_bulge_ojama", "a_indicator_death_margin",
        "a_indicator_death_margin_neighbor", "a_indicator_dig_resistance",
        "a_indicator_ukeyasusa",
    ),
    "connectivity_and_shape": (
        "a_indicator_chain_articulation_point_count", "a_indicator_chain_efficiency",
        "a_indicator_color_diversity_evenness", "a_indicator_column_bumpiness",
        "a_indicator_isolated_pair_count", "a_indicator_main_linked_pair_count",
        "a_indicator_main_linked_ratio", "a_indicator_multi_color_ignition",
        "a_indicator_simultaneous_pop_richness", "a_indicator_sub_chain_count",
    ),
    "firepower_and_ignition": (
        "a_indicator_current_max_chain", "a_indicator_ignition_point_count",
        "a_indicator_immediate_fire_power", "a_indicator_min_puyos_to_ignite",
        "a_indicator_saturation_chain_upper", "a_indicator_second_chain_potential",
    ),
}


class Old308LedgerAnalysisError(RuntimeError):
    """old308 ledger診断成果物の契約違反。"""


@dataclass(frozen=True, slots=True)
class TrainingArtifact:
    """receipt検証済み学習成果物。"""

    plan: Mapping[str, Any]
    results: Mapping[str, Any]
    predictions: Mapping[str, np.ndarray]
    receipt: Mapping[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Old308LedgerAnalysisError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise Old308LedgerAnalysisError(f"JSON objectではありません: {path}")
    return value


def _load_artifact(
    root: Path, samples: trainer.Samples, dataset_manifest: Mapping[str, Any],
) -> TrainingArtifact:
    paths = {
        "plan_sha256": root / "PLAN.json",
        "results_sha256": root / "RESULTS.json",
        "predictions_sha256": root / "oof_predictions.npz",
    }
    complete = _load_json(root / "COMPLETE")
    if complete.get("format_version") != trainer.FORMAT_VERSION:
        raise Old308LedgerAnalysisError("training COMPLETE versionが不正です")
    if complete.get("not_production") is not True:
        raise Old308LedgerAnalysisError("training COMPLETEはnot_production必須です")
    for name, path in paths.items():
        if complete.get(name) != base.file_sha256(path):
            raise Old308LedgerAnalysisError(f"training receiptが不一致です: {name}")
    plan, results = _load_json(paths["plan_sha256"]), _load_json(paths["results_sha256"])
    _validate_plan(plan, results, dataset_manifest)
    predictions = _load_predictions(paths["predictions_sha256"], samples, plan)
    receipt = {name: complete[name] for name in paths}
    return TrainingArtifact(plan, results, predictions, receipt)


def _validate_plan(
    plan: Mapping[str, Any], results: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
) -> None:
    if results.get("plan") != plan or results.get("diagnostic_only") is not True:
        raise Old308LedgerAnalysisError("training plan/result契約が不正です")
    if tuple(plan.get("seeds", ())) != trainer.DEFAULT_SEEDS:
        raise Old308LedgerAnalysisError("固定3seedではありません")
    if tuple(plan.get("variants", ())) != trainer.LEDGER_VARIANTS:
        raise Old308LedgerAnalysisError("固定4variantではありません")
    if plan.get("format_version") != trainer.PLAN_VERSION:
        raise Old308LedgerAnalysisError("training PLAN versionが不正です")
    if results.get("format_version") != trainer.FORMAT_VERSION:
        raise Old308LedgerAnalysisError("training RESULTS versionが不正です")
    if plan.get("dataset_sha256") != dataset_manifest["dataset"]["sha256"]:
        raise Old308LedgerAnalysisError("training dataset hashが一致しません")
    code = plan.get("code_sha256", {})
    if {k: v for k, v in code.items() if k != "production_config"} != {
        k: v for k, v in trainer._code_hashes().items() if k != "production_config"
    }:
        raise Old308LedgerAnalysisError("training後に実験コードが変化しています")
    if not saved_dependency_compatible(code.get("production_config"), plan.get("production_dependency_contract")):
        raise Old308LedgerAnalysisError("training時のproduction config hashが不正です")
    if not production_compatible():
        raise Old308LedgerAnalysisError("現在のproduction依存値が変化しています")


def _load_predictions(
    path: Path, samples: trainer.Samples, plan: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    identity = {
        "labels": samples.labels, "weights": samples.weights, "folds": samples.folds,
        "state_ids": samples.state_ids, "source_groups": samples.groups,
        "game_keys": samples.games, "anchor_input_usable": samples.anchor_input_usable,
    }
    if any(not np.array_equal(arrays.get(name), value) for name, value in identity.items()):
        raise Old308LedgerAnalysisError("training予測とdataset identityが不一致です")
    expected = set(identity) | _expected_prediction_names(plan)
    if set(arrays) != expected:
        raise Old308LedgerAnalysisError("training予測array集合が不正です")
    _validate_probability_arrays(arrays, samples.anchor_input_usable, expected - set(identity))
    return arrays


def _expected_prediction_names(plan: Mapping[str, Any]) -> set[str]:
    variants = (BASELINE, *VARIANTS)
    return {
        f"{variant}__seed_{seed}__{kind}"
        for variant in variants for seed in plan["seeds"] for kind in PREDICTION_KINDS
    }


def _validate_probability_arrays(
    arrays: Mapping[str, np.ndarray], usable: np.ndarray, names: set[str],
) -> None:
    for name in names:
        value = np.asarray(arrays[name])
        finite = np.isfinite(value)
        if value.shape != usable.shape or not np.array_equal(finite, usable):
            raise Old308LedgerAnalysisError(f"予測finite maskが不正です: {name}")
        if np.any(value[finite] < 0.0) or np.any(value[finite] > 1.0):
            raise Old308LedgerAnalysisError(f"予測確率が不正です: {name}")


def _ensemble(predictions: Mapping[str, np.ndarray], variant: str) -> np.ndarray:
    rows = [
        predictions[f"{variant}__seed_{seed}__calibrated"]
        for seed in trainer.DEFAULT_SEEDS
    ]
    return np.mean(np.stack(rows).astype(np.float64), axis=0)


def _metric_rows(
    samples: trainer.Samples, probability: np.ndarray, mask: np.ndarray,
) -> dict[str, float]:
    return weighted_probability_metrics(
        samples.labels[mask], probability[mask], samples.weights[mask],
    )


def _group_rows(
    samples: trainer.Samples, baseline: np.ndarray, candidate: np.ndarray,
    mask: np.ndarray, groups: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for group in np.unique(groups[mask]):
        selected = mask & (groups == group)
        old = _metric_rows(samples, baseline, selected)["log_loss"]
        new = _metric_rows(samples, candidate, selected)["log_loss"]
        rows.append({"group": str(group), "baseline_log_loss": old,
                     "candidate_log_loss": new, "delta_log_loss": new - old})
    return rows


def _bootstrap(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, float | int]:
    values = np.asarray([row["delta_log_loss"] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "repeats": BOOTSTRAP_REPEATS, "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "probability_candidate_better": float(np.mean(means < 0.0)),
    }


def _metric_comparison(
    samples: trainer.Samples, baseline: np.ndarray, candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    old, new = _metric_rows(samples, baseline, mask), _metric_rows(samples, candidate, mask)
    delta_names = (
        "log_loss", "brier", "auc", "calibration_ece_10bin",
        "calibration_max_gap_10bin", "severe_wrong_rate",
    )
    return {
        "state_count": int(mask.sum()), "game_count": len(set(samples.games[mask].tolist())),
        "baseline": old, "candidate": new,
        "delta": {name: new[name] - old[name] for name in delta_names},
    }


def _extreme_rows(
    samples: trainer.Samples, available_ms: np.ndarray,
    candidate_winner: np.ndarray, mask: np.ndarray,
) -> list[dict[str, Any]]:
    order = np.flatnonzero(mask)[np.argsort(candidate_winner[mask])[:EXAMPLE_LIMIT]]
    return [{
        "state_id": str(samples.state_ids[index]), "source": str(samples.groups[index]),
        "game": str(samples.games[index]), "fold": int(samples.folds[index]),
        "available_ms": int(available_ms[index]),
        "candidate_winner_probability": float(candidate_winner[index]),
    } for index in order]


def _extreme_audit(
    samples: trainer.Samples, baseline: np.ndarray, candidate: np.ndarray,
    finite: np.ndarray, available_ms: np.ndarray,
) -> dict[str, Any]:
    label = samples.labels
    old = np.where(label == 1, baseline, 1.0 - baseline)
    new = np.where(label == 1, candidate, 1.0 - candidate)
    output: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        old_bad, new_bad = finite & (old < threshold), finite & (new < threshold)
        newly_bad, resolved = new_bad & ~old_bad, old_bad & ~new_bad
        output[str(threshold)] = {
            "baseline_count": int(old_bad.sum()), "candidate_count": int(new_bad.sum()),
            "new_count": int(newly_bad.sum()), "resolved_count": int(resolved.sum()),
            "common_count": int((old_bad & new_bad).sum()),
            "candidate_nonincrease": bool(new_bad.sum() <= old_bad.sum()),
            "new_worst_examples": _extreme_rows(
                samples, available_ms, new, newly_bad,
            ),
        }
    return output


def _comparison(
    samples: trainer.Samples, baseline: np.ndarray, candidate: np.ndarray,
    strata: Mapping[str, np.ndarray], available_ms: np.ndarray, seed: int,
) -> dict[str, Any]:
    finite = samples.anchor_input_usable & np.isfinite(baseline) & np.isfinite(candidate)
    source_rows = _group_rows(samples, baseline, candidate, finite, samples.groups)
    fold_rows = _group_rows(samples, baseline, candidate, finite, samples.folds)
    return {
        "overall": _metric_comparison(samples, baseline, candidate, finite),
        "strata": {name: _metric_comparison(samples, baseline, candidate, finite & value)
                   for name, value in strata.items() if bool((finite & value).any())},
        "source_rows": source_rows, "source_win_count": sum(
            row["delta_log_loss"] < 0.0 for row in source_rows
        ),
        "source_bootstrap": _bootstrap(source_rows, seed),
        "fold_rows": fold_rows,
        "fold_win_count": sum(row["delta_log_loss"] < 0.0 for row in fold_rows),
        "extreme": _extreme_audit(samples, baseline, candidate, finite, available_ms),
    }


def _load_strata(
    dataset_root: Path, samples: trainer.Samples, manifest: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    parent, _ = m2_trainer.load_m2_dataset(Path(str(manifest["m2_dataset_root"])))
    if not np.array_equal(parent.state_ids, samples.state_ids):
        raise Old308LedgerAnalysisError("M2 boardとold308 state順が一致しません")
    with np.load(dataset_root / "dataset.npz", allow_pickle=False) as data:
        available_ms = np.asarray(data["available_ms"], dtype=np.int64)
    raw = {
        "boards": parent.boards, "ledger_values": samples.ledger_values,
        "ledger_usable": samples.ledger_usable, "available_ms": available_ms,
    }
    view = SimpleNamespace(labels=samples.labels, game_keys=samples.games)
    return strata_analysis.build_strata(view, raw), available_ms


def _selection_audit(
    samples: trainer.Samples, artifact: TrainingArtifact,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for seed in trainer.DEFAULT_SEEDS:
        seed_result = artifact.results["seeds"][str(seed)]
        output[str(seed)] = {}
        for short_name, value in seed_result["variants"].items():
            variant = f"m2_old308_{short_name}"
            folds = value["folds"]
            output[str(seed)][short_name] = _selection_variant_audit(
                samples, artifact.predictions, variant, seed, folds,
            )
    return output


def _selection_variant_audit(
    samples: trainer.Samples, predictions: Mapping[str, np.ndarray], variant: str,
    seed: int, folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mismatch = epoch_contract = 0
    epochs = []
    for row in folds:
        epoch, fallback = int(row["selected_epoch"]), bool(row["fallback_to_anchor"])
        epochs.append(epoch)
        epoch_contract += int((epoch == 0) != fallback)
        if fallback:
            mask = samples.anchor_input_usable & (samples.folds == int(row["eval_fold"]))
            for kind in PREDICTION_KINDS:
                old = predictions[f"{BASELINE}__seed_{seed}__{kind}"][mask]
                new = predictions[f"{variant}__seed_{seed}__{kind}"][mask]
                mismatch += int(np.count_nonzero(old != new))
    return {
        "selected_epochs": epochs, "epoch0_count": epochs.count(0),
        "fallback_count": sum(bool(row["fallback_to_anchor"]) for row in folds),
        "epoch0_fallback_contract_mismatch_count": epoch_contract,
        "fallback_prediction_bit_mismatch_count": mismatch,
    }


def _seed_stability(
    samples: trainer.Samples, predictions: Mapping[str, np.ndarray],
    strata: Mapping[str, np.ndarray], available_ms: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for seed in trainer.DEFAULT_SEEDS:
        baseline = predictions[f"{BASELINE}__seed_{seed}__calibrated"]
        output[str(seed)] = {}
        for offset, variant in enumerate(VARIANTS):
            candidate = predictions[f"{variant}__seed_{seed}__calibrated"]
            compared = _comparison(
                samples, baseline, candidate, strata, available_ms, seed + offset,
            )
            output[str(seed)][variant] = {
                "overall": compared["overall"],
                "fold_win_count": compared["fold_win_count"],
                "source_win_count": compared["source_win_count"],
                "source_bootstrap": compared["source_bootstrap"],
                "extreme": compared["extreme"],
            }
    return output


def _unsupported_audit(
    samples: trainer.Samples, ensembles: Mapping[str, np.ndarray], baseline: np.ndarray,
) -> dict[str, Any]:
    mask = samples.anchor_input_usable & ~samples.ledger_usable
    return {
        variant: _unsupported_variant(value, baseline, mask)
        for variant, value in ensembles.items()
    }


def _unsupported_variant(
    candidate: np.ndarray, baseline: np.ndarray, mask: np.ndarray,
) -> dict[str, float | int]:
    delta = np.abs(candidate[mask] - baseline[mask])
    return {
        "row_count": int(mask.sum()),
        "max_absolute_probability_delta": float(np.max(delta)) if len(delta) else 0.0,
        "bit_mismatch_count": int(np.count_nonzero(delta)),
    }


def _gate_summary(comparison: Mapping[str, Any]) -> dict[str, bool]:
    delta = comparison["overall"]["delta"]
    extreme = comparison["extreme"]
    return {
        "log_loss_improves": delta["log_loss"] < 0.0,
        "source_bootstrap_ci_high_below_zero": comparison["source_bootstrap"]["ci95_high"] < 0.0,
        "at_least_five_of_six_folds": comparison["fold_win_count"] >= 5,
        "brier_nonworse": delta["brier"] <= 0.0,
        "no_new_winner_below_10_percent": extreme["0.1"]["new_count"] == 0,
        "winner_below_20_percent_nonincrease": extreme["0.2"]["candidate_nonincrease"],
    }


def _feature_base(name: str) -> str:
    for suffix in AGGREGATION_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    raise Old308LedgerAnalysisError(f"308列の集約suffixが不明です: {name}")


def _family_columns(names: Sequence[str]) -> dict[str, np.ndarray]:
    owner = {
        base_name: family for family, base_names in FEATURE_FAMILY_BASES.items()
        for base_name in base_names
    }
    if len(owner) != sum(map(len, FEATURE_FAMILY_BASES.values())):
        raise Old308LedgerAnalysisError("feature family定義が重複しています")
    unknown = sorted({_feature_base(name) for name in names} - set(owner))
    if unknown:
        raise Old308LedgerAnalysisError(f"未分類feature baseがあります: {unknown}")
    return {
        family: np.asarray([
            index for index, name in enumerate(names) if owner[_feature_base(name)] == family
        ], dtype=np.int64)
        for family in FEATURE_FAMILY_BASES
    }


def _within_source_permutation(groups: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(len(groups), dtype=np.int64)
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        order[positions] = positions[rng.permutation(len(positions))]
    return order


def _load_anchor_payload(
    row: Mapping[str, Any], names: Sequence[str],
) -> tuple[Any, Any]:
    path = Path(str(row["path"]))
    if base.file_sha256(path) != row["sha256"]:
        raise Old308LedgerAnalysisError(f"anchor model hashが不一致です: {path}")
    payload = pickle.loads(path.read_bytes())
    if payload.get("diagnostic_only") is not True:
        raise Old308LedgerAnalysisError("anchor modelはdiagnostic_only必須です")
    if tuple(payload.get("feature_names", ())) != tuple(names):
        raise Old308LedgerAnalysisError("anchor model feature順が不一致です")
    return payload["model"], payload["record"]


def _permuted_seed_predictions(
    samples: trainer.Samples, artifact: TrainingArtifact,
    columns: Mapping[str, np.ndarray], seed: int,
) -> dict[str, np.ndarray]:
    output = {
        family: np.full(len(samples.labels), np.nan, dtype=np.float64)
        for family in columns
    }
    rows = artifact.results["seeds"][str(seed)]["anchor_models"]
    for row in rows:
        fold = int(row["eval_fold"])
        indices = np.flatnonzero(samples.anchor_input_usable & (samples.folds == fold))
        order = _within_source_permutation(samples.groups[indices], seed + fold * 100)
        model, record = _load_anchor_payload(row, samples.feature_names)
        _predict_permuted_families(
            samples, indices, order, columns, model, record, output,
        )
    return output


def _predict_permuted_families(
    samples: trainer.Samples, indices: np.ndarray, order: np.ndarray,
    columns: Mapping[str, np.ndarray], model: Any, record: Any,
    output: dict[str, np.ndarray],
) -> None:
    original = samples.features[indices]
    for family, selected_columns in columns.items():
        matrix = original.copy()
        matrix[:, selected_columns] = original[order][:, selected_columns]
        raw = symmetric_predict_probability(model, matrix, samples.feature_names)
        output[family][indices] = apply_platt_calibration(raw, record.calibration)


def _family_ensembles(
    samples: trainer.Samples, artifact: TrainingArtifact,
    columns: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    by_seed = [
        _permuted_seed_predictions(samples, artifact, columns, seed)
        for seed in trainer.DEFAULT_SEEDS
    ]
    return {
        family: np.mean(np.stack([row[family] for row in by_seed]), axis=0)
        for family in columns
    }


def _family_summary(
    samples: trainer.Samples, baseline: np.ndarray, permuted: np.ndarray,
    strata: Mapping[str, np.ndarray], available_ms: np.ndarray, seed: int,
) -> dict[str, Any]:
    compared = _comparison(samples, baseline, permuted, strata, available_ms, seed)
    return {
        "permuted_minus_anchor": compared["overall"]["delta"],
        "anchor_better_fold_count": sum(
            row["delta_log_loss"] > 0.0 for row in compared["fold_rows"]
        ),
        "anchor_better_source_count": sum(
            row["delta_log_loss"] > 0.0 for row in compared["source_rows"]
        ),
        "source_bootstrap": compared["source_bootstrap"],
        "extreme_after_permutation": compared["extreme"],
    }


def _feature_family_audit(
    samples: trainer.Samples, artifact: TrainingArtifact, baseline: np.ndarray,
    strata: Mapping[str, np.ndarray], available_ms: np.ndarray, seed: int,
) -> dict[str, Any]:
    columns = _family_columns(samples.feature_names)
    predictions = _family_ensembles(samples, artifact, columns)
    rows = {
        family: {
            "column_count": len(columns[family]),
            "shortcut": family.startswith("shortcut_"),
            **_family_summary(
                samples, baseline, value, strata, available_ms, seed + index,
            ),
        }
        for index, (family, value) in enumerate(predictions.items())
    }
    recommended = [
        family for family, row in rows.items()
        if not row["shortcut"]
        and row["permuted_minus_anchor"]["log_loss"] > 0.0
        and row["source_bootstrap"]["ci95_low"] > 0.0
    ]
    return {
        "method": "within_source_joint_row_permutation_per_macro_family",
        "optimization_used": False, "families": rows,
        "robust_nonshortcut_auxiliary_candidates": recommended,
        "interpretation": "importance is diagnostic association, not causal adoption proof",
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    """学習成果物を検証し、同一状態の3seed ensemble診断を新規保存する。"""

    if args.output_root.exists():
        raise Old308LedgerAnalysisError(f"出力先は新規必須です: {args.output_root}")
    samples, manifest = trainer.load_dataset(args.dataset_root)
    artifact = _load_artifact(args.training_root, samples, manifest)
    strata, available_ms = _load_strata(args.dataset_root, samples, manifest)
    baseline = _ensemble(artifact.predictions, BASELINE)
    ensembles = {variant: _ensemble(artifact.predictions, variant) for variant in VARIANTS}
    comparisons = {
        variant: _comparison(
            samples, baseline, candidate, strata, available_ms, args.bootstrap_seed + index,
        ) for index, (variant, candidate) in enumerate(ensembles.items())
    }
    random = ensembles["m2_old308_random_control"]
    versus_random = {
        variant: _comparison(
            samples, random, candidate, strata, available_ms, args.bootstrap_seed + 100 + index,
        ) for index, (variant, candidate) in enumerate(ensembles.items())
        if variant != "m2_old308_random_control"
    }
    family_audit = _feature_family_audit(
        samples, artifact, baseline, strata, available_ms, args.bootstrap_seed + 1_000,
    )
    report = _report(
        args, samples, manifest, artifact, comparisons, versus_random,
        ensembles, baseline, strata, available_ms, family_audit,
    )
    return _write(args.output_root, report)


def _report(
    args: argparse.Namespace, samples: trainer.Samples, manifest: Mapping[str, Any],
    artifact: TrainingArtifact, comparisons: Mapping[str, Any],
    versus_random: Mapping[str, Any], ensembles: Mapping[str, np.ndarray],
    baseline: np.ndarray, strata: Mapping[str, np.ndarray], available_ms: np.ndarray,
    family_audit: Mapping[str, Any],
) -> dict[str, Any]:
    faults = int(np.count_nonzero(samples.ledger_availability[..., _FAULT_INDEX] > 0.5))
    return {
        "format_version": FORMAT_VERSION, "not_production": True, "diagnostic_only": True,
        "dataset": {
            "root": str(args.dataset_root.resolve()), "sha256": manifest["dataset"]["sha256"],
            "source_count": len(set(samples.groups.tolist())),
            "game_count": len(set(samples.games.tolist())), "state_count": len(samples.labels),
            "feature_count": len(samples.feature_names),
            "weight_contract": manifest["weight_contract"],
        },
        "ensemble": {
            "method": "equal_arithmetic_probability_mean", "seeds": list(trainer.DEFAULT_SEEDS),
            "seed_count": len(trainer.DEFAULT_SEEDS), "optimization_used": False,
        },
        "stratum_counts": {name: int(mask.sum()) for name, mask in strata.items()},
        "comparisons_vs_old308_anchor": dict(comparisons),
        "comparisons_vs_random_control": dict(versus_random),
        "seed_stability": _seed_stability(
            samples, artifact.predictions, strata, available_ms,
        ),
        "selection_audit": _selection_audit(samples, artifact),
        "unsupported_fallback_audit": _unsupported_audit(samples, ensembles, baseline),
        "integrity_fault_cell_count": faults,
        "gate_summary": {name: _gate_summary(value) for name, value in comparisons.items()},
        "feature_family_permutation_audit": dict(family_audit),
        "training_receipt": dict(artifact.receipt),
        "formal100_used": False, "hidden_reserve_used": False,
        "attack_difference_ten_percent_correction": False,
        "production_config_changed": False,
        "production_config_sha256": base.file_sha256(Path("src/production_config.py")),
        "analysis_code_sha256": base.file_sha256(Path(__file__)),
    }


def _write(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    path = root / "analysis.json"
    base._write_json_exclusive(path, report)
    base._write_json_exclusive(root / "COMPLETE", {
        "format_version": COMPLETE_VERSION, "analysis_sha256": base.file_sha256(path),
    })
    return dict(report)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=trainer.DEFAULT_DATASET_ROOT)
    parser.add_argument("--training-root", type=Path, default=trainer.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = analyze(parse_args(argv))
    summary = {
        name: value["overall"]["delta"]
        for name, value in report["comparisons_vs_old308_anchor"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
