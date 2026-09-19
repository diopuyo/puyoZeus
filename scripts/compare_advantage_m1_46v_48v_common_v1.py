"""M1 46動画版と48動画版を共通state_id上で直接paired監査する。"""

from __future__ import annotations

from scripts.production_dependency_contract import (
    dependency_receipt, production_compatible, saved_dependency_compatible,
)

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import analyze_advantage_m1_fixed_oof_v1 as single
from scripts import analyze_advantage_m1_multiseed_ensemble_v1 as ensemble
from scripts import train_advantage_m0_current_cnn_v1 as m0


FORMAT_VERSION = "advantage-m1-common-dataset-comparison/v1"
COMPLETE_VERSION = "advantage-m1-common-dataset-comparison-complete/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_VARIANT = "m1_zero_values_and_masks"
RANDOM_VARIANT = "m1_zero_random_control"
EXPECTED_PRODUCTION_CONFIG_SHA256 = (
    "3fe3c2578b3196a60cffffcafa594c1f64d2a913bb0487932ac5cd8f6f86d376"
)
EXACT_FIELDS = (
    "boards", "queues", "ledger_values", "ledger_availability", "labels",
    "source_groups", "game_keys", "folds", "state_ids", "available_ms",
    "online_segment_index", "ledger_usable", "projected_input_digest",
    "projected_gate_status",
)
WEIGHT_RATIO_TOLERANCE = 1e-6
MAX_FOLD_REGRESSION = 0.005
MIN_BOOTSTRAP_PROBABILITY = 0.80
MIN_FULL_BOOTSTRAP_PROBABILITY = 0.95
MIN_AUC_DELTA = -0.0005
MAX_INCOMING_30_REGRESSION = 0.002
COMPARISON_ALIASES = {
    "common_primary": "common_champion",
    "common_base": "common_m0",
    "full_new_primary_vs_base": "full48_vs_m0",
    "full_new_primary_vs_random": "full48_vs_random",
}
COMPARISON_ROLES = {
    "common_champion": {
        "m0": "old_dataset_primary_variant_reference",
        "candidate": "new_dataset_primary_variant",
    },
    "common_m0": {
        "m0": "old_dataset_m0_reference", "candidate": "new_dataset_m0",
    },
    "full48_vs_m0": {
        "m0": "new_dataset_m0_reference", "candidate": "new_dataset_primary_variant",
    },
    "full48_vs_random": {
        "m0": "new_dataset_random_control_reference",
        "candidate": "new_dataset_primary_variant",
    },
}


class CommonDatasetComparisonError(RuntimeError):
    """共通state比較契約または学習成果物が不正。"""


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """両datasetと3seed OOFを検証し、固定gateを評価する。"""

    old_samples, old_manifest = single._load_analysis_dataset(args.old_dataset_root)
    new_samples, new_manifest = single._load_analysis_dataset(args.new_dataset_root)
    positions, identity = _identity_audit(old_samples, new_samples)
    production_sha = m0.file_sha256(REPO_ROOT / "src/production_config.py")
    identity["production_config_sha256"] = production_sha
    identity["production_dependency_contract"] = dependency_receipt(REPO_ROOT)
    identity["production_config_unchanged"] = (
        production_sha == EXPECTED_PRODUCTION_CONFIG_SHA256
    )
    expected_groups = tuple(args.expected_extra_source_groups)
    if len(expected_groups) != len(set(expected_groups)):
        raise CommonDatasetComparisonError("expected extra source groupが重複しています")
    extras = _extra_source_audit(
        old_samples, new_samples, new_manifest, positions,
        expected_groups,
    )
    old_artifacts, old_predictions = _ensemble(
        old_samples, old_manifest, args.old_training_roots,
    )
    new_artifacts, new_predictions = _ensemble(
        new_samples, new_manifest, args.new_training_roots,
    )
    comparisons = _comparisons(
        old_samples, new_samples, positions, old_predictions, new_predictions,
        args.seed,
    )
    gates = _decision_policy(
        _gate_checks(identity, extras, comparisons),
        bool(getattr(args, "diagnostic_only", False)), comparisons,
    )
    return _write(args, old_manifest, new_manifest, old_artifacts, new_artifacts,
                  identity, extras, comparisons, gates)


def _identity_audit(old: Any, new: Any) -> tuple[np.ndarray, dict[str, Any]]:
    old_ids, new_ids = map(np.asarray, (old.state_ids, new.state_ids))
    if len(set(map(str, old_ids))) != len(old_ids):
        raise CommonDatasetComparisonError("旧datasetのstate_idが重複しています")
    new_index = {str(value): index for index, value in enumerate(new_ids)}
    if len(new_index) != len(new_ids) or any(str(value) not in new_index for value in old_ids):
        raise CommonDatasetComparisonError("新datasetが旧state_idを一意に包含していません")
    positions = np.asarray([new_index[str(value)] for value in old_ids], dtype=np.int64)
    exact = {
        name: _array_equal(getattr(old, name), getattr(new, name)[positions])
        for name in EXACT_FIELDS
    }
    weight = _weight_audit(old, new, positions)
    report = {
        "old_state_count": len(old_ids), "new_state_count": len(new_ids),
        "extra_state_count": len(new_ids) - len(old_ids),
        "old_order_preserved": bool(np.array_equal(positions, np.arange(len(old_ids)))),
        "exact_fields": exact, "all_exact_fields_pass": all(exact.values()),
        "weight_semantics": weight,
    }
    return positions, report


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    kwargs = {"equal_nan": True} if np.issubdtype(left.dtype, np.inexact) else {}
    return bool(np.array_equal(left, right, **kwargs))


def _weight_audit(old: Any, new: Any, positions: np.ndarray) -> dict[str, Any]:
    old_weights = np.asarray(old.weights, dtype=np.float64)
    new_common = np.asarray(new.weights[positions], dtype=np.float64)
    ratios = new_common / old_weights
    expected_new = m0._equal_game_weights(np.asarray(new.game_keys))
    ratio_spread = float(ratios.max() - ratios.min())
    return {
        "old_recomputed_bit_equal": _array_equal(
            np.asarray(old.weights), m0._equal_game_weights(np.asarray(old.game_keys)),
        ),
        "new_recomputed_bit_equal": _array_equal(np.asarray(new.weights), expected_new),
        "common_positive": bool(np.all(old_weights > 0.0) and np.all(new_common > 0.0)),
        "common_ratio_mean": float(ratios.mean()),
        "common_ratio_spread": ratio_spread,
        "relative_weight_pass": ratio_spread <= WEIGHT_RATIO_TOLERANCE,
    }


def _extra_source_audit(
    old: Any, new: Any, manifest: Mapping[str, Any], positions: np.ndarray,
    expected_groups: tuple[str, ...],
) -> dict[str, Any]:
    common = np.zeros(len(new.state_ids), dtype=bool)
    common[positions] = True
    actual_groups = sorted(set(map(str, new.source_groups[~common])))
    old_games = set(map(str, old.game_keys))
    extra_games = set(map(str, new.game_keys[~common]))
    receipts = _extra_source_receipts(manifest, set(expected_groups))
    return {
        "expected_source_groups": sorted(expected_groups),
        "actual_source_groups": actual_groups,
        "source_groups_pass": actual_groups == sorted(expected_groups),
        "game_key_collision_count": len(old_games & extra_games),
        "quarantine_contract_pass": len(receipts) == len(expected_groups) and all(
            row["quarantine_contract_pass"]
            and row["all_unsupported_games_marked"] is True for row in receipts
        ),
        "source_receipts": receipts,
    }


def _extra_source_receipts(
    manifest: Mapping[str, Any], expected: set[str],
) -> list[dict[str, Any]]:
    groups, sources = manifest.get("source_group_ids"), manifest.get("sources")
    if not isinstance(groups, list) or not isinstance(sources, list) or len(groups) != len(sources):
        raise CommonDatasetComparisonError("新datasetのsource receiptが不正です")
    output = []
    for group, source in zip(groups, sources, strict=True):
        if str(group) not in expected:
            continue
        table = _load_json(Path(str(source["table_root"])) / "manifest.json")
        physical = table.get("physical_cross", {})
        output.append({
            "source_group_id": str(group), "target_id": source.get("target_id"),
            "partition_fold": source.get("partition_fold"),
            "quarantine_contract_pass": physical.get("quarantine_contract_pass") is True,
            "unsupported_game_count": physical.get("unsupported_game_count"),
            "all_unsupported_games_marked": physical.get("all_unsupported_games_marked"),
        })
    return output


def _ensemble(
    samples: Any, manifest: Mapping[str, Any], roots: Sequence[Path],
) -> tuple[tuple[Any, ...], dict[str, np.ndarray]]:
    artifacts = ensemble._load_artifacts(tuple(Path(root) for root in roots), samples, manifest)
    predictions = ensemble._ensemble_predictions(artifacts)
    required = {
        f"{name}__seed_{ensemble.ENSEMBLE_SEED_TOKEN}__calibrated"
        for name in ("m0", PRIMARY_VARIANT, RANDOM_VARIANT)
    }
    if not required.issubset(predictions):
        raise CommonDatasetComparisonError("比較に必要なensemble予測がありません")
    return artifacts, predictions


def _raw(samples: Any) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(samples, name)) for name in (
        "boards", "ledger_values", "ledger_usable", "available_ms",
        "online_segment_index",
    )}


def _probability(predictions: Mapping[str, np.ndarray], variant: str) -> np.ndarray:
    return np.asarray(predictions[
        f"{variant}__seed_{ensemble.ENSEMBLE_SEED_TOKEN}__calibrated"
    ])


def _comparisons(
    old: Any, new: Any, positions: np.ndarray,
    old_predictions: Mapping[str, np.ndarray], new_predictions: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    old_strata, new_strata = single.build_strata(old, _raw(old)), single.build_strata(new, _raw(new))
    old_champion = _probability(old_predictions, PRIMARY_VARIANT)
    new_champion = _probability(new_predictions, PRIMARY_VARIANT)
    common = single.compare_candidate(
        old, old_champion, new_champion[positions], old_strata, seed,
    )
    common_m0 = single.compare_candidate(
        old, _probability(old_predictions, "m0"),
        _probability(new_predictions, "m0")[positions], old_strata, seed + 1,
    )
    full_m0 = single.compare_candidate(
        new, _probability(new_predictions, "m0"), new_champion, new_strata, seed + 2,
    )
    full_random = single.compare_candidate(
        new, _probability(new_predictions, RANDOM_VARIANT), new_champion,
        new_strata, seed + 3,
    )
    return {"common_champion": common, "common_m0": common_m0,
            "full48_vs_m0": full_m0, "full48_vs_random": full_random}


def _gate_checks(
    identity: Mapping[str, Any], extras: Mapping[str, Any],
    comparisons: Mapping[str, Any],
) -> dict[str, Any]:
    common = comparisons["common_champion"]
    full = comparisons["full48_vs_m0"]
    random = comparisons["full48_vs_random"]
    common_overall, full_overall = common["overall"], full["overall"]
    incoming = common["strata"].get("incoming_at_least_30", {})
    checks = {
        "data_exact": identity["all_exact_fields_pass"],
        "weight_semantics": all((
            identity["weight_semantics"]["old_recomputed_bit_equal"],
            identity["weight_semantics"]["new_recomputed_bit_equal"],
            identity["weight_semantics"]["common_positive"],
            identity["weight_semantics"]["relative_weight_pass"],
        )),
        "production_dependencies_compatible": (
            identity["production_dependency_contract"]["compatible"]
            if "production_dependency_contract" in identity else identity["production_config_unchanged"]
        ),
        "extra_sources": extras["source_groups_pass"],
        "no_game_collision": extras["game_key_collision_count"] == 0,
        "quarantine": extras["quarantine_contract_pass"],
        "full_log_loss": full_overall["candidate_minus_m0_log_loss"] < 0.0,
        "full_fold_wins": full["fold_win_count"] >= 4,
        "full_bootstrap": full["source_cluster_bootstrap"]["probability_candidate_better"] >= MIN_FULL_BOOTSTRAP_PROBABILITY,
        "full_random": random["overall"]["candidate_minus_m0_log_loss"] < 0.0,
        "full_severe10": full_overall["candidate_severe"]["winner_below_10_percent_count"] <= full_overall["m0_severe"]["winner_below_10_percent_count"],
        "common_log_loss": common_overall["candidate_minus_m0_log_loss"] < 0.0,
        "common_fold_wins": common["fold_win_count"] >= 4,
        "common_fold_stability": max(row["candidate_minus_m0_log_loss"] for row in common["fold_rows"]) <= MAX_FOLD_REGRESSION,
        "common_bootstrap": common["source_cluster_bootstrap"]["probability_candidate_better"] >= MIN_BOOTSTRAP_PROBABILITY,
        "common_auc": common_overall["candidate"]["auc"] - common_overall["m0"]["auc"] >= MIN_AUC_DELTA,
        "common_severe10": common_overall["candidate_severe"]["winner_below_10_percent_count"] <= common_overall["m0_severe"]["winner_below_10_percent_count"],
        "common_severe20": common_overall["candidate_severe"]["winner_below_20_percent_count"] <= common_overall["m0_severe"]["winner_below_20_percent_count"],
        "common_incoming30": bool(incoming) and incoming["candidate_minus_m0_log_loss"] <= MAX_INCOMING_30_REGRESSION,
    }
    return {"checks": checks, "all_pass": all(checks.values()),
            "decision": "PROMOTE_DEVELOPMENT_CHAMPION" if all(checks.values()) else "KEEP_46V_CHAMPION"}


def _decision_policy(
    gates: Mapping[str, Any], diagnostic_only: bool,
    comparisons: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(gates)
    result["checks"] = dict(gates["checks"])
    result["diagnostic_only"] = diagnostic_only
    if diagnostic_only:
        result["standard_gate_decision"] = result["decision"]
        result["would_pass_standard_gate"] = bool(result["all_pass"])
        exact = _diagnostic_m0_exact_checks(comparisons)
        result["checks"].update(exact)
        result["diagnostic_integrity_pass"] = all(exact.values())
        result["all_pass"] = bool(result["all_pass"] and all(exact.values()))
        result["decision"] = "DIAGNOSTIC_ONLY"
    return result


def _diagnostic_m0_exact_checks(
    comparisons: Mapping[str, Any] | None,
) -> dict[str, bool]:
    """固定M0診断でensemble経路まで厳密同一かを検査する。"""

    if comparisons is None:
        return {}
    overall = comparisons["common_m0"]["overall"]
    log_loss_delta = float(overall["candidate_minus_m0_log_loss"])
    auc_delta = float(overall["candidate"]["auc"] - overall["m0"]["auc"])
    return {
        "diagnostic_common_m0_log_loss_exact": log_loss_delta == 0.0,
        "diagnostic_common_m0_auc_exact": auc_delta == 0.0,
    }


def _artifact_receipts(artifacts: Sequence[Any]) -> list[Mapping[str, Any]]:
    return [dict(item.receipt) for item in artifacts]


def _write(
    args: argparse.Namespace, old_manifest: Mapping[str, Any],
    new_manifest: Mapping[str, Any], old_artifacts: Sequence[Any],
    new_artifacts: Sequence[Any], identity: Mapping[str, Any], extras: Mapping[str, Any],
    comparisons: Mapping[str, Any], gates: Mapping[str, Any],
) -> dict[str, Any]:
    if args.output_root.exists():
        raise CommonDatasetComparisonError(f"出力先は新規必須です: {args.output_root}")
    report = {
        "format_version": FORMAT_VERSION, "not_production": True,
        "primary_variant": PRIMARY_VARIANT, "old_dataset": old_manifest["dataset"],
        "new_dataset": new_manifest["dataset"], "identity_audit": identity,
        "extra_source_audit": extras, "old_training": _artifact_receipts(old_artifacts),
        "new_training": _artifact_receipts(new_artifacts),
        "comparison_aliases": COMPARISON_ALIASES, "comparison_roles": COMPARISON_ROLES,
        "comparisons": comparisons, "adoption_gate": gates,
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    report_path = m0._write_json_exclusive(args.output_root / "comparison.json", report)
    m0._write_json_exclusive(args.output_root / "COMPLETE", {
        "format_version": COMPLETE_VERSION,
        "comparison_sha256": m0.file_sha256(report_path),
    })
    return report


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CommonDatasetComparisonError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise CommonDatasetComparisonError(f"JSON objectではありません: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dataset-root", type=Path, required=True)
    parser.add_argument("--new-dataset-root", type=Path, required=True)
    parser.add_argument("--old-training-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--new-training-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-extra-source-groups", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--diagnostic-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = compare(parse_args(argv))
    print(json.dumps(report["adoption_gate"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
