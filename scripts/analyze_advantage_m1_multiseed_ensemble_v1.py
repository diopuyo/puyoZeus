"""固定seed別M1 OOFを等重み平均し、再現可能な層別監査を行う。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import analyze_advantage_m1_fixed_oof_v1 as single
from scripts import train_advantage_m0_current_cnn_v1 as base


ANALYSIS_VERSION = "advantage-m1-multiseed-ensemble-analysis/v1"
COMPLETE_VERSION = "advantage-m1-multiseed-ensemble-analysis-complete/v1"
ENSEMBLE_SEED_TOKEN = "ensemble"
ENSEMBLE_METHOD = "equal_arithmetic_mean_across_fixed_seeds"
PREDICTION_KINDS = ("raw", "calibrated")
PLAN_SIGNATURE_FIELDS = (
    "dataset_sha256", "dataset_format_version", "input_schema_version",
    "model_version", "variants", "folds", "epochs", "patience", "batch_size",
    "learning_rate", "weight_decay", "calibration_policy", "code_sha256",
    "source_count", "state_count", "game_count", "fixed_outer_folds",
    "projected_inputs_used", "projected_join_role", "fallback_policy",
    "random_policy", "residual_definition", "source_group_fold_mapping_sha256",
    "dataset_manifest_sha256", "dataset_complete_sha256",
    "source_complete_format_version", "source_plan_format_version",
    "source_results_format_version",
)


class MultiSeedEnsembleAnalysisError(RuntimeError):
    """seed別OOFまたは等重み監査契約が不正。"""


@dataclass(frozen=True, slots=True)
class TrainingArtifact:
    root: Path
    seeds: tuple[int, ...]
    variants: tuple[str, ...]
    report: Mapping[str, Any]
    predictions: Mapping[str, np.ndarray]
    receipt: Mapping[str, Any]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    """検証済み複数seed OOFを等重み平均し、新規rootへ監査結果を書く。"""

    samples, manifest = single._load_analysis_dataset(args.dataset_root)
    roots = tuple(Path(root) for root in args.training_roots)
    artifacts = _load_artifacts(roots, samples, manifest)
    predictions = _ensemble_predictions(artifacts)
    raw = single._load_raw_dataset(args.dataset_root)
    strata = single.build_strata(samples, raw)
    comparisons = _m0_comparisons(samples, predictions, strata, args.seed)
    matched = _matched_random_comparisons(samples, predictions, strata, args.seed)
    state_rows = single._state_rows_if_requested(args, manifest)
    severe = single.build_severe_diagnostics(samples, raw, predictions, state_rows)
    report = _report(
        args, samples, manifest, artifacts, predictions, strata,
        comparisons, matched, severe,
    )
    return _write(args.output_root, report, artifacts)


def _load_artifacts(
    roots: Sequence[Path], samples: single.AnalysisSamples,
    manifest: Mapping[str, Any],
) -> tuple[TrainingArtifact, ...]:
    if not roots or len(set(root.resolve() for root in roots)) != len(roots):
        raise MultiSeedEnsembleAnalysisError("training rootは重複なしの1件以上が必要です")
    loaded = tuple(_load_artifact(root, samples, manifest) for root in roots)
    artifacts = tuple(sorted(loaded, key=lambda item: item.seeds))
    seeds = tuple(seed for item in artifacts for seed in item.seeds)
    if len(set(seeds)) != len(seeds):
        raise MultiSeedEnsembleAnalysisError("training seedが重複しています")
    if len(seeds) < 2:
        raise MultiSeedEnsembleAnalysisError("training seedは合計2件以上必要です")
    _validate_cross_root_contract(artifacts)
    return artifacts


def _load_artifact(
    root: Path, samples: single.AnalysisSamples, manifest: Mapping[str, Any],
) -> TrainingArtifact:
    resolved = root.resolve()
    report = single._load_training(resolved)
    single._validate_training_dataset(report, manifest)
    predictions = single._load_predictions(resolved, samples)
    complete = _load_json(resolved / "COMPLETE")
    _validate_complete(complete, resolved)
    _validate_plan_receipt(report, resolved)
    seeds, variants = _prediction_contract(report, predictions, len(samples.labels))
    return TrainingArtifact(
        resolved, seeds, variants, report, predictions,
        _artifact_receipt(resolved, complete, report, predictions),
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MultiSeedEnsembleAnalysisError(f"JSON receiptを読めません: {path}") from error
    if not isinstance(value, dict):
        raise MultiSeedEnsembleAnalysisError(f"JSON objectではありません: {path}")
    return value


def _validate_complete(complete: Mapping[str, Any], root: Path) -> None:
    if not isinstance(complete.get("format_version"), str):
        raise MultiSeedEnsembleAnalysisError(f"training COMPLETE versionがありません: {root}")
    receipts = {
        "plan_sha256": root / "PLAN.json",
        "results_sha256": root / "results.json",
        "predictions_sha256": root / "oof_predictions.npz",
    }
    for key, path in receipts.items():
        if not path.is_file() or complete.get(key) != base.file_sha256(path):
            raise MultiSeedEnsembleAnalysisError(f"training {key}が不一致です: {root}")


def _validate_plan_receipt(report: Mapping[str, Any], root: Path) -> None:
    if report.get("plan") != _load_json(root / "PLAN.json"):
        raise MultiSeedEnsembleAnalysisError(f"PLAN.jsonとresults planが不一致です: {root}")


def _prediction_contract(
    report: Mapping[str, Any], predictions: Mapping[str, np.ndarray], count: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    parsed = [_parse_prediction_key(name) for name in predictions if name not in {"labels", "folds"}]
    if not parsed or any(item is None for item in parsed):
        raise MultiSeedEnsembleAnalysisError("prediction key形式が不正です")
    entries = tuple(item for item in parsed if item is not None)
    seeds = tuple(sorted({item[1] for item in entries}))
    variants = tuple(sorted({item[0] for item in entries}))
    if not seeds or "m0" not in variants:
        raise MultiSeedEnsembleAnalysisError("rootごとに1件以上のseedとM0が必要です")
    _validate_prediction_arrays(predictions, variants, seeds, count)
    _validate_report_keys(report, variants, seeds)
    return seeds, variants


def _parse_prediction_key(name: str) -> tuple[str, int, str] | None:
    model_seed, separator, kind = name.rpartition("__")
    variant, seed_separator, seed_text = model_seed.rpartition("__seed_")
    if not separator or not seed_separator or kind not in PREDICTION_KINDS:
        return None
    try:
        return variant, int(seed_text), kind
    except ValueError:
        return None


def _validate_prediction_arrays(
    predictions: Mapping[str, np.ndarray], variants: Sequence[str],
    seeds: Sequence[int], count: int,
) -> None:
    expected = {"labels", "folds"} | {
        f"{variant}__seed_{seed}__{kind}"
        for variant in variants for seed in seeds for kind in PREDICTION_KINDS
    }
    if set(predictions) != expected:
        raise MultiSeedEnsembleAnalysisError("raw/calibrated prediction key集合が不一致です")
    for name in expected - {"labels", "folds"}:
        values = np.asarray(predictions[name])
        if values.shape != (count,) or not np.issubdtype(values.dtype, np.floating):
            raise MultiSeedEnsembleAnalysisError(f"prediction shape/dtypeが不正です: {name}")
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise MultiSeedEnsembleAnalysisError(f"prediction値が不正です: {name}")


def _validate_report_keys(
    report: Mapping[str, Any], variants: Sequence[str], seeds: Sequence[int],
) -> None:
    plan, results = report.get("plan"), report.get("results")
    if report.get("not_production") is not True:
        raise MultiSeedEnsembleAnalysisError("training resultsはnot_production必須です")
    if not isinstance(plan, Mapping) or not isinstance(results, Mapping):
        raise MultiSeedEnsembleAnalysisError("training plan/resultsが不正です")
    expected = {f"{variant}__seed_{seed}" for variant in variants for seed in seeds}
    plan_seeds = plan.get("seeds")
    if set(results) != expected or not isinstance(plan_seeds, list):
        raise MultiSeedEnsembleAnalysisError("results/PLANのseed・variant集合が不一致です")
    if any(type(value) is not int for value in plan_seeds):
        raise MultiSeedEnsembleAnalysisError("results/PLANのseed・variant集合が不一致です")
    if sorted(plan_seeds) != list(seeds) or len(set(plan_seeds)) != len(plan_seeds):
        raise MultiSeedEnsembleAnalysisError("results/PLANのseed・variant集合が不一致です")


def _artifact_receipt(
    root: Path, complete: Mapping[str, Any], report: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    plan = report["plan"]
    return {
        "training_root": str(root), "complete_sha256": base.file_sha256(root / "COMPLETE"),
        "results_sha256": complete["results_sha256"],
        "predictions_sha256": complete["predictions_sha256"],
        "plan_sha256": complete["plan_sha256"],
        "dataset_sha256": plan["dataset_sha256"],
        "labels_sha256": _array_sha256(predictions["labels"]),
        "folds_sha256": _array_sha256(predictions["folds"]),
        "seeds": sorted(report["plan"]["seeds"]),
        "complete_format_version": complete["format_version"],
        "results_format_version": report.get("format_version"),
    }


def _validate_cross_root_contract(artifacts: Sequence[TrainingArtifact]) -> None:
    variants = artifacts[0].variants
    signature = _plan_signature(artifacts[0].report["plan"])
    labels_sha = artifacts[0].receipt["labels_sha256"]
    folds_sha = artifacts[0].receipt["folds_sha256"]
    for item in artifacts[1:]:
        if item.variants != variants or _plan_signature(item.report["plan"]) != signature:
            raise MultiSeedEnsembleAnalysisError("seed間のvariant/実験PLANが一致しません")
        if item.receipt["labels_sha256"] != labels_sha:
            raise MultiSeedEnsembleAnalysisError("seed間のlabels順が一致しません")
        if item.receipt["folds_sha256"] != folds_sha:
            raise MultiSeedEnsembleAnalysisError("seed間のfolds順が一致しません")


def _plan_signature(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {name: plan.get(name) for name in PLAN_SIGNATURE_FIELDS}


def _ensemble_predictions(
    artifacts: Sequence[TrainingArtifact],
) -> dict[str, np.ndarray]:
    first = artifacts[0]
    output = {
        "labels": np.asarray(first.predictions["labels"]).copy(),
        "folds": np.asarray(first.predictions["folds"]).copy(),
    }
    for variant in first.variants:
        for kind in PREDICTION_KINDS:
            arrays = [
                item.predictions[f"{variant}__seed_{seed}__{kind}"]
                for item in artifacts for seed in item.seeds
            ]
            name = f"{variant}__seed_{ENSEMBLE_SEED_TOKEN}__{kind}"
            output[name] = np.mean(np.stack(arrays).astype(np.float64), axis=0)
    return output


def _candidate_variants(predictions: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    suffix = f"__seed_{ENSEMBLE_SEED_TOKEN}__calibrated"
    variants = tuple(sorted(name[:-len(suffix)] for name in predictions if name.endswith(suffix)))
    return tuple(name for name in variants if name != "m0" and not name.endswith("random_control"))


def _m0_comparisons(
    samples: single.AnalysisSamples, predictions: Mapping[str, np.ndarray],
    strata: Mapping[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    baseline = predictions[f"m0__seed_{ENSEMBLE_SEED_TOKEN}__calibrated"]
    return {
        variant: single.compare_candidate(
            samples, baseline,
            predictions[f"{variant}__seed_{ENSEMBLE_SEED_TOKEN}__calibrated"],
            strata, seed,
        )
        for variant in _candidate_variants(predictions)
    }


def _matched_random_comparisons(
    samples: single.AnalysisSamples, predictions: Mapping[str, np.ndarray],
    strata: Mapping[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in _candidate_variants(predictions):
        reference = single._matched_random_variant(variant)
        reference_key = f"{reference}__seed_{ENSEMBLE_SEED_TOKEN}__calibrated"
        if reference is None or reference_key not in predictions:
            raise MultiSeedEnsembleAnalysisError(f"matched randomがありません: {variant}")
        compared = single.compare_candidate(
            samples, predictions[reference_key],
            predictions[f"{variant}__seed_{ENSEMBLE_SEED_TOKEN}__calibrated"],
            strata, seed,
        )
        output[variant] = single._genericize_comparison(compared)
        output[variant]["reference_prediction"] = reference
    return output


def _dataset_order_receipt(
    samples: single.AnalysisSamples, manifest: Mapping[str, Any],
) -> dict[str, Any]:
    state_ids = getattr(samples, "state_ids", None)
    if state_ids is not None:
        computed = _builder_array_sha256(np.asarray(state_ids))
        recorded = _manifest_state_id_sha256(manifest)
        if recorded is None or recorded != computed:
            raise MultiSeedEnsembleAnalysisError(
                "dataset state_id ordered SHA receiptが不一致です"
            )
        return {
            "state_id_available": True, "row_count": len(samples.labels),
            "manifest_state_id_ordered_sha256": recorded,
            "computed_state_id_ordered_sha256": computed,
            "ordered_state_id_json_lines_sha256": _ordered_text_sha256(state_ids),
            "method": "dataset_builder_dtype_nul_shape_nul_c_order_bytes",
        }
    identities = (
        f"{index}:{group}:{game}:{fold}"
        for index, (group, game, fold) in enumerate(zip(
            samples.source_groups, samples.game_keys, samples.folds, strict=True,
        ))
    )
    return {
        "state_id_available": False, "row_count": len(samples.labels),
        "ordered_dataset_identity_sha256": _ordered_text_sha256(identities),
        "method": "row_index_source_group_game_fold_utf8_json_lines",
        "limitation": "datasetにstate_id列がなく完全なstate identity receiptではない",
    }


def _ordered_text_sha256(values: Sequence[Any] | Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        line = json.dumps(str(value), ensure_ascii=False, separators=(",", ":"))
        digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def _manifest_state_id_sha256(manifest: Mapping[str, Any]) -> str | None:
    for key in ("shared_tensorizer_full_receipt", "shared_tensorizer_receipt"):
        receipt = manifest.get(key)
        if isinstance(receipt, Mapping):
            value = receipt.get("state_id_ordered_sha256")
            return str(value) if value else None
    return None


def _builder_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return _builder_array_sha256(values)


def _report(
    args: argparse.Namespace, samples: single.AnalysisSamples, manifest: Mapping[str, Any],
    artifacts: Sequence[TrainingArtifact], predictions: Mapping[str, np.ndarray],
    strata: Mapping[str, np.ndarray], comparisons: Mapping[str, Any],
    matched: Mapping[str, Any], severe: Mapping[str, Any],
) -> dict[str, Any]:
    seeds = sorted(seed for item in artifacts for seed in item.seeds)
    return {
        "format_version": ANALYSIS_VERSION, "not_production": True,
        "analysis_code_sha256": base.file_sha256(Path(__file__)),
        "dataset": {
            "root": str(args.dataset_root.resolve()),
            "sha256": manifest["dataset"]["sha256"],
            "order_receipt": _dataset_order_receipt(samples, manifest),
        },
        "ensemble": {
            "method": ENSEMBLE_METHOD, "optimization_used": False,
            "seeds": seeds, "seed_count": len(seeds),
            "equal_weight": 1.0 / len(seeds), "arithmetic_dtype": "float64",
            "prediction_kinds": list(PREDICTION_KINDS),
            "variants": list(artifacts[0].variants),
            "synthetic_prediction_seed_token": ENSEMBLE_SEED_TOKEN,
            "training_artifacts": [dict(item.receipt) for item in artifacts],
            "ensembled_labels_sha256": _array_sha256(predictions["labels"]),
            "ensembled_folds_sha256": _array_sha256(predictions["folds"]),
            "ensemble_prediction_sha256": {
                name: _array_sha256(value) for name, value in predictions.items()
                if name not in {"labels", "folds"}
            },
            "array_sha256_semantics": "dtype NUL shape-json NUL C-order-bytes",
        },
        "stratum_counts": {name: int(mask.sum()) for name, mask in strata.items()},
        "comparisons_vs_ensemble_m0": dict(comparisons),
        "matched_random_direct_comparisons": dict(matched),
        "severe_error_diagnostics": dict(severe),
        "interpretation_contract": {
            "probability_ensemble_is_unoptimized_equal_mean": True,
            "raw_and_calibrated_ensembled_separately": True,
            "winner_label_used_for_diagnosis_only": True,
        },
    }


def _assert_artifacts_unchanged(artifacts: Sequence[TrainingArtifact]) -> None:
    for item in artifacts:
        current = {
            "complete_sha256": base.file_sha256(item.root / "COMPLETE"),
            "results_sha256": base.file_sha256(item.root / "results.json"),
            "predictions_sha256": base.file_sha256(item.root / "oof_predictions.npz"),
        }
        if any(current[name] != item.receipt[name] for name in current):
            raise MultiSeedEnsembleAnalysisError(f"解析中にtraining rootが変化しました: {item.root}")


def _write(
    root: Path, report: Mapping[str, Any], artifacts: Sequence[TrainingArtifact],
) -> dict[str, Any]:
    if root.exists():
        raise MultiSeedEnsembleAnalysisError(f"出力先は新規必須です: {root}")
    root.mkdir(parents=True, exist_ok=False)
    _assert_artifacts_unchanged(artifacts)
    path = base._write_json_exclusive(root / "analysis.json", report)
    _assert_artifacts_unchanged(artifacts)
    base._write_json_exclusive(root / "COMPLETE", {
        "format_version": COMPLETE_VERSION,
        "analysis_sha256": base.file_sha256(path),
    })
    return dict(report)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--table-roots", type=Path, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = analyze(parse_args(argv))
    print(json.dumps({
        "seed_count": report["ensemble"]["seed_count"],
        "comparison_count": len(report["comparisons_vs_ensemble_m0"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
