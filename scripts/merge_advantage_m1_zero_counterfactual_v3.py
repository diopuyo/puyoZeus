"""3分割したzero-counterfactual V3 OOFを完全な6-fold成果物へ統合する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import train_advantage_m0_current_cnn_v1 as base
from scripts import train_advantage_m1_causal_ledger_v1 as legacy
from scripts import train_advantage_m1_zero_counterfactual_v3 as trainer


MERGE_PLAN_VERSION = "advantage-m1-zero-counterfactual-oof-merge-plan/v3"
MERGE_RESULTS_VERSION = "advantage-m1-zero-counterfactual-oof-merged/v3"
MERGE_COMPLETE_VERSION = "advantage-m1-zero-counterfactual-oof-merge-complete/v3"
EXPECTED_COMPONENT_COUNT = 3
EXPECTED_FOLDS = tuple(range(1, 7))
PREDICTION_KINDS = ("raw", "calibrated")


class ZeroCounterfactualOofMergeV3Error(RuntimeError):
    """V3 partial OOFまたは統合receiptが固定契約に違反した。"""


@dataclass(frozen=True, slots=True)
class ComponentArtifactV3:
    root: Path
    complete: Mapping[str, Any]
    complete_sha256: str
    plan: Mapping[str, Any]
    report: Mapping[str, Any]
    predictions: Mapping[str, np.ndarray]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ZeroCounterfactualOofMergeV3Error(f"JSON receiptを読めません: {path}") from error
    if not isinstance(value, dict):
        raise ZeroCounterfactualOofMergeV3Error(f"JSON objectではありません: {path}")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as error:
        raise ZeroCounterfactualOofMergeV3Error(
            f"partial predictionを読めません: {path}"
        ) from error


def _validate_complete(
    complete: Mapping[str, Any], plan: Path, results: Path, predictions: Path,
) -> None:
    if complete.get("format_version") != trainer.COMPLETE_VERSION:
        raise ZeroCounterfactualOofMergeV3Error(f"partial COMPLETE形式が不正です: {plan.parent}")
    receipts = {
        "plan_sha256": plan, "results_sha256": results,
        "predictions_sha256": predictions,
    }
    for key, path in receipts.items():
        if not path.is_file() or complete.get(key) != base.file_sha256(path):
            raise ZeroCounterfactualOofMergeV3Error(
                f"partial {key}が一致しません: {path.parent}"
            )


def _load_component(root: Path) -> ComponentArtifactV3:
    resolved = root.resolve()
    complete_path, plan_path = resolved / "COMPLETE", resolved / "PLAN.json"
    result_path, prediction_path = resolved / "results.json", resolved / "oof_predictions.npz"
    complete = _load_json(complete_path)
    _validate_complete(complete, plan_path, result_path, prediction_path)
    plan, report = _load_json(plan_path), _load_json(result_path)
    if plan.get("format_version") != trainer.PLAN_VERSION or plan.get("not_production") is not True:
        raise ZeroCounterfactualOofMergeV3Error(f"partial PLAN形式が不正です: {resolved}")
    if report.get("format_version") != trainer.TRAINING_VERSION:
        raise ZeroCounterfactualOofMergeV3Error(f"partial results形式が不正です: {resolved}")
    if report.get("not_production") is not True or report.get("plan") != plan:
        raise ZeroCounterfactualOofMergeV3Error(
            f"partial resultsのPLAN receiptが不正です: {resolved}"
        )
    _validate_plan_contract(plan, report, resolved)
    return ComponentArtifactV3(
        resolved, complete, base.file_sha256(complete_path), plan, report,
        _load_npz(prediction_path),
    )


def _validate_plan_contract(
    plan: Mapping[str, Any], report: Mapping[str, Any], root: Path,
) -> None:
    expected = (
        ("dataset_format_version", trainer.builder.FORMAT_VERSION),
        ("input_schema_version", trainer.M1_INPUT_SCHEMA_VERSION),
        ("model_version", trainer.ZERO_COUNTERFACTUAL_MODEL_VERSION),
    )
    if any(plan.get(key) != value for key, value in expected):
        raise ZeroCounterfactualOofMergeV3Error(f"partial V3 schema契約が不正です: {root}")
    if plan.get("projected_inputs_used") is not False:
        raise ZeroCounterfactualOofMergeV3Error(f"projected一次入力は禁止です: {root}")
    if report.get("dataset_source_count") != plan.get("source_count"):
        raise ZeroCounterfactualOofMergeV3Error(f"dataset source件数receiptが不正です: {root}")


def _plan_signature(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "folds"}


def _component_folds(component: ComponentArtifactV3) -> tuple[int, ...]:
    values = component.plan.get("folds")
    if not isinstance(values, list) or not values:
        raise ZeroCounterfactualOofMergeV3Error(f"partial foldsが不正です: {component.root}")
    folds = tuple(int(value) for value in values)
    if len(folds) != len(set(folds)) or any(value not in EXPECTED_FOLDS for value in folds):
        raise ZeroCounterfactualOofMergeV3Error(f"partial foldsが不正です: {component.root}")
    return folds


def _validate_components(components: Sequence[ComponentArtifactV3]) -> None:
    if len(components) != EXPECTED_COMPONENT_COUNT:
        raise ZeroCounterfactualOofMergeV3Error(
            f"partial rootは{EXPECTED_COMPONENT_COUNT}件必須です"
        )
    roots = [component.root for component in components]
    if len(set(roots)) != len(roots):
        raise ZeroCounterfactualOofMergeV3Error("partial rootが重複しています")
    signature = _plan_signature(components[0].plan)
    if any(_plan_signature(item.plan) != signature for item in components[1:]):
        raise ZeroCounterfactualOofMergeV3Error("partial PLANはfolds以外を完全一致させてください")
    folds = [fold for item in components for fold in _component_folds(item)]
    if len(folds) != len(set(folds)) or sorted(folds) != list(EXPECTED_FOLDS):
        raise ZeroCounterfactualOofMergeV3Error("partial foldsが重複、欠損、または範囲外です")


def _result_keys(plan: Mapping[str, Any]) -> tuple[str, ...]:
    variants, seeds = plan.get("variants"), plan.get("seeds")
    if not isinstance(variants, list) or not isinstance(seeds, list):
        raise ZeroCounterfactualOofMergeV3Error("PLAN variants/seedsが不正です")
    if (not variants or len(variants) != len(set(variants))
            or any(value not in trainer.ZERO_VARIANTS for value in variants)):
        raise ZeroCounterfactualOofMergeV3Error("PLAN variantsが不正です")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ZeroCounterfactualOofMergeV3Error("PLAN seedsが不正です")
    names = ("m0", *(f"m1_zero_{value}" for value in variants))
    return tuple(f"{name}__seed_{int(seed)}" for seed in seeds for name in names)


def _load_shared_dataset(
    components: Sequence[ComponentArtifactV3],
) -> tuple[trainer.CanonicalSamplesV3, Mapping[str, Any]]:
    root = Path(str(components[0].plan.get("dataset_root")))
    samples, manifest = trainer.load_canonical_dataset_v2(root)
    expected = components[0].plan.get("dataset_sha256")
    if manifest.get("dataset", {}).get("sha256") != expected:
        raise ZeroCounterfactualOofMergeV3Error("共有dataset SHAがpartial PLANと一致しません")
    if len(samples.labels) != components[0].plan.get("state_count"):
        raise ZeroCounterfactualOofMergeV3Error("共有dataset行数がpartial PLANと一致しません")
    _validate_dataset_identity(components, root, manifest)
    return samples, manifest


def _validate_dataset_identity(
    components: Sequence[ComponentArtifactV3], root: Path,
    manifest: Mapping[str, Any],
) -> None:
    current = trainer.dataset_identity_receipt(root, manifest)
    for component in components:
        expected = {
            key: component.plan.get(key) for key in current
        }
        if expected != current:
            raise ZeroCounterfactualOofMergeV3Error(
                f"dataset/fold receiptがpartial PLANと不一致です: {component.root}"
            )


def _validate_prediction_payload(
    component: ComponentArtifactV3, samples: trainer.CanonicalSamplesV3,
) -> None:
    results, keys = component.report.get("results"), _result_keys(component.plan)
    if not isinstance(results, dict) or set(results) != set(keys):
        raise ZeroCounterfactualOofMergeV3Error(
            f"partial result variant集合が不正です: {component.root}"
        )
    expected_names = {"labels", "folds"} | {
        f"{key}__{kind}" for key in keys for kind in PREDICTION_KINDS
    }
    if set(component.predictions) != expected_names:
        raise ZeroCounterfactualOofMergeV3Error(
            f"partial prediction schemaが不正です: {component.root}"
        )
    if not np.array_equal(component.predictions["labels"], samples.labels):
        raise ZeroCounterfactualOofMergeV3Error(
            f"partial labelsがdatasetと不一致です: {component.root}"
        )
    if not np.array_equal(component.predictions["folds"], samples.folds):
        raise ZeroCounterfactualOofMergeV3Error(
            f"partial folds配列がdatasetと不一致です: {component.root}"
        )
    expected_mask = np.isin(samples.folds, _component_folds(component))
    for key in keys:
        _validate_prediction_pair(component, key, expected_mask)


def _validate_prediction_pair(
    component: ComponentArtifactV3, key: str, expected_mask: np.ndarray,
) -> None:
    raw = np.asarray(component.predictions[f"{key}__raw"])
    calibrated = np.asarray(component.predictions[f"{key}__calibrated"])
    if raw.shape != expected_mask.shape or calibrated.shape != expected_mask.shape:
        raise ZeroCounterfactualOofMergeV3Error(
            f"prediction shapeが不正です: {component.root}/{key}"
        )
    if not np.issubdtype(raw.dtype, np.floating) or calibrated.dtype != raw.dtype:
        raise ZeroCounterfactualOofMergeV3Error(
            f"prediction dtypeが不正です: {component.root}/{key}"
        )
    raw_finite, calibrated_finite = np.isfinite(raw), np.isfinite(calibrated)
    if not np.array_equal(raw_finite, calibrated_finite):
        raise ZeroCounterfactualOofMergeV3Error(
            f"raw/calibrated finite maskが不一致です: {component.root}/{key}"
        )
    if not np.array_equal(raw_finite, expected_mask):
        raise ZeroCounterfactualOofMergeV3Error(
            f"prediction maskと指定foldが不一致です: {component.root}/{key}"
        )


def _records(component: ComponentArtifactV3, key: str) -> list[Mapping[str, Any]]:
    result = component.report["results"][key]
    records = result.get("records") if isinstance(result, dict) else None
    if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
        raise ZeroCounterfactualOofMergeV3Error(f"recordsが不正です: {component.root}/{key}")
    folds = [int(row.get("eval_fold", -1)) for row in records]
    expected = sorted(_component_folds(component))
    if sorted(folds) != expected or len(folds) != len(set(folds)):
        raise ZeroCounterfactualOofMergeV3Error(
            f"component record foldsが不正です: {component.root}/{key}"
        )
    return records


def _baseline_key(key: str) -> str:
    try:
        return f"m0__seed_{key.rsplit('__seed_', 1)[1]}"
    except IndexError as error:
        raise ZeroCounterfactualOofMergeV3Error(f"result keyが不正です: {key}") from error


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _validate_m0_link(
    component: ComponentArtifactV3, key: str, record: Mapping[str, Any],
) -> None:
    if key.startswith("m0__seed_"):
        if not _is_sha256(record.get("m0_state_sha256")):
            raise ZeroCounterfactualOofMergeV3Error("M0 state SHAがありません")
        return
    if (record.get("model_version") != trainer.ZERO_COUNTERFACTUAL_MODEL_VERSION
            or record.get("input_schema_version") != trainer.M1_INPUT_SCHEMA_VERSION):
        raise ZeroCounterfactualOofMergeV3Error(f"candidateのV3 schema receiptが不一致です: {key}")
    baseline = {
        int(row["eval_fold"]): row for row in _records(component, _baseline_key(key))
    }
    source = baseline[int(record["eval_fold"])]
    if (record.get("m0_model") != source.get("model")
            or record.get("m0_model_sha256") != source.get("model_sha256")):
        raise ZeroCounterfactualOofMergeV3Error(
            f"candidateのM0 checkpoint receiptが不一致です: {key}"
        )
    state = source.get("m0_state_sha256")
    if (not _is_sha256(state) or record.get("m0_state_sha256_before") != state
            or record.get("m0_state_sha256_after") != state):
        raise ZeroCounterfactualOofMergeV3Error(f"candidateのM0 state SHAが不一致です: {key}")


def _checkpoint_reference(
    root: Path, name: Any, expected_sha: Any, cache: dict[Path, str],
) -> dict[str, str]:
    path = (root / str(name)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ZeroCounterfactualOofMergeV3Error(f"checkpoint参照が不正です: {path}")
    actual = cache.get(path)
    if actual is None:
        actual = base.file_sha256(path)
        cache[path] = actual
    if actual != expected_sha:
        raise ZeroCounterfactualOofMergeV3Error(f"checkpoint SHAが不一致です: {path}")
    return {"component_root": str(root), "path": str(path), "sha256": actual}


def _referenced_record(
    component: ComponentArtifactV3, key: str, record: Mapping[str, Any],
    cache: dict[Path, str],
) -> dict[str, Any]:
    _validate_m0_link(component, key, record)
    output = dict(record)
    output["checkpoint_reference"] = _checkpoint_reference(
        component.root, record.get("model"), record.get("model_sha256"), cache,
    )
    if not key.startswith("m0__seed_"):
        output["m0_checkpoint_reference"] = _checkpoint_reference(
            component.root, record.get("m0_model"), record.get("m0_model_sha256"), cache,
        )
    return output


def _merged_records(
    components: Sequence[ComponentArtifactV3], key: str, cache: dict[Path, str],
) -> list[dict[str, Any]]:
    output = [
        _referenced_record(component, key, record, cache)
        for component in components for record in _records(component, key)
    ]
    folds = [int(record["eval_fold"]) for record in output]
    if sorted(folds) != list(EXPECTED_FOLDS) or len(folds) != len(set(folds)):
        raise ZeroCounterfactualOofMergeV3Error(
            f"統合record eval_foldが1..6一意ではありません: {key}"
        )
    return sorted(output, key=lambda record: int(record["eval_fold"]))


def _merge_prediction_array(
    components: Sequence[ComponentArtifactV3], name: str, count: int,
) -> np.ndarray:
    first = np.asarray(components[0].predictions[name])
    merged = np.full(count, np.nan, dtype=first.dtype)
    coverage = np.zeros(count, dtype=np.int8)
    for component in components:
        value = np.asarray(component.predictions[name])
        if value.dtype != first.dtype:
            raise ZeroCounterfactualOofMergeV3Error(
                f"prediction dtypeがroot間で不一致です: {name}"
            )
        mask = np.isfinite(value)
        if np.any(coverage[mask]):
            raise ZeroCounterfactualOofMergeV3Error(
                f"partial predictionがroot間で重複しています: {name}"
            )
        merged[mask], coverage[mask] = value[mask], coverage[mask] + 1
    if not np.all(coverage == 1) or not np.isfinite(merged).all():
        raise ZeroCounterfactualOofMergeV3Error(
            f"partial predictionが全行を1回coverしません: {name}"
        )
    return merged


def _merge_predictions(
    components: Sequence[ComponentArtifactV3], samples: trainer.CanonicalSamplesV3,
) -> dict[str, np.ndarray]:
    output = {"labels": samples.labels.copy(), "folds": samples.folds.copy()}
    for key in _result_keys(components[0].plan):
        for kind in PREDICTION_KINDS:
            name = f"{key}__{kind}"
            output[name] = _merge_prediction_array(components, name, len(samples.labels))
    return output


def _merge_results(
    components: Sequence[ComponentArtifactV3], samples: trainer.CanonicalSamplesV3,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    cache: dict[Path, str] = {}
    output: dict[str, Any] = {}
    for key in _result_keys(components[0].plan):
        raw, calibrated = predictions[f"{key}__raw"], predictions[f"{key}__calibrated"]
        output[key] = {
            "records": _merged_records(components, key, cache),
            "metrics_raw": legacy.metrics(samples, raw),
            "metrics_calibrated": legacy.metrics(samples, calibrated),
        }
    return output


def _checkpoint_receipts(component: ComponentArtifactV3) -> list[dict[str, str]]:
    cache: dict[Path, str] = {}
    for key in _result_keys(component.plan):
        for record in _records(component, key):
            _referenced_record(component, key, record, cache)
    return [
        {"path": str(path), "sha256": digest}
        for path, digest in sorted(cache.items(), key=lambda item: str(item[0]))
    ]


def _component_receipt(component: ComponentArtifactV3) -> dict[str, Any]:
    complete_path = component.root / "COMPLETE"
    if base.file_sha256(complete_path) != component.complete_sha256:
        raise ZeroCounterfactualOofMergeV3Error(
            f"merge中にCOMPLETEが変化しました: {component.root}"
        )
    current = _load_json(complete_path)
    if current != component.complete:
        raise ZeroCounterfactualOofMergeV3Error(
            f"merge中にCOMPLETEが変化しました: {component.root}"
        )
    plan, results = component.root / "PLAN.json", component.root / "results.json"
    predictions = component.root / "oof_predictions.npz"
    _validate_complete(current, plan, results, predictions)
    return {
        "component_root": str(component.root), "folds": list(_component_folds(component)),
        "complete_sha256": base.file_sha256(complete_path),
        "plan_sha256": base.file_sha256(plan), "results_sha256": base.file_sha256(results),
        "predictions_sha256": base.file_sha256(predictions),
        "checkpoints": _checkpoint_receipts(component),
    }


def _merged_plan(components: Sequence[ComponentArtifactV3]) -> dict[str, Any]:
    plan = dict(components[0].plan)
    plan.update({
        "format_version": MERGE_PLAN_VERSION,
        "source_plan_format_version": trainer.PLAN_VERSION,
        "source_results_format_version": trainer.TRAINING_VERSION,
        "source_complete_format_version": trainer.COMPLETE_VERSION,
        "folds": list(EXPECTED_FOLDS),
        "component_receipts": [_component_receipt(item) for item in components],
        "merger_sha256": base.file_sha256(Path(__file__)), "not_production": True,
    })
    return plan


def _assert_sources_unchanged(
    components: Sequence[ComponentArtifactV3], receipts: Sequence[Mapping[str, Any]],
) -> None:
    if len(components) != len(receipts):
        raise ZeroCounterfactualOofMergeV3Error("component receipt件数が不正です")
    for component, expected in zip(components, receipts, strict=True):
        if _component_receipt(component) != expected:
            raise ZeroCounterfactualOofMergeV3Error(
                f"merge中にsourceが変化しました: {component.root}"
            )


def _write_output(
    root: Path, plan: Mapping[str, Any], results: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray], manifest: Mapping[str, Any],
    components: Sequence[ComponentArtifactV3],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    plan_path = base._write_json_exclusive(root / "PLAN.json", plan)
    prediction_path = root / "oof_predictions.npz"
    legacy._save_npz_exclusive(prediction_path, predictions)
    report = {
        "format_version": MERGE_RESULTS_VERSION, "not_production": True,
        "dataset_source_count": manifest["source_count"],
        "plan": dict(plan), "results": dict(results),
    }
    result_path = base._write_json_exclusive(root / "results.json", report)
    _assert_sources_unchanged(components, plan["component_receipts"])
    base._write_json_exclusive(root / "COMPLETE", {
        "format_version": MERGE_COMPLETE_VERSION,
        "plan_sha256": base.file_sha256(plan_path),
        "results_sha256": base.file_sha256(result_path),
        "predictions_sha256": base.file_sha256(prediction_path),
    })
    return report


def merge(component_roots: Sequence[Path], output_root: Path) -> dict[str, Any]:
    """検証済みV3 partial OOFのみを新規rootへ統合する。"""

    if output_root.exists():
        raise ZeroCounterfactualOofMergeV3Error(f"出力先は新規必須です: {output_root}")
    components = tuple(_load_component(root) for root in component_roots)
    _validate_components(components)
    samples, manifest = _load_shared_dataset(components)
    for component in components:
        _validate_prediction_payload(component, samples)
    predictions = _merge_predictions(components, samples)
    results = _merge_results(components, samples, predictions)
    plan = _merged_plan(components)
    return _write_output(
        output_root, plan, results, predictions, manifest, components,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.component_root) != EXPECTED_COMPONENT_COUNT:
        parser.error(f"--component-rootは{EXPECTED_COMPONENT_COUNT}件必須です")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = merge(args.component_root, args.output_root)
    print(json.dumps({"result_count": len(report["results"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
