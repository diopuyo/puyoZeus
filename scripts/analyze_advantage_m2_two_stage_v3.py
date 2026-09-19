"""M2二段階3seed OOFをM1およびaux-offと同一行で独立監査する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import build_advantage_m2_dataset_v1 as builder
from scripts import train_advantage_m0_current_cnn_v1 as base
from scripts import train_advantage_m1_causal_ledger_v1 as legacy
from scripts import train_advantage_m2_auxiliary_v1 as m2_train


ANALYSIS_VERSION = "advantage-m2-two-stage-multiseed-analysis/v1"
COMPLETE_VERSION = "advantage-m2-two-stage-multiseed-analysis-complete/v1"
DEFAULT_SEEDS = (20260904, 20260905, 20260906)
VARIANTS = ("m2_aux_off", "m2_aux_on_0p3")
M1_VARIANT = "m1_zero_values_and_masks"
BOOTSTRAP_REPLICATES = 2_000


class AdvantageM2AnalysisError(RuntimeError):
    """M2/M1 OOF同一性、coverage、監査保存契約に違反した。"""


@dataclass(frozen=True, slots=True)
class MergedM2:
    labels: np.ndarray
    folds: np.ndarray
    state_ids: np.ndarray
    predictions: Mapping[str, np.ndarray]
    records: Mapping[str, tuple[Mapping[str, Any], ...]]
    receipts: tuple[Mapping[str, Any], ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM2AnalysisError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise AdvantageM2AnalysisError(f"JSON objectではありません: {path}")
    return value


def _load_artifact(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    complete, results = _read_json(root / "COMPLETE"), _read_json(root / "results.json")
    prediction_path = root / "oof_predictions.npz"
    checks = (
        ("results_sha256", root / "results.json"),
        ("predictions_sha256", prediction_path),
    )
    if any(complete.get(field) != base.file_sha256(path) for field, path in checks):
        raise AdvantageM2AnalysisError(f"成果物hashが一致しません: {root}")
    try:
        with np.load(prediction_path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as error:
        raise AdvantageM2AnalysisError(f"OOF NPZを読めません: {root}") from error
    return arrays, results, complete


def merge_m2_roots(roots: Sequence[Path], seeds: Sequence[int]) -> MergedM2:
    """分割保存されたM2 OOFを重複なしで全行へ結合する。"""

    identity: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    merged: dict[str, np.ndarray] = {}
    records: dict[str, list[Mapping[str, Any]]] = {}
    receipts: list[Mapping[str, Any]] = []
    for root in roots:
        arrays, results, complete = _load_artifact(root)
        current = (arrays["labels"], arrays["folds"], arrays["state_ids"])
        identity = _same_identity(identity, current)
        _merge_m2_arrays(merged, arrays, seeds)
        _merge_records(records, results, seeds)
        receipts.append(_root_receipt(root, complete))
    expected = {
        f"{variant}__seed_{seed}__{kind}" for variant in VARIANTS
        for seed in seeds for kind in ("raw", "calibrated", "auxiliary_mse_rows")
    }
    if set(merged) != expected or any(not np.isfinite(value).all() for value in merged.values()):
        raise AdvantageM2AnalysisError("M2 OOFのseed/fold coverageが未完了です")
    assert identity is not None
    return MergedM2(*identity, merged, {key: tuple(value) for key, value in records.items()},
                    tuple(receipts))


def _same_identity(
    expected: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    current: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if expected is None:
        return current
    if any(not np.array_equal(left, right) for left, right in zip(expected, current, strict=True)):
        raise AdvantageM2AnalysisError("M2 root間でlabels/folds/state_idsが不一致です")
    return expected


def _merge_m2_arrays(
    merged: dict[str, np.ndarray], arrays: Mapping[str, np.ndarray], seeds: Sequence[int],
) -> None:
    for variant in VARIANTS:
        for seed in seeds:
            for kind in ("raw", "calibrated", "auxiliary_mse_rows"):
                name = f"{variant}__seed_{seed}__{kind}"
                if name in arrays:
                    merged[name] = _merge_vector(merged.get(name), arrays[name], name)


def _merge_vector(current: np.ndarray | None, new: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(new, dtype=np.float32)
    if current is None:
        return np.array(value, copy=True)
    if current.shape != value.shape:
        raise AdvantageM2AnalysisError(f"OOF shapeが不一致です: {name}")
    incoming, existing = np.isfinite(value), np.isfinite(current)
    if np.any(incoming & existing):
        raise AdvantageM2AnalysisError(f"OOF行が重複しています: {name}")
    output = np.array(current, copy=True)
    output[incoming] = value[incoming]
    return output


def _merge_records(
    output: dict[str, list[Mapping[str, Any]]], report: Mapping[str, Any],
    seeds: Sequence[int],
) -> None:
    results = report.get("results", {})
    for variant in VARIANTS:
        for seed in seeds:
            name = f"{variant}__seed_{seed}"
            if name in results:
                output.setdefault(name, []).extend(results[name].get("records", ()))


def _root_receipt(root: Path, complete: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root": str(root.resolve()), "complete_sha256": base.file_sha256(root / "COMPLETE"),
        "results_sha256": str(complete["results_sha256"]),
        "predictions_sha256": str(complete["predictions_sha256"]),
    }


def merge_m1_roots(
    roots: Sequence[Path], seeds: Sequence[int], labels: np.ndarray, folds: np.ndarray,
) -> tuple[dict[int, np.ndarray], tuple[Mapping[str, Any], ...]]:
    merged: dict[int, np.ndarray] = {}
    receipts: list[Mapping[str, Any]] = []
    for root in roots:
        arrays, _results, complete = _load_artifact(root)
        if not np.array_equal(arrays["labels"], labels) or not np.array_equal(arrays["folds"], folds):
            raise AdvantageM2AnalysisError("M1/M2 labelsまたはfoldsが不一致です")
        for seed in seeds:
            name = f"{M1_VARIANT}__seed_{seed}__calibrated"
            if name in arrays:
                merged[seed] = _merge_vector(merged.get(seed), arrays[name], name)
        receipts.append(_root_receipt(root, complete))
    if set(merged) != set(seeds) or any(not np.isfinite(value).all() for value in merged.values()):
        raise AdvantageM2AnalysisError("M1 3seed OOFが揃っていません")
    return merged, tuple(receipts)


def _ensemble(
    predictions: Mapping[str, np.ndarray], variant: str, seeds: Sequence[int], kind: str,
) -> np.ndarray:
    rows = [predictions[f"{variant}__seed_{seed}__{kind}"] for seed in seeds]
    return np.mean(np.stack(rows).astype(np.float64), axis=0)


def _winner_summary(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    winner = np.where(labels == 1.0, probability, 1.0 - probability)
    return {
        "mean_winner_probability": float(winner.mean()),
        "winner_below_10_percent_count": int(np.count_nonzero(winner < 0.10)),
        "winner_below_20_percent_count": int(np.count_nonzero(winner < 0.20)),
        "wrong_side_fraction": float(np.mean(winner < 0.50)),
    }


def _fold_rows(
    samples: m2_train.M2SamplesV1, baseline: np.ndarray, candidate: np.ndarray,
) -> list[dict[str, float | int]]:
    output = []
    for fold in range(1, 7):
        mask = samples.folds == fold
        subset = samples.subset(mask)
        base_loss = legacy.metrics(subset, baseline[mask])["game_equal_log_loss"]
        candidate_loss = legacy.metrics(subset, candidate[mask])["game_equal_log_loss"]
        output.append({
            "fold": fold, "baseline_log_loss": base_loss,
            "candidate_log_loss": candidate_loss,
            "candidate_minus_baseline_log_loss": candidate_loss - base_loss,
        })
    return output


def _source_deltas(
    samples: m2_train.M2SamplesV1, baseline: np.ndarray, candidate: np.ndarray,
) -> np.ndarray:
    output = []
    for source in np.unique(samples.source_groups):
        mask = samples.source_groups == source
        subset = samples.subset(mask)
        base_loss = legacy.metrics(subset, baseline[mask])["game_equal_log_loss"]
        candidate_loss = legacy.metrics(subset, candidate[mask])["game_equal_log_loss"]
        output.append(candidate_loss - base_loss)
    return np.asarray(output, dtype=np.float64)


def _bootstrap_ci(deltas: np.ndarray, seed: int = 20260905) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(BOOTSTRAP_REPLICATES, len(deltas)))
    means = deltas[draws].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _comparison(
    samples: m2_train.M2SamplesV1, baseline: np.ndarray, candidate: np.ndarray,
) -> dict[str, Any]:
    base_metrics, candidate_metrics = legacy.metrics(samples, baseline), legacy.metrics(samples, candidate)
    folds = _fold_rows(samples, baseline, candidate)
    sources = _source_deltas(samples, baseline, candidate)
    return {
        "baseline": base_metrics, "candidate": candidate_metrics,
        "candidate_minus_baseline_log_loss": (
            candidate_metrics["game_equal_log_loss"] - base_metrics["game_equal_log_loss"]
        ),
        "candidate_minus_baseline_auc": candidate_metrics["auc"] - base_metrics["auc"],
        "fold_rows": folds,
        "fold_win_count": sum(row["candidate_minus_baseline_log_loss"] < 0 for row in folds),
        "fold_tie_count": sum(row["candidate_minus_baseline_log_loss"] == 0 for row in folds),
        "source_win_count": int(np.count_nonzero(sources < 0)),
        "source_count": len(sources), "source_mean_delta_log_loss": float(sources.mean()),
        "source_bootstrap_95_ci": _bootstrap_ci(sources),
        "baseline_severe": _winner_summary(samples.labels, baseline),
        "candidate_severe": _winner_summary(samples.labels, candidate),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    """3seed ensembleを作り、M1とaux-offに対するpaired監査を保存する。"""

    if args.output_root.exists():
        raise AdvantageM2AnalysisError(f"出力先は新規必須です: {args.output_root}")
    samples, dataset_manifest = m2_train.load_m2_dataset(args.dataset_root)
    m2 = merge_m2_roots(args.m2_roots, args.seeds)
    _validate_dataset_identity(samples, m2)
    m1, m1_receipts = merge_m1_roots(
        args.m1_roots, args.seeds, m2.labels, m2.folds,
    )
    baseline = np.mean(np.stack([m1[seed] for seed in args.seeds]), axis=0)
    off = _ensemble(m2.predictions, "m2_aux_off", args.seeds, "calibrated")
    on = _ensemble(m2.predictions, "m2_aux_on_0p3", args.seeds, "calibrated")
    report = _report(args, samples, dataset_manifest, m2, m1_receipts, baseline, off, on)
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = base._write_json_exclusive(args.output_root / "analysis.json", report)
    base._write_json_exclusive(args.output_root / "COMPLETE", {
        "format_version": COMPLETE_VERSION, "analysis_sha256": base.file_sha256(path),
    })
    return report


def _validate_dataset_identity(samples: m2_train.M2SamplesV1, merged: MergedM2) -> None:
    values = (
        np.array_equal(samples.labels, merged.labels),
        np.array_equal(samples.folds, merged.folds),
        np.array_equal(samples.state_ids, merged.state_ids),
    )
    if not all(values):
        raise AdvantageM2AnalysisError("M2 datasetとOOFの行同定が一致しません")


def _report(
    args: argparse.Namespace, samples: m2_train.M2SamplesV1,
    dataset_manifest: Mapping[str, Any], merged: MergedM2,
    m1_receipts: Sequence[Mapping[str, Any]], baseline: np.ndarray,
    off: np.ndarray, on: np.ndarray,
) -> dict[str, Any]:
    auxiliary = {
        variant: m2_train._weighted_auxiliary_mse(
            samples, _ensemble(merged.predictions, variant, args.seeds, "auxiliary_mse_rows"),
        ) for variant in VARIANTS
    }
    return {
        "format_version": ANALYSIS_VERSION, "not_production": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_sha256": dataset_manifest["dataset"]["sha256"],
        "state_count": len(samples.labels), "game_count": len(np.unique(samples.game_keys)),
        "source_count": len(np.unique(samples.source_groups)), "seeds": list(args.seeds),
        "ensemble_policy": "equal_arithmetic_mean_of_calibrated_fixed_seed_oof",
        "comparisons": {
            "m2_aux_off_vs_m1": _comparison(samples, baseline, off),
            "m2_aux_on_vs_m1": _comparison(samples, baseline, on),
            "m2_aux_on_vs_aux_off": _comparison(samples, off, on),
        },
        "auxiliary_mse": auxiliary,
        "fallback_counts": {
            name: sum(bool(row["fallback_to_m1"]) for row in rows)
            for name, rows in merged.records.items()
        },
        "m2_artifacts": list(merged.receipts), "m1_artifacts": list(m1_receipts),
        "formal100_used": False, "hidden_reserve_used": False,
        "production_config_changed": False,
        "production_config_sha256": base.file_sha256(Path("src/production_config.py")),
        "state_id_ordered_sha256": builder._array_sha256(samples.state_ids),
        "analysis_code_sha256": base.file_sha256(Path(__file__)),
    }


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("整数のカンマ区切りが必要です") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--m2-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--m1-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    args = parser.parse_args(argv)
    if tuple(args.seeds) != DEFAULT_SEEDS or args.output_root.exists():
        parser.error("固定3seedと新規output-rootが必要です")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = analyze(parse_args(argv))
    print(json.dumps(report["comparisons"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
