"""M2 V4 bounded auxiliaryの3seed OOFをM1・nullと独立監査する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import analyze_advantage_m1_fixed_oof_v1 as strata_analysis
from scripts import analyze_advantage_m2_two_stage_v3 as old_analysis
from scripts import build_advantage_m2_dataset_v1 as receipt
from scripts import train_advantage_m0_current_cnn_v1 as train_base
from scripts import train_advantage_m1_causal_ledger_v1 as metrics
from scripts import train_advantage_m2_anchored_auxiliary_v2 as anchors
from scripts import train_advantage_m2_bounded_auxiliary_v4 as trainer
from scripts import train_advantage_m2_auxiliary_v1 as m2_data
from src.advantage_m2_bounded_auxiliary_v4 import MAX_SHRINK_FRACTION


FORMAT_VERSION = "advantage-m2-bounded-auxiliary-analysis/v4-exact-anchor"
COMPLETE_VERSION = "advantage-m2-bounded-auxiliary-analysis-complete/v4-exact-anchor"
DEFAULT_TRAINING_ROOT = Path(
    "data/verify/advantage_m2_bounded_auxiliary_oof_46v_2026-09-06_v4_3seed_allfolds"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/verify/advantage_m2_bounded_auxiliary_audit_46v_2026-09-06_v4_exact_anchor_v2"
)
THRESHOLDS = (0.10, 0.20)
BOOTSTRAP_REPEATS = 10_000
PROBABILITY_EPSILON = 1e-7
EXAMPLE_LIMIT = 20


class AdvantageM2BoundedAnalysisError(RuntimeError):
    """V4成果物、同一行、監査保存契約の違反。"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM2BoundedAnalysisError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise AdvantageM2BoundedAnalysisError(f"JSON objectではありません: {path}")
    return value


def load_training_artifact(
    root: Path, samples: m2_data.M2SamplesV1,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """COMPLETEと全model hashを検証してV4 OOFを読む。"""

    complete, report = _read_json(root / "COMPLETE"), _read_json(root / "results.json")
    prediction_path = root / "oof_predictions.npz"
    checks = (
        complete.get("format_version") == trainer.COMPLETE_VERSION,
        complete.get("results_sha256") == receipt.file_sha256(root / "results.json"),
        complete.get("predictions_sha256") == receipt.file_sha256(prediction_path),
        complete.get("production_config_changed") is False,
    )
    if not all(checks):
        raise AdvantageM2BoundedAnalysisError("V4 COMPLETE receiptが不正です")
    with np.load(prediction_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    _validate_identity(samples, arrays)
    _validate_prediction_coverage(arrays)
    _validate_model_receipts(root, report)
    return arrays, report, complete


def _validate_identity(
    samples: m2_data.M2SamplesV1, arrays: Mapping[str, np.ndarray],
) -> None:
    pairs = (
        ("labels", samples.labels), ("weights", samples.weights),
        ("folds", samples.folds), ("state_ids", samples.state_ids),
        ("source_groups", samples.source_groups), ("game_keys", samples.game_keys),
    )
    if any(name not in arrays or not np.array_equal(arrays[name], value) for name, value in pairs):
        raise AdvantageM2BoundedAnalysisError("V4 predictionとdatasetの行同定が不一致です")


def _validate_prediction_coverage(arrays: Mapping[str, np.ndarray]) -> None:
    expected = {
        f"m2_bounded_{variant}__seed_{seed}__{kind}"
        for variant in trainer.AUXILIARY_VARIANTS for seed in trainer.DEFAULT_SEEDS
        for kind in ("raw", "calibrated", "auxiliary_mse_rows", "shrink_fraction")
    }
    missing = sorted(expected - set(arrays))
    if missing or any(not np.isfinite(arrays[name]).all() for name in expected):
        raise AdvantageM2BoundedAnalysisError(f"V4 OOF coverageが未完了です: {missing[:3]}")


def _validate_model_receipts(root: Path, report: Mapping[str, Any]) -> None:
    for value in report.get("results", {}).values():
        for record in value.get("records", ()):
            path = root / str(record.get("model", ""))
            if not path.is_file() or receipt.file_sha256(path) != record.get("model_sha256"):
                raise AdvantageM2BoundedAnalysisError(f"V4 model hashが不一致です: {path}")


def load_m1_predictions(
    roots: Sequence[Path], samples: m2_data.M2SamplesV1,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    """分割M1成果物から3seed raw/calibratedを全行結合する。"""

    output: dict[str, np.ndarray] = {}
    receipts: list[dict[str, str]] = []
    for root in roots:
        arrays, _report, complete = old_analysis._load_artifact(root)
        if not np.array_equal(arrays["labels"], samples.labels):
            raise AdvantageM2BoundedAnalysisError("M1/M2 labelが不一致です")
        if not np.array_equal(arrays["folds"], samples.folds):
            raise AdvantageM2BoundedAnalysisError("M1/M2 foldが不一致です")
        _merge_m1_root(output, arrays)
        receipts.append({
            "root": str(root.resolve()), "complete_sha256": receipt.file_sha256(root / "COMPLETE"),
            "predictions_sha256": str(complete["predictions_sha256"]),
        })
    expected = {f"seed_{seed}__{kind}" for seed in trainer.DEFAULT_SEEDS
                for kind in ("raw", "calibrated")}
    if set(output) != expected or any(not np.isfinite(value).all() for value in output.values()):
        raise AdvantageM2BoundedAnalysisError("M1 3seed OOFが揃っていません")
    return output, receipts


def _merge_m1_root(output: dict[str, np.ndarray], arrays: Mapping[str, np.ndarray]) -> None:
    for seed in trainer.DEFAULT_SEEDS:
        for kind in ("raw", "calibrated"):
            source = f"m1_zero_values_and_masks__seed_{seed}__{kind}"
            if source not in arrays:
                continue
            target = f"seed_{seed}__{kind}"
            output[target] = old_analysis._merge_vector(
                output.get(target), arrays[source], target,
            )


def reconstruct_exact_m1(
    roots: Sequence[Path], samples: m2_data.M2SamplesV1,
    training_report: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """V4と同じbatch/deviceでfold別M1 anchorを再推論する。"""

    plan = training_report.get("plan", {})
    if plan.get("seeds") != list(trainer.DEFAULT_SEEDS):
        raise AdvantageM2BoundedAnalysisError("training seed契約が不正です")
    if plan.get("folds") != list(trainer.DEFAULT_FOLDS):
        raise AdvantageM2BoundedAnalysisError("training fold契約が不正です")
    registry, _receipts = anchors.load_anchor_registry(roots)
    anchors._required_anchors(registry, trainer.DEFAULT_SEEDS, trainer.DEFAULT_FOLDS)
    device = train_base._device(str(plan.get("device", "cuda")))
    loader_args = SimpleNamespace(batch_size=int(plan["batch_size"]))
    output: dict[str, np.ndarray] = {}
    for seed in trainer.DEFAULT_SEEDS:
        raw = np.full(len(samples.labels), np.nan, dtype=np.float32)
        calibrated = np.full(len(samples.labels), np.nan, dtype=np.float32)
        for fold in trainer.DEFAULT_FOLDS:
            mask = samples.folds == fold
            reference = registry[(seed, fold)]
            anchor = anchors._load_anchor(reference, device)
            values = anchors._predict_anchor(
                anchor, samples.subset(mask), loader_args, device,
            )
            raw[mask] = values
            calibrated[mask] = metrics.calibrate(values, reference.slope)
            del anchor
        output[f"seed_{seed}__raw"] = raw
        output[f"seed_{seed}__calibrated"] = calibrated
    if any(not np.isfinite(value).all() for value in output.values()):
        raise AdvantageM2BoundedAnalysisError("exact M1再構成に欠損があります")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _official_anchor_drift(
    official: Mapping[str, np.ndarray], exact: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for seed in trainer.DEFAULT_SEEDS:
        output[str(seed)] = {}
        for kind in ("raw", "calibrated"):
            name = f"seed_{seed}__{kind}"
            delta = np.abs(official[name].astype(np.float64) - exact[name].astype(np.float64))
            output[str(seed)][kind] = {
                "bit_mismatch_count": int(np.count_nonzero(delta)),
                "max_absolute_probability_delta": float(delta.max(initial=0.0)),
                "mean_absolute_probability_delta": float(delta.mean()),
            }
    return output


def _ensemble(arrays: Mapping[str, np.ndarray], variant: str, kind: str) -> np.ndarray:
    values = [arrays[f"m2_bounded_{variant}__seed_{seed}__{kind}"]
              for seed in trainer.DEFAULT_SEEDS]
    return np.mean(np.stack(values).astype(np.float64), axis=0)


def _m1_ensemble(arrays: Mapping[str, np.ndarray], kind: str) -> np.ndarray:
    return np.mean(np.stack([
        arrays[f"seed_{seed}__{kind}"] for seed in trainer.DEFAULT_SEEDS
    ]).astype(np.float64), axis=0)


def _group_rows(
    samples: m2_data.M2SamplesV1, baseline: np.ndarray, candidate: np.ndarray,
    groups: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        subset = samples.subset(mask)
        old, new = metrics.metrics(subset, baseline[mask]), metrics.metrics(subset, candidate[mask])
        rows.append({
            "group": str(group), "row_count": int(mask.sum()),
            "baseline_log_loss": old["game_equal_log_loss"],
            "candidate_log_loss": new["game_equal_log_loss"],
            "delta_log_loss": new["game_equal_log_loss"] - old["game_equal_log_loss"],
        })
    return rows


def _bootstrap(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, float | int]:
    deltas = np.asarray([row["delta_log_loss"] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(BOOTSTRAP_REPEATS, len(deltas)))
    means = deltas[draws].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return {"repeat_count": BOOTSTRAP_REPEATS, "mean": float(deltas.mean()),
            "ci95_low": float(low), "ci95_high": float(high)}


def _metric_delta(
    samples: m2_data.M2SamplesV1, baseline: np.ndarray, candidate: np.ndarray,
) -> dict[str, Any]:
    old, new = metrics.metrics(samples, baseline), metrics.metrics(samples, candidate)
    return {
        "baseline": old, "candidate": new,
        "delta": {
            "log_loss": new["game_equal_log_loss"] - old["game_equal_log_loss"],
            "brier": new["game_equal_brier"] - old["game_equal_brier"],
            "auc": new["auc"] - old["auc"],
        },
    }


def _extreme_summary(
    samples: m2_data.M2SamplesV1, baseline: np.ndarray,
    candidate: np.ndarray, available_ms: np.ndarray,
) -> dict[str, Any]:
    old = np.where(samples.labels == 1, baseline, 1.0 - baseline)
    new = np.where(samples.labels == 1, candidate, 1.0 - candidate)
    output: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        old_bad, new_bad = old < threshold, new < threshold
        added, resolved = new_bad & ~old_bad, old_bad & ~new_bad
        output[str(threshold)] = {
            "baseline_count": int(old_bad.sum()), "candidate_count": int(new_bad.sum()),
            "new_count": int(added.sum()), "resolved_count": int(resolved.sum()),
            "common_count": int((old_bad & new_bad).sum()),
            "candidate_nonincrease": bool(new_bad.sum() <= old_bad.sum()),
            "new_worst_examples": _examples(samples, new, added, available_ms),
        }
    return output


def _examples(
    samples: m2_data.M2SamplesV1, winner: np.ndarray,
    mask: np.ndarray, available_ms: np.ndarray,
) -> list[dict[str, Any]]:
    indices = np.flatnonzero(mask)
    indices = indices[np.argsort(winner[indices])[:EXAMPLE_LIMIT]]
    return [{
        "state_id": str(samples.state_ids[index]),
        "source": str(samples.source_groups[index]),
        "game": str(samples.game_keys[index]), "fold": int(samples.folds[index]),
        "available_ms": int(available_ms[index]),
        "winner_probability": float(winner[index]),
    } for index in indices]


def _comparison(
    samples: m2_data.M2SamplesV1, baseline: np.ndarray, candidate: np.ndarray,
    strata: Mapping[str, np.ndarray], available_ms: np.ndarray, seed: int,
) -> dict[str, Any]:
    source_rows = _group_rows(samples, baseline, candidate, samples.source_groups)
    fold_rows = _group_rows(samples, baseline, candidate, samples.folds)
    return {
        "overall": _metric_delta(samples, baseline, candidate),
        "strata": {name: _metric_delta(samples.subset(mask), baseline[mask], candidate[mask])
                   for name, mask in strata.items() if mask.any()},
        "source_rows": source_rows,
        "source_win_count": sum(row["delta_log_loss"] < 0 for row in source_rows),
        "source_bootstrap": _bootstrap(source_rows, seed),
        "fold_rows": fold_rows,
        "fold_win_count": sum(row["delta_log_loss"] < 0 for row in fold_rows),
        "extreme": _extreme_summary(samples, baseline, candidate, available_ms),
    }


def _load_strata(
    samples: m2_data.M2SamplesV1, manifest: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    root = Path(str(manifest["parent_m2_root"]))
    with np.load(root / "dataset.npz", allow_pickle=False) as data:
        available_ms = np.asarray(data["available_ms"], dtype=np.int64)
        ledger_usable = np.asarray(data["ledger_usable"], dtype=np.bool_)
    raw = {"boards": samples.boards, "ledger_values": samples.ledger_values,
           "ledger_usable": ledger_usable, "available_ms": available_ms}
    return strata_analysis.build_strata(samples, raw), available_ms, ledger_usable


def _bounded_contract(
    training: Mapping[str, np.ndarray], m1: Mapping[str, np.ndarray],
    ledger_usable: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in trainer.AUXILIARY_VARIANTS:
        rows = []
        for seed in trainer.DEFAULT_SEEDS:
            candidate = training[f"m2_bounded_{variant}__seed_{seed}__raw"].astype(np.float64)
            anchor = m1[f"seed_{seed}__raw"].astype(np.float64)
            old_logit, new_logit = _logit(anchor), _logit(candidate)
            rows.append({
                "seed": seed,
                "magnitude_expansion_count": int(np.count_nonzero(
                    np.abs(new_logit) > np.abs(old_logit) + 1e-6
                )),
                "sign_flip_count": int(np.count_nonzero(old_logit * new_logit < 0.0)),
                "unsupported_bit_mismatch_count": int(np.count_nonzero(
                    candidate[~ledger_usable] != anchor[~ledger_usable]
                )),
                "max_saved_shrink": float(training[
                    f"m2_bounded_{variant}__seed_{seed}__shrink_fraction"
                ].max(initial=0.0)),
            })
        output[variant] = rows
    return output


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.log(value / (1.0 - value))


def _selection_audit(
    training: Mapping[str, np.ndarray], m1: Mapping[str, np.ndarray],
    report: Mapping[str, Any], samples: m2_data.M2SamplesV1,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, result in report.get("results", {}).items():
        seed = int(name.rsplit("_", 1)[-1])
        mismatches, epochs = 0, []
        for record in result.get("records", ()):
            fold, epoch = int(record["eval_fold"]), int(record["best_epoch"])
            epochs.append(epoch)
            if bool(record["fallback_to_m1"]):
                mask = samples.folds == fold
                candidate = training[f"{name}__raw"][mask]
                mismatches += int(np.count_nonzero(candidate != m1[f"seed_{seed}__raw"][mask]))
        output[name] = {"selected_epochs": epochs, "epoch0_count": epochs.count(0),
                        "fallback_prediction_bit_mismatch_count": mismatches}
    return output


def _seed_stability(
    samples: m2_data.M2SamplesV1, training: Mapping[str, np.ndarray],
    m1: Mapping[str, np.ndarray], strata: Mapping[str, np.ndarray],
    available_ms: np.ndarray, seed_base: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for seed in trainer.DEFAULT_SEEDS:
        baseline = m1[f"seed_{seed}__calibrated"]
        output[str(seed)] = {}
        for index, variant in enumerate(trainer.AUXILIARY_VARIANTS):
            candidate = training[f"m2_bounded_{variant}__seed_{seed}__calibrated"]
            compared = _comparison(
                samples, baseline, candidate, strata, available_ms,
                seed_base + seed + index,
            )
            output[str(seed)][variant] = {
                "overall": compared["overall"],
                "fold_win_count": compared["fold_win_count"],
                "source_win_count": compared["source_win_count"],
                "source_bootstrap": compared["source_bootstrap"],
                "extreme": compared["extreme"],
            }
    return output


def _gate(comparison: Mapping[str, Any], contract: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    delta, extreme = comparison["overall"]["delta"], comparison["extreme"]
    return {
        "log_loss_improves": delta["log_loss"] < 0.0,
        "brier_nonworse": delta["brier"] <= 0.0,
        "at_least_five_of_six_folds": comparison["fold_win_count"] >= 5,
        "source_bootstrap_ci_high_below_zero": comparison["source_bootstrap"]["ci95_high"] < 0,
        "no_new_winner_below_10_percent": extreme["0.1"]["new_count"] == 0,
        "winner_below_20_percent_nonincrease": extreme["0.2"]["candidate_nonincrease"],
        "nonexpanding_all_seeds": all(row["magnitude_expansion_count"] == 0 for row in contract),
        "sign_preserving_all_seeds": all(row["sign_flip_count"] == 0 for row in contract),
        "unsupported_exact_all_seeds": all(
            row["unsupported_bit_mismatch_count"] == 0 for row in contract
        ),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    """3seed平均をM1、aux-off、matched-randomと比較して排他保存する。"""

    if args.output_root.exists():
        raise AdvantageM2BoundedAnalysisError(f"出力先は新規必須です: {args.output_root}")
    samples, manifest = trainer.load_family_dataset(args.dataset_root)
    training, training_report, complete = load_training_artifact(args.training_root, samples)
    official_m1, m1_receipts = load_m1_predictions(args.m1_roots, samples)
    exact_m1 = reconstruct_exact_m1(args.m1_roots, samples, training_report)
    strata, available_ms, ledger_usable = _load_strata(samples, manifest)
    baseline = _m1_ensemble(exact_m1, "calibrated")
    ensembles = {name: _ensemble(training, name, "calibrated")
                 for name in trainer.AUXILIARY_VARIANTS}
    comparisons = {name: _comparison(
        samples, baseline, value, strata, available_ms, args.bootstrap_seed + index,
    ) for index, (name, value) in enumerate(ensembles.items())}
    versus_random = {name: _comparison(
        samples, ensembles["matched_random"], value, strata, available_ms,
        args.bootstrap_seed + 100 + index,
    ) for index, (name, value) in enumerate(ensembles.items()) if name != "matched_random"}
    versus_off = {name: _comparison(
        samples, ensembles["aux_off"], value, strata, available_ms,
        args.bootstrap_seed + 200 + index,
    ) for index, (name, value) in enumerate(ensembles.items()) if name != "aux_off"}
    contract = _bounded_contract(training, exact_m1, ledger_usable)
    report = _report(
        args, samples, manifest, complete, m1_receipts, comparisons,
        versus_random, versus_off, contract, training, exact_m1, training_report,
        strata, available_ms, _official_anchor_drift(official_m1, exact_m1),
    )
    return _write(args.output_root, report)


def _report(
    args: argparse.Namespace, samples: m2_data.M2SamplesV1,
    manifest: Mapping[str, Any], complete: Mapping[str, Any],
    m1_receipts: Sequence[Mapping[str, Any]], comparisons: Mapping[str, Any],
    versus_random: Mapping[str, Any], versus_off: Mapping[str, Any],
    contract: Mapping[str, Sequence[Mapping[str, Any]]],
    training: Mapping[str, np.ndarray], m1: Mapping[str, np.ndarray],
    training_report: Mapping[str, Any], strata: Mapping[str, np.ndarray],
    available_ms: np.ndarray, official_anchor_drift: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION, "not_production": True, "diagnostic_only": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_sha256": manifest["dataset"]["sha256"],
        "state_count": len(samples.labels), "source_count": len(np.unique(samples.source_groups)),
        "game_count": len(np.unique(samples.game_keys)),
        "seeds": list(trainer.DEFAULT_SEEDS), "folds": list(trainer.DEFAULT_FOLDS),
        "ensemble_policy": "equal_arithmetic_mean_of_calibrated_fixed_seed_oof",
        "comparisons_vs_m1": dict(comparisons),
        "comparisons_vs_matched_random": dict(versus_random),
        "comparisons_vs_aux_off": dict(versus_off),
        "seed_stability": _seed_stability(
            samples, training, m1, strata, available_ms, 10_000,
        ),
        "bounded_contract": dict(contract),
        "official_m1_vs_exact_reconstructed_anchor": dict(official_anchor_drift),
        "selection_audit": _selection_audit(training, m1, training_report, samples),
        "gate_summary": {name: _gate(value, contract[name])
                         for name, value in comparisons.items()},
        "training_complete_sha256": receipt.file_sha256(args.training_root / "COMPLETE"),
        "training_predictions_sha256": str(complete["predictions_sha256"]),
        "m1_artifacts": list(m1_receipts), "max_shrink_fraction": MAX_SHRINK_FRACTION,
        "formal100_used": False, "hidden_reserve_used": False,
        "attack_difference_ten_percent_correction": False,
        "production_config_changed": False,
        "production_config_sha256": receipt.file_sha256(Path("src/production_config.py")),
        "analysis_code_sha256": receipt.file_sha256(Path(__file__)),
    }


def _write(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    path = train_base_write(root / "analysis.json", report)
    train_base_write(root / "COMPLETE", {
        "format_version": COMPLETE_VERSION, "not_production": True,
        "analysis_sha256": receipt.file_sha256(path),
    })
    return dict(report)


def train_base_write(path: Path, value: Mapping[str, Any]) -> Path:
    if path.exists():
        raise AdvantageM2BoundedAnalysisError(f"出力は新規必須です: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=trainer.DEFAULT_DATASET_ROOT)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--m1-roots", type=Path, nargs="+", default=list(anchors.DEFAULT_ANCHOR_ROOTS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        parser.error("output-rootは新規必須です")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = analyze(parse_args(argv))
    print(json.dumps(report["gate_summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
