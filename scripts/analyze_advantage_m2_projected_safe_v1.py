"""M2 projected-safe-shrink OOFを事前登録gateで独立監査する。"""

from __future__ import annotations

from scripts.production_dependency_contract import (
    dependency_receipt, production_compatible, saved_dependency_compatible,
)

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts import analyze_advantage_m1_fixed_oof_v1 as strata_analysis
from scripts import analyze_advantage_m2_bounded_auxiliary_v4 as common_analysis
from scripts import build_advantage_m2_dataset_v1 as receipt
from scripts import build_advantage_m2_projected_safe_dataset_v1 as dataset_builder
from scripts import train_advantage_m1_zero_counterfactual_v3 as m1_trainer
from scripts import train_advantage_m2_projected_safe_v1 as trainer
from src.projected_set_bounded_shrink_cnn_v2 import (
    MAX_PROBABILITY_SHRINK,
    ProjectedSetBoundedShrinkCNNV2,
)
from src.projected_set_residual_cnn_v1 import SCALAR_SWAP_ORDER


FORMAT_VERSION = "advantage-m2-projected-safe-analysis/v1"
COMPLETE_VERSION = "advantage-m2-projected-safe-analysis-complete/v1"
DEFAULT_PARENT_ROOT = Path(
    "data/verify/advantage_m1_canonical_dataset_46v_2026-09-05_v3_finalized_primary6"
)
DEFAULT_TRAINING_ROOT = trainer.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "data/verify/advantage_m2_projected_safe_audit_46v_2026-09-06_v1"
)
REQUIRED_STRATA = (
    "incoming_at_least_30",
    "large_nonfill_death_cell_open",
    "ledger_usable",
    "ledger_unusable",
)
SYMMETRY_TOLERANCE = 1e-6
REPRODUCTION_TOLERANCE = 1e-12
SAFETY_TOLERANCE = trainer.SAFETY_ABS_TOLERANCE
BOOTSTRAP_SEED = 20260906
REPO_ROOT = Path(__file__).resolve().parents[1]


class AdvantageM2ProjectedSafeAnalysisError(RuntimeError):
    """学習成果物または事前登録監査契約の違反。"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM2ProjectedSafeAnalysisError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise AdvantageM2ProjectedSafeAnalysisError(f"JSON objectではありません: {path}")
    return value


def _validate_training_receipts(
    root: Path, plan: Mapping[str, Any], results: Mapping[str, Any],
    complete: Mapping[str, Any],
) -> None:
    prediction_path = root / "oof_predictions.npz"
    hashes = plan.get("code_sha256", {})
    checks = (
        plan.get("format_version") == trainer.PLAN_VERSION,
        results.get("format_version") == trainer.FORMAT_VERSION,
        complete.get("format_version") == trainer.COMPLETE_VERSION,
        results.get("plan") == plan,
        complete.get("plan_sha256") == receipt.file_sha256(root / "PLAN.json"),
        complete.get("results_sha256") == receipt.file_sha256(root / "results.json"),
        complete.get("predictions_sha256") == receipt.file_sha256(prediction_path),
        complete.get("production_config_changed") is False,
        hashes.get("trainer") == receipt.file_sha256(Path(trainer.__file__)),
        hashes.get("model") == receipt.file_sha256(
            REPO_ROOT / "src/projected_set_bounded_shrink_cnn_v2.py"
        ),
        hashes.get("dataset_builder") == receipt.file_sha256(Path(dataset_builder.__file__)),
        saved_dependency_compatible(hashes.get("production_config"), plan.get("production_dependency_contract")),
        production_compatible(REPO_ROOT),
    )
    if not all(checks):
        raise AdvantageM2ProjectedSafeAnalysisError("学習receiptまたはcode hashが不正です")


def _validate_plan(plan: Mapping[str, Any]) -> None:
    checks = (
        tuple(plan.get("seeds", ())) == tuple(trainer.TRAINING_SEEDS),
        tuple(plan.get("folds", ())) == trainer.DEFAULT_FOLDS,
        plan.get("epochs") == trainer.EPOCH_COUNT,
        plan.get("max_probability_shrink") == MAX_PROBABILITY_SHRINK,
        plan.get("formal100_used") is False,
        plan.get("hidden_reserve_used") is False,
        plan.get("production_config_changed") is False,
    )
    if not all(checks):
        raise AdvantageM2ProjectedSafeAnalysisError("固定seed/fold/epoch/隔離planと一致しません")


def _load_prediction_arrays(root: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(root / "oof_predictions.npz", allow_pickle=False) as data:
            return {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as error:
        raise AdvantageM2ProjectedSafeAnalysisError("OOF predictionを読めません") from error


def _validate_prediction_identity(
    arrays: Mapping[str, np.ndarray], samples: trainer.ProjectedSafeSamplesV1,
) -> None:
    pairs = (
        ("labels", samples.full["labels"]), ("weights", samples.full["weights"]),
        ("folds", samples.full["folds"]), ("state_ids", samples.full["state_ids"]),
        ("source_groups", samples.full["source_groups"]),
        ("game_keys", samples.full["game_keys"]),
        ("baseline_probability", samples.full["baseline_probability"]),
        ("supported_global_indices", samples.global_indices),
    )
    if any(name not in arrays or not np.array_equal(arrays[name], value)
           for name, value in pairs):
        raise AdvantageM2ProjectedSafeAnalysisError("OOFとprojected datasetの行同一性が不正です")
    expected = {f"candidate__seed_{seed}" for seed in trainer.TRAINING_SEEDS}
    if not expected.issubset(arrays):
        raise AdvantageM2ProjectedSafeAnalysisError("固定3seed predictionが揃っていません")


def _validate_parent_identity(samples: Any, projected: trainer.ProjectedSafeSamplesV1) -> None:
    pairs = (
        (samples.labels, projected.full["labels"]),
        (samples.weights, projected.full["weights"]),
        (samples.folds, projected.full["folds"]),
        (samples.state_ids, projected.full["state_ids"]),
        (samples.source_groups, projected.full["source_groups"]),
        (samples.game_keys, projected.full["game_keys"]),
    )
    if any(not np.array_equal(left, right) for left, right in pairs):
        raise AdvantageM2ProjectedSafeAnalysisError("M1正本とprojected datasetの行が不一致です")


def _validate_model_receipts(
    root: Path, results: Mapping[str, Any], projected: trainer.ProjectedSafeSamplesV1,
) -> list[dict[str, Any]]:
    records = results.get("fold_records")
    if not isinstance(records, list):
        raise AdvantageM2ProjectedSafeAnalysisError("fold receiptがlistではありません")
    expected = {(seed, fold) for seed in trainer.TRAINING_SEEDS for fold in trainer.DEFAULT_FOLDS}
    observed: set[tuple[int, int]] = set()
    for record in records:
        key = (int(record["seed"]), int(record["eval_fold"]))
        path = root / f"m2_projected_safe__seed_{key[0]}__fold_{key[1]}.pt"
        count = int(np.count_nonzero(projected.supported_folds == key[1]))
        valid = path.is_file() and record.get("model_sha256") == receipt.file_sha256(path)
        valid &= record.get("eval_supported_count") == count
        if not valid or key in observed:
            raise AdvantageM2ProjectedSafeAnalysisError(f"model receiptが不正です: {key}")
        observed.add(key)
    if observed != expected:
        raise AdvantageM2ProjectedSafeAnalysisError("18 fold modelが完全ではありません")
    return records


def load_training_artifact(
    root: Path, projected: trainer.ProjectedSafeSamplesV1,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """学習COMPLETE、予測、18 modelをfail-closedで読む。"""

    plan = _read_json(root / "PLAN.json")
    results = _read_json(root / "results.json")
    complete = _read_json(root / "COMPLETE")
    _validate_training_receipts(root, plan, results, complete)
    _validate_plan(plan)
    arrays = _load_prediction_arrays(root)
    _validate_prediction_identity(arrays, projected)
    records = _validate_model_receipts(root, results, projected)
    return arrays, plan, records


def _safety_contract(
    baseline: np.ndarray, candidate: np.ndarray, supported_indices: np.ndarray,
) -> dict[str, Any]:
    value = np.asarray(candidate, np.float64)
    base = np.asarray(baseline, np.float64)
    unsupported = np.ones(len(base), dtype=bool)
    unsupported[np.asarray(supported_indices, np.int64)] = False
    moved = np.abs(value - base)
    allowed = MAX_PROBABILITY_SHRINK * np.abs(0.5 - base)
    expansion = np.abs(value - 0.5) > np.abs(base - 0.5) + SAFETY_TOLERANCE
    crossing = (base - 0.5) * (value - 0.5) < -SAFETY_TOLERANCE
    ratio = np.divide(moved, np.abs(0.5 - base), out=np.zeros_like(moved),
                      where=np.abs(0.5 - base) > 0.0)
    return {
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(value))),
        "probability_expansion_count": int(np.count_nonzero(expansion)),
        "even_crossing_count": int(np.count_nonzero(crossing)),
        "unsupported_bit_mismatch_count": int(np.count_nonzero(value[unsupported] != base[unsupported])),
        "maximum_shrink_fraction": float(ratio.max(initial=0.0)),
        "maximum_bound_excess": float((moved - allowed).max(initial=0.0)),
    }


def _contract_passed(contract: Mapping[str, Any]) -> bool:
    counts = (
        "nonfinite_count", "probability_expansion_count", "even_crossing_count",
        "unsupported_bit_mismatch_count",
    )
    return (
        all(contract[name] == 0 for name in counts)
        and contract["maximum_shrink_fraction"] <= MAX_PROBABILITY_SHRINK + SAFETY_TOLERANCE
        and contract["maximum_bound_excess"] <= SAFETY_TOLERANCE
    )


def _swap_batch(batch: Any) -> tuple[torch.Tensor, ...]:
    order = torch.as_tensor(SCALAR_SWAP_ORDER, dtype=torch.long, device=batch.scalar.device)
    return (
        batch.current.flip(1), batch.post_chain.flip(1), batch.landing.flip(2),
        batch.branch_mask, batch.scalar.index_select(1, order),
        batch.quantity_present_mask.index_select(1, order),
    )


def _audit_model_symmetry(
    model: ProjectedSetBoundedShrinkCNNV2, projected: trainer.ProjectedSafeSamplesV1,
    local_indices: np.ndarray, saved: np.ndarray, device: torch.device,
) -> tuple[float, float]:
    symmetry_error, reproduction_error = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(local_indices), trainer.LOGICAL_BATCH_SIZE):
            local = local_indices[start:start + trainer.LOGICAL_BATCH_SIZE]
            batch = trainer._batch(projected, local, device)
            global_indices = projected.global_indices[local]
            baseline = torch.as_tensor(projected.full["baseline_probability"][global_indices],
                                       dtype=torch.float64, device=device)
            first = model(
                batch.current, batch.post_chain, batch.landing, batch.branch_mask,
                batch.scalar, batch.quantity_present_mask, baseline,
            )
            swapped = model(*_swap_batch(batch), 1.0 - baseline)
            expected = torch.as_tensor(saved[global_indices], dtype=torch.float64, device=device)
            symmetry_error = max(symmetry_error, float(torch.max(torch.abs(
                swapped.raw_probability - (1.0 - first.raw_probability)
            )).item()))
            reproduction_error = max(reproduction_error, float(torch.max(torch.abs(
                first.raw_probability - expected
            )).item()))
    return symmetry_error, reproduction_error


def _load_model(path: Path, device: torch.device) -> ProjectedSetBoundedShrinkCNNV2:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = ProjectedSetBoundedShrinkCNNV2().to(device)
        model.load_state_dict(payload["state_dict"])
    except (OSError, RuntimeError, KeyError) as error:
        raise AdvantageM2ProjectedSafeAnalysisError(f"modelを復元できません: {path}") from error
    return model


def _operational_model_audit(
    root: Path, records: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray],
    projected: trainer.ProjectedSafeSamplesV1, device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        seed, fold = int(record["seed"]), int(record["eval_fold"])
        path = root / f"m2_projected_safe__seed_{seed}__fold_{fold}.pt"
        model = _load_model(path, device)
        local = np.flatnonzero(projected.supported_folds == fold)
        symmetry, reproduction = _audit_model_symmetry(
            model, projected, local, arrays[f"candidate__seed_{seed}"], device,
        )
        rows.append({"seed": seed, "fold": fold,
                     "side_swap_max_absolute_error": symmetry,
                     "saved_prediction_max_absolute_error": reproduction})
        del model
    return rows


def _load_strata(parent_root: Path, samples: Any) -> dict[str, np.ndarray]:
    with np.load(parent_root / "dataset.npz", allow_pickle=False) as data:
        raw = {name: np.asarray(data[name]) for name in (
            "boards", "ledger_values", "ledger_usable", "available_ms",
            "online_segment_index",
        )}
    return strata_analysis.build_strata(samples, raw)


def _metric_gate(comparison: Mapping[str, Any]) -> dict[str, bool]:
    delta = comparison["overall"]["delta"]
    extreme = comparison["extreme"]
    strata = comparison["strata"]
    return {
        "overall_log_loss_improved": delta["log_loss"] < 0.0,
        "fold_wins_at_least_5_of_6": comparison["fold_win_count"] >= 5,
        "source_bootstrap_ci95_high_below_zero":
            comparison["source_bootstrap"]["ci95_high"] < 0.0,
        "brier_noninferior": delta["brier"] <= 0.0,
        "auc_noninferior": delta["auc"] >= 0.0,
        "winner_below_10_new_zero": extreme["0.1"]["new_count"] == 0,
        "winner_below_20_new_zero": extreme["0.2"]["new_count"] == 0,
        "winner_below_10_total_nonincrease": extreme["0.1"]["candidate_nonincrease"],
        "winner_below_20_total_nonincrease": extreme["0.2"]["candidate_nonincrease"],
        **{f"stratum_{name}_noninferior": strata[name]["delta"]["log_loss"] <= 0.0
           for name in REQUIRED_STRATA},
    }


def _gate(
    comparison: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]],
    operational: Sequence[Mapping[str, Any]], plan: Mapping[str, Any],
) -> dict[str, Any]:
    checks = _metric_gate(comparison)
    checks["all_seed_probability_contracts"] = all(
        _contract_passed(value) for value in contracts.values()
    )
    checks["all_models_side_swap_within_tolerance"] = all(
        row["side_swap_max_absolute_error"] <= SYMMETRY_TOLERANCE for row in operational
    )
    checks["all_models_reproduce_saved_predictions"] = all(
        row["saved_prediction_max_absolute_error"] <= REPRODUCTION_TOLERANCE
        for row in operational
    )
    checks["isolated_from_production_and_holdouts"] = (
        plan.get("production_config_changed") is False
        and plan.get("formal100_used") is False and plan.get("hidden_reserve_used") is False
        and production_compatible(REPO_ROOT)
    )
    return {"passed": all(checks.values()), "checks": checks}


def _markdown(report: Mapping[str, Any]) -> str:
    comparison, gate = report["ensemble_comparison"], report["gate"]
    delta = comparison["overall"]["delta"]
    lines = [
        "# M2 projected-safe-shrink 事前登録監査", "",
        f"- 総合判定: **{'PASS' if gate['passed'] else 'FAIL'}**",
        f"- game-equal log-loss差: `{delta['log_loss']:+.12f}`",
        f"- Brier差: `{delta['brier']:+.12f}`",
        f"- AUC差: `{delta['auc']:+.12f}`",
        f"- fold勝数: `{comparison['fold_win_count']} / 6`",
        f"- source bootstrap 95% CI: "
        f"`[{comparison['source_bootstrap']['ci95_low']:+.12f}, "
        f"{comparison['source_bootstrap']['ci95_high']:+.12f}]`", "", "## Gate", "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] {name}"
                 for name, passed in gate["checks"].items())
    return "\n".join(lines) + "\n"


def _write_exclusive(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _write_outputs(root: Path, report: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    report_path = root / "report.json"
    markdown_path = root / "report.md"
    _write_exclusive(report_path, json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2,
    ) + "\n")
    _write_exclusive(markdown_path, _markdown(report))
    complete = {
        "format_version": COMPLETE_VERSION, "not_production": True,
        "report_sha256": receipt.file_sha256(report_path),
        "markdown_sha256": receipt.file_sha256(markdown_path),
        "gate_passed": report["gate"]["passed"],
        "production_config_changed": False,
    }
    _write_exclusive(root / "COMPLETE", json.dumps(
        complete, ensure_ascii=False, sort_keys=True, indent=2,
    ) + "\n")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    """OOF成果を再計算し、事前登録gateの合否を新規rootへ保存する。"""

    if args.output_root.exists():
        raise AdvantageM2ProjectedSafeAnalysisError(f"出力先は新規必須です: {args.output_root}")
    projected, _ = trainer.load_dataset(args.dataset_root)
    samples, _ = m1_trainer.load_canonical_dataset_v2(args.parent_root)
    _validate_parent_identity(samples, projected)
    arrays, plan, records = load_training_artifact(args.training_root, projected)
    baseline = np.asarray(arrays["baseline_probability"], np.float64)
    candidates = {str(seed): np.asarray(arrays[f"candidate__seed_{seed}"], np.float64)
                  for seed in trainer.TRAINING_SEEDS}
    ensemble = np.mean(np.stack(tuple(candidates.values())), axis=0)
    contracts = {seed: _safety_contract(baseline, value, projected.global_indices)
                 for seed, value in candidates.items()}
    strata = _load_strata(args.parent_root, samples)
    supported = np.zeros(len(baseline), dtype=bool)
    supported[projected.global_indices] = True
    strata.update({"projected_supported": supported, "projected_unsupported": ~supported})
    comparison = common_analysis._comparison(
        samples, baseline, ensemble, strata, samples.available_ms, BOOTSTRAP_SEED,
    )
    per_seed = {seed: common_analysis._comparison(
        samples, baseline, value, strata, samples.available_ms, BOOTSTRAP_SEED + index + 1,
    ) for index, (seed, value) in enumerate(candidates.items())}
    device = trainer._device(args.device)
    operational = _operational_model_audit(
        args.training_root, records, arrays, projected, device,
    )
    gate = _gate(comparison, contracts, operational, plan)
    report = {
        "format_version": FORMAT_VERSION, "not_production": True,
        "training_root": str(args.training_root.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "parent_root": str(args.parent_root.resolve()),
        "state_count": len(baseline), "supported_count": len(projected.global_indices),
        "ensemble_comparison": comparison, "per_seed_comparison": per_seed,
        "probability_contracts": contracts, "operational_model_audit": operational,
        "gate": gate, "production_config_changed": False,
    }
    _write_outputs(args.output_root, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=trainer.DEFAULT_DATASET_ROOT)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = analyze(parse_args(argv))
    summary = report["ensemble_comparison"]["overall"]["delta"]
    print(json.dumps({"gate_passed": report["gate"]["passed"], **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
