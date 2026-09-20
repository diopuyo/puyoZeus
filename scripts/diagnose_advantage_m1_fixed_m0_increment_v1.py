"""固定M0増分のseed安定性・tune感度・重大誤りを既存OOFから並列集計する。"""

from __future__ import annotations

from scripts.production_dependency_contract import (
    dependency_receipt, production_compatible, saved_dependency_compatible,
)

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from scripts import analyze_advantage_m1_fixed_oof_v1 as single
from scripts import compare_advantage_m1_46v_48v_common_v1 as comparison


ROOT = Path("data/verify")
OLD_DATASET = ROOT / "advantage_m1_canonical_dataset_46v_2026-09-06_v4_quarantine_clean"
NEW_DATASET = ROOT / "advantage_m1_canonical_dataset_48v_2026-09-06_v2_quarantine_clean"
OLD_TRAINING = tuple(ROOT / name for name in (
    "advantage_m1_zero_counterfactual_46v_clean_2026-09-06_v1_allfolds_seed20260904_merged",
    "advantage_m1_zero_counterfactual_46v_clean_2026-09-06_v1_allfolds_seeds20260905_20260906_merged",
))
NEW_TRAINING = ROOT / "advantage_m1_fixed_m0_anchor_48v_2026-09-07_v1"
COMPARISON_ROOT = ROOT / "advantage_m1_clean46_fixed_m0_48v_common_comparison_2026-09-07_v1"
SEEDS = (20260904, 20260905, 20260906)
BOOTSTRAP_SEED = 20260906
CPU_WORKERS = 4
TUNE_SENSITIVITY_FOLDS = (1, 3, 4, 5)
SEED_RANGE_MULTIPLIER = 4.0
SEVERE_THRESHOLDS = (0.10, 0.20)


def _seed_audit(
    seed: int, old: Any, new: Any, positions: np.ndarray,
    old_predictions: dict[str, np.ndarray], new_predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """同一seedの共通stateでM0完全一致とM1差を評価する。"""
    arrays = []
    for variant in ("m0", comparison.PRIMARY_VARIANT):
        key = f"{variant}__seed_{seed}__calibrated"
        arrays.extend((old_predictions[key], new_predictions[key][positions]))
    old_m0, new_m0, old_m1, new_m1 = arrays
    if not np.array_equal(old_m0, new_m0):
        raise ValueError(f"seed {seed}: 共通M0が一致しません")
    all_common = np.ones(len(positions), dtype=bool)
    old_delta = single._stratum_result(old, old_m0, old_m1, all_common)
    new_delta = single._stratum_result(old, new_m0, new_m1, all_common)
    increment = single.compare_candidate(old, old_m1, new_m1, {}, BOOTSTRAP_SEED)
    return {
        "seed": seed, "common_m0_bit_exact": True,
        "old_residual_delta": old_delta["candidate_minus_m0_log_loss"],
        "new_common_residual_delta": new_delta["candidate_minus_m0_log_loss"],
        "new_full_residual_delta": _full_seed_delta(seed),
        "increment_common_comparison": increment,
    }


def _full_seed_delta(seed: int) -> float:
    report = comparison._load_json(NEW_TRAINING / "results.json")["results"]
    metric = "game_equal_log_loss"
    baseline = report[f"m0__seed_{seed}"]["metrics_calibrated"][metric]
    candidate = report[f"{comparison.PRIMARY_VARIANT}__seed_{seed}"]["metrics_calibrated"][metric]
    return float(candidate - baseline)


def _severe_changes(old: Any, first: np.ndarray, second: np.ndarray) -> list[dict[str, Any]]:
    """閾値を跨いだ全stateを残し、連続frame件数と試合件数を区別できるようにする。"""
    first_win = np.where(old.labels == 1, first, 1 - first)
    second_win = np.where(old.labels == 1, second, 1 - second)
    rows = []
    for threshold in SEVERE_THRESHOLDS:
        changed = (first_win < threshold) != (second_win < threshold)
        for index in np.flatnonzero(changed):
            rows.append({
                "threshold": threshold, "state_id": str(old.state_ids[index]),
                "source_group": str(old.source_groups[index]),
                "game_key": str(old.game_keys[index]),
                "available_ms": int(old.available_ms[index]),
                "old_winner_probability": float(first_win[index]),
                "new_winner_probability": float(second_win[index]),
                "new_severe": bool(second_win[index] < threshold),
            })
    return rows


def _spread(values: list[float]) -> dict[str, Any]:
    spread = float(np.ptp(values))
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)),
            "range": spread, "sign_split": min(values) < 0 < max(values)}


def diagnose(output: Path) -> dict[str, Any]:
    """検証済み成果物を使い、事前登録に残る感度診断を4 CPU workerで完了する。"""
    old, old_manifest = single._load_analysis_dataset(OLD_DATASET)
    new, new_manifest = single._load_analysis_dataset(NEW_DATASET)
    positions, identity = comparison._identity_audit(old, new)
    if not identity["all_exact_fields_pass"]:
        raise ValueError("共通stateの入力が一致しません")
    old_artifacts, old_mean = comparison._ensemble(old, old_manifest, OLD_TRAINING)
    new_artifacts, new_mean = comparison._ensemble(new, new_manifest, (NEW_TRAINING,))
    old_predictions = {key: value for item in old_artifacts for key, value in item.predictions.items()}
    new_predictions = {key: value for item in new_artifacts for key, value in item.predictions.items()}
    first = comparison._probability(old_mean, comparison.PRIMARY_VARIANT)
    second = comparison._probability(new_mean, comparison.PRIMARY_VARIANT)[positions]
    mask = np.isin(old.folds, TUNE_SENSITIVITY_FOLDS)
    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as pool:
        jobs = [pool.submit(_seed_audit, seed, old, new, positions, old_predictions,
                            new_predictions) for seed in SEEDS]
        sensitivity = pool.submit(single.compare_candidate, old.subset(mask),
                                  first[mask], second[mask], {}, BOOTSTRAP_SEED)
        rows = [job.result() for job in jobs]
        tune = sensitivity.result()
    report = _report(rows, tune, _severe_changes(old, first, second))
    sha = comparison.m0.file_sha256(Path("src/production_config.py"))
    if not production_compatible():
        raise ValueError("production_configのSHAが変化しました")
    report["production_config_sha256"] = sha
    output.mkdir(parents=True, exist_ok=False)
    path = comparison.m0._write_json_exclusive(output / "REPORT.json", report)
    comparison.m0._write_json_exclusive(output / "COMPLETE", {
        "report_sha256": comparison.m0.file_sha256(path),
    })
    return report


def _report(rows: list[dict[str, Any]], tune: dict[str, Any], severe: list[dict[str, Any]]) -> dict[str, Any]:
    previous = _spread([row["old_residual_delta"] for row in rows])
    current = _spread([row["new_common_residual_delta"] for row in rows])
    primary = comparison._load_json(COMPARISON_ROOT / "comparison.json")
    common = primary["comparisons"]["common_champion"]
    if not primary["adoption_gate"]["diagnostic_integrity_pass"]:
        raise ValueError("固定M0の主比較が整合性gateを通過していません")
    return {
        "not_production": True, "cpu_workers": CPU_WORKERS,
        "seed_rows": rows, "old_seed_spread": previous, "new_common_seed_spread": current,
        "seed_instability": current["sign_split"] or current["range"] > SEED_RANGE_MULTIPLIER * previous["range"],
        "new_full_seed_spread": _spread([row["new_full_residual_delta"] for row in rows]),
        "tune_sensitivity_folds": TUNE_SENSITIVITY_FOLDS, "tune_sensitivity": tune,
        "tune_sign_opposes_primary": bool(tune["overall"]["candidate_minus_m0_log_loss"]
                                         * common["overall"]["candidate_minus_m0_log_loss"] < 0),
        "severe_threshold_crossings": severe, "primary_adoption_gate": primary["adoption_gate"],
        "interpretation": "判定保留・clean46維持。平均差は小さいが重大誤り非増加条件は不成立。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.output_root)
    print("固定M0のseed・tune感度診断が完了しました", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
