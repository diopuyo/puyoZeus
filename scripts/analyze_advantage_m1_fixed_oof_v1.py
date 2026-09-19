"""M0/M1固定OOFを因果撃ち合い・局面・重大誤りで層別監査する。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts import train_advantage_m0_current_cnn_v1 as base
from scripts import build_advantage_m1_dataset_v1 as dataset_v1
from scripts import build_advantage_m1_dataset_v2 as dataset_v2
from scripts.train_advantage_m1_causal_ledger_v1 import (
    CanonicalSamples,
    load_canonical_dataset,
    metrics,
)
from scripts.train_advantage_m1_zero_counterfactual_v3 import (
    CanonicalSamplesV3,
    load_canonical_dataset_v2,
)
from src.advantage_m1_causal_ledger_v1 import (
    LEDGER_CONTEXT_FIELDS as V1_LEDGER_CONTEXT_FIELDS,
    LEDGER_SIDE_FIELDS as V1_LEDGER_SIDE_FIELDS,
    NORMALIZATION_SCALES as V1_NORMALIZATION_SCALES,
)
from src.advantage_m1_causal_ledger_v3 import (
    LEDGER_SIDE_FIELDS as V3_LEDGER_SIDE_FIELDS,
    NORMALIZATION_SCALES as V3_NORMALIZATION_SCALES,
)


ANALYSIS_VERSION = "advantage-m1-fixed-oof-analysis/v1"
BOOTSTRAP_REPEATS = 10000
LARGE_INCOMING = 12.0
VERY_LARGE_INCOMING = 30.0
ELAPSED_STRATUM_MS = 60_000
SEVERE_WINNER_PROBABILITY = 0.20
ADOPTION_GATE_WINNER_PROBABILITY = 0.10
SEVERE_EPISODE_GAP_MS = 5_000
BOARD_CELL_COUNT = 13 * 6
DEATH_CELL_FLAT_INDEX = 1 * 6 + 2
PENDING_DISAGREEMENT_FIELD = "quality_causal_ledger_pending_disagreement"
STATE_DIAGNOSTIC_COLUMNS = (
    "b_p1_chain_active", "b_p2_chain_active",
    "b_p1_pending_garbage", "b_p2_pending_garbage",
    "b_p1_causal_pending_garbage", "b_p2_causal_pending_garbage",
    "b_p1_causal_effective_rate", "b_p2_causal_effective_rate",
    "b_p1_provisional_generated", "b_p2_provisional_generated",
    "b_p1_provisional_score", "b_p2_provisional_score",
    "b_p1_provisional_chain_count", "b_p2_provisional_chain_count",
    "b_causal_exchange_usable", "b_causal_observed_attack_balance",
    "a_p1_occupied_count", "a_p2_occupied_count", "a_p1_grid", "a_p2_grid",
)
GENERIC_COMPARISON_KEYS = {
    "m0": "reference",
    "m0_severe": "reference_severe",
    "m0_log_loss": "reference_log_loss",
    "candidate_minus_m0_log_loss": "candidate_minus_reference_log_loss",
}
AnalysisSamples = CanonicalSamples | CanonicalSamplesV3


class AdvantageM1AnalysisError(RuntimeError):
    """固定OOF成果物または解析母集団が不正。"""


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    samples, dataset_manifest = _load_analysis_dataset(args.dataset_root)
    training = _load_training(args.training_root)
    _validate_training_dataset(training, dataset_manifest)
    predictions = _load_predictions(args.training_root, samples)
    raw_dataset = _load_raw_dataset(args.dataset_root)
    strata = build_strata(samples, raw_dataset)
    comparisons = _comparisons(samples, predictions, strata, args.seed)
    matched_random = _matched_random_comparisons(
        samples, predictions, strata, args.seed,
    )
    state_rows = _state_rows_if_requested(args, dataset_manifest)
    pending_audit = _pending_audit_if_requested(
        args, samples, raw_dataset, predictions, dataset_manifest, state_rows,
    )
    severe_diagnostics = build_severe_diagnostics(
        samples, raw_dataset, predictions, state_rows,
    )
    report = _report(
        args, dataset_manifest, training, strata, comparisons,
        matched_random, pending_audit, severe_diagnostics,
    )
    return _write(args.output_root, report)


def _load_analysis_dataset(root: Path) -> tuple[AnalysisSamples, dict[str, Any]]:
    """manifest世代に対応する排他的dataset loaderだけを呼ぶ。"""

    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM1AnalysisError("dataset manifestを読めません") from error
    version = manifest.get("format_version") if isinstance(manifest, dict) else None
    if version == dataset_v1.FORMAT_VERSION:
        return load_canonical_dataset(root)
    if version == dataset_v2.FORMAT_VERSION:
        return load_canonical_dataset_v2(root)
    raise AdvantageM1AnalysisError(f"未対応dataset formatです: {version}")


def _load_training(root: Path) -> dict[str, Any]:
    result_path, prediction_path = root / "results.json", root / "oof_predictions.npz"
    complete = json.loads((root / "COMPLETE").read_text(encoding="utf-8"))
    if complete.get("results_sha256") != base.file_sha256(result_path):
        raise AdvantageM1AnalysisError("training results hashが一致しません")
    if complete.get("predictions_sha256") != base.file_sha256(prediction_path):
        raise AdvantageM1AnalysisError("OOF prediction hashが一致しません")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _validate_training_dataset(
    training: Mapping[str, Any], dataset: Mapping[str, Any],
) -> None:
    plan, dataset_entry = training.get("plan"), dataset.get("dataset")
    if not isinstance(plan, Mapping) or not isinstance(dataset_entry, Mapping):
        raise AdvantageM1AnalysisError("trainingまたはdatasetのSHA receiptがありません")
    training_sha = plan.get("dataset_sha256")
    dataset_sha = dataset_entry.get("sha256")
    if not training_sha or training_sha != dataset_sha:
        raise AdvantageM1AnalysisError("trainingとdatasetのSHA-256が一致しません")


def _load_predictions(root: Path, samples: AnalysisSamples) -> dict[str, np.ndarray]:
    with np.load(root / "oof_predictions.npz", allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    if not np.array_equal(arrays.get("labels"), samples.labels):
        raise AdvantageM1AnalysisError("predictionとdatasetのlabel順が一致しません")
    if not np.array_equal(arrays.get("folds"), samples.folds):
        raise AdvantageM1AnalysisError("predictionとdatasetのfold順が一致しません")
    return arrays


def _load_raw_dataset(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "dataset.npz", allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in (
            "boards", "ledger_values", "ledger_usable",
            "available_ms", "online_segment_index",
        )}


def build_strata(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    count = len(samples.labels)
    occupied = np.count_nonzero(raw["boards"], axis=(1, 2, 3))
    low, high = np.quantile(occupied, (1.0 / 3.0, 2.0 / 3.0))
    incoming = _incoming_amounts(raw["ledger_values"])
    maximum, receiver = incoming.max(axis=1), incoming.argmax(axis=1)
    selected_board = raw["boards"][np.arange(count), receiver]
    visible_empty = np.count_nonzero(selected_board[:, 1:, :] == 0, axis=(1, 2))
    nonfill = (maximum >= LARGE_INCOMING) & (maximum < visible_empty)
    death_cell_empty = selected_board[:, 1, 2] == 0
    fields, _ = _ledger_contract(raw["ledger_values"])
    chain_index = fields.index("chain_active")
    available_ms = raw.get("available_ms")
    elapsed = (
        _elapsed_ge_60s_mask(samples.game_keys, available_ms)
        if available_ms is not None else np.zeros(count, dtype=bool)
    )
    return {
        "all": np.ones(count, dtype=bool),
        "ledger_usable": np.asarray(raw["ledger_usable"], dtype=bool),
        "ledger_unusable": ~np.asarray(raw["ledger_usable"], dtype=bool),
        "early": occupied <= low, "middle": (occupied > low) & (occupied <= high),
        "late": occupied > high,
        "any_chain_active": raw["ledger_values"][:, :, chain_index].max(axis=1) > 0.5,
        "incoming_at_least_12": maximum >= LARGE_INCOMING,
        "incoming_at_least_30": maximum >= VERY_LARGE_INCOMING,
        "large_nonfill_death_cell_open": nonfill & death_cell_empty,
        "elapsed_ge_60s": elapsed,
    }


def _elapsed_ge_60s_mask(game_keys: np.ndarray, available_ms: np.ndarray) -> np.ndarray:
    """各試合の先頭観測から60秒以上経過した行だけを返す。"""

    times = np.asarray(available_ms, dtype=np.int64)
    if times.shape != np.asarray(game_keys).shape:
        raise AdvantageM1AnalysisError("game keyとavailable_msの行数が一致しません")
    result = np.zeros(len(times), dtype=bool)
    for game in np.unique(game_keys):
        indices = np.flatnonzero(game_keys == game)
        start = int(times[indices[0]])
        result[indices] = times[indices] - start >= ELAPSED_STRATUM_MS
    return result


def _incoming_amounts(values: np.ndarray) -> np.ndarray:
    fields, scales = _ledger_contract(values)
    pending = _decode(
        values[:, :, fields.index("pending_garbage")], "pending_garbage", scales,
    )
    if fields == V3_LEDGER_SIDE_FIELDS:
        return pending
    residual = _decode(
        values[:, :, fields.index("post_cancel_residual")],
        "post_cancel_residual", scales,
    )
    return np.maximum(pending, residual)


def _ledger_contract(
    values: np.ndarray,
) -> tuple[tuple[str, ...], Mapping[str, float]]:
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[1] != 2:
        raise AdvantageM1AnalysisError("ledger value shapeが不正です")
    v1_fields = V1_LEDGER_SIDE_FIELDS + V1_LEDGER_CONTEXT_FIELDS
    if array.shape[2] == len(v1_fields):
        return v1_fields, V1_NORMALIZATION_SCALES
    if array.shape[2] == len(V3_LEDGER_SIDE_FIELDS):
        return V3_LEDGER_SIDE_FIELDS, V3_NORMALIZATION_SCALES
    raise AdvantageM1AnalysisError(f"未対応ledger列数です: {array.shape[2]}")


def _decode(
    values: np.ndarray, field: str,
    scales: Mapping[str, float] = V1_NORMALIZATION_SCALES,
) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0 - 1e-7)
    scale = float(scales[field])
    return scale * clipped / (1.0 - clipped)


def _comparisons(
    samples: AnalysisSamples, predictions: Mapping[str, np.ndarray],
    strata: Mapping[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    candidate_keys = sorted(key[:-12] for key in predictions if key.endswith("__calibrated"))
    for key in candidate_keys:
        variant, seed_text = key.rsplit("__seed_", 1)
        if variant == "m0":
            continue
        baseline_key = f"m0__seed_{seed_text}__calibrated"
        if baseline_key not in predictions:
            raise AdvantageM1AnalysisError(f"同一seed M0がありません: {key}")
        output[key] = compare_candidate(
            samples, predictions[baseline_key], predictions[f"{key}__calibrated"],
            strata, seed + int(seed_text),
        )
    return output


def _matched_random_comparisons(
    samples: AnalysisSamples, predictions: Mapping[str, np.ndarray],
    strata: Mapping[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    calibrated = sorted(key[:-12] for key in predictions if key.endswith("__calibrated"))
    for key in calibrated:
        variant, seed_text = key.rsplit("__seed_", 1)
        reference_variant = _matched_random_variant(variant)
        if reference_variant is None:
            continue
        reference = f"{reference_variant}__seed_{seed_text}"
        prediction_key = f"{reference}__calibrated"
        if prediction_key not in predictions:
            raise AdvantageM1AnalysisError(f"同一seed matched randomがありません: {key}")
        compared = compare_candidate(
            samples, predictions[prediction_key], predictions[f"{key}__calibrated"],
            strata, seed + int(seed_text),
        )
        output[key] = _genericize_comparison(compared)
        output[key]["reference_prediction"] = reference
        output[key]["candidate_prediction"] = key
    return output


def _matched_random_variant(variant: str) -> str | None:
    """M1世代ごとのarchitecture-matched random対照名を返す。"""

    for prefix in ("m1_frozen_", "m1_zero_"):
        if variant.startswith(prefix) and not variant.endswith("random_control"):
            return f"{prefix}random_control"
    return None


def _genericize_comparison(value: Any) -> Any:
    """M0専用の既存結果名を直接paired比較用の汎用名へ変換する。"""

    if isinstance(value, dict):
        return {
            GENERIC_COMPARISON_KEYS.get(str(key), str(key)): _genericize_comparison(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_genericize_comparison(item) for item in value]
    return value


def compare_candidate(
    samples: AnalysisSamples, baseline: np.ndarray, candidate: np.ndarray,
    strata: Mapping[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    baseline_finite, candidate_finite = np.isfinite(baseline), np.isfinite(candidate)
    if not np.array_equal(baseline_finite, candidate_finite):
        raise AdvantageM1AnalysisError("baselineとcandidateのfinite maskが一致しません")
    finite = baseline_finite
    if not finite.any():
        raise AdvantageM1AnalysisError("比較可能なOOF予測がありません")
    losses = _losses(samples.labels, baseline, candidate)
    source_rows = _group_deltas(samples.source_groups, samples.weights, losses, finite)
    fold_rows = _group_deltas(samples.folds, samples.weights, losses, finite)
    return {
        "state_count": int(finite.sum()),
        "overall": _stratum_result(samples, baseline, candidate, finite),
        "strata": {name: _stratum_result(samples, baseline, candidate, finite & mask)
                   for name, mask in strata.items() if bool((finite & mask).any())},
        "source_win_count": sum(row["candidate_minus_m0_log_loss"] < 0 for row in source_rows),
        "source_count": len(source_rows), "source_rows": source_rows,
        "fold_win_count": sum(row["candidate_minus_m0_log_loss"] < 0 for row in fold_rows),
        "fold_count": len(fold_rows), "fold_rows": fold_rows,
        "source_cluster_bootstrap": _bootstrap(source_rows, seed),
    }


def _stratum_result(
    samples: AnalysisSamples, baseline: np.ndarray,
    candidate: np.ndarray, mask: np.ndarray,
) -> dict[str, Any]:
    subset = samples.subset(mask)
    base_probability, candidate_probability = baseline[mask], candidate[mask]
    base_metrics, candidate_metrics = metrics(subset, base_probability), metrics(subset, candidate_probability)
    return {
        "state_count": int(mask.sum()), "game_count": len(np.unique(subset.game_keys)),
        "m0": _finite_metrics(base_metrics), "candidate": _finite_metrics(candidate_metrics),
        "candidate_minus_m0_log_loss": candidate_metrics["game_equal_log_loss"] - base_metrics["game_equal_log_loss"],
        "m0_severe": _severe(subset.labels, base_probability),
        "candidate_severe": _severe(subset.labels, candidate_probability),
    }


def _finite_metrics(values: Mapping[str, float]) -> dict[str, float | None]:
    return {key: float(value) if np.isfinite(value) else None for key, value in values.items()}


def _severe(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    winner_probability = np.where(labels == 1.0, probability, 1.0 - probability)
    return {
        "mean_winner_probability": float(np.mean(winner_probability)),
        "wrong_side_fraction": float(np.mean(winner_probability < 0.5)),
        "winner_below_20_percent_count": int(np.count_nonzero(
            winner_probability < SEVERE_WINNER_PROBABILITY
        )),
        "winner_below_10_percent_count": int(np.count_nonzero(
            winner_probability < ADOPTION_GATE_WINNER_PROBABILITY
        )),
    }


def _severe_mask(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    winner = np.where(labels == 1.0, probability, 1.0 - probability)
    return np.asarray(winner < ADOPTION_GATE_WINNER_PROBABILITY, dtype=bool)


def build_severe_diagnostics(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    state_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """重大誤りの発生・解消行を未来labelを監査だけに使って列挙する。"""

    joined = _joined_state_rows(samples, raw, state_rows)
    output: dict[str, Any] = {}
    for candidate_key in _diagnostic_candidate_keys(predictions):
        arrays, names, finite = _diagnostic_prediction_arrays(
            samples, predictions, candidate_key,
        )
        masks = {name: _severe_mask(samples.labels, value) & finite
                 for name, value in arrays.items()}
        candidate = masks["candidate"]
        output[candidate_key] = {
            "candidate_prediction": candidate_key,
            "m0_prediction": names["m0"],
            "matched_random_prediction": names.get("matched_random"),
            "evaluated_state_count": int(finite.sum()),
            "candidate_vs_m0": _severe_transition(
                samples, raw, arrays, joined, candidate, masks["m0"],
            ),
            "candidate_vs_matched_random": (
                _severe_transition(
                    samples, raw, arrays, joined, candidate, masks["matched_random"],
                ) if "matched_random" in masks else None
            ),
        }
    return {
        "winner_probability_threshold": ADOPTION_GATE_WINNER_PROBABILITY,
        "episode_gap_ms": SEVERE_EPISODE_GAP_MS,
        "prediction_kind": "calibrated", "label_use": "audit_only_not_model_input",
        "state_context_joined": state_rows is not None, "candidates": output,
    }


def _diagnostic_candidate_keys(predictions: Mapping[str, np.ndarray]) -> list[str]:
    output: list[str] = []
    for key in sorted(predictions):
        if not key.endswith("__calibrated"):
            continue
        model_seed = key[:-len("__calibrated")]
        variant = model_seed.rpartition("__seed_")[0]
        if variant != "m0" and not variant.endswith("random_control"):
            output.append(model_seed)
    return output


def _diagnostic_prediction_arrays(
    samples: AnalysisSamples, predictions: Mapping[str, np.ndarray], candidate_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, str], np.ndarray]:
    variant, separator, seed_text = candidate_key.rpartition("__seed_")
    if not separator:
        raise AdvantageM1AnalysisError(f"candidate keyが不正です: {candidate_key}")
    names = {"candidate": candidate_key, "m0": f"m0__seed_{seed_text}"}
    matched = _matched_random_variant(variant)
    if matched is not None:
        names["matched_random"] = f"{matched}__seed_{seed_text}"
    arrays = _named_calibrated_arrays(predictions, names, len(samples.labels))
    masks = [np.isfinite(value) for value in arrays.values()]
    if any(not np.array_equal(masks[0], mask) for mask in masks[1:]):
        raise AdvantageM1AnalysisError(f"重大誤り診断のfinite maskが不一致です: {candidate_key}")
    if not masks[0].any():
        raise AdvantageM1AnalysisError(f"重大誤り診断の評価行がありません: {candidate_key}")
    return arrays, names, masks[0]


def _named_calibrated_arrays(
    predictions: Mapping[str, np.ndarray], names: Mapping[str, str], count: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for role, name in names.items():
        key = f"{name}__calibrated"
        if key not in predictions:
            raise AdvantageM1AnalysisError(f"重大誤り診断の予測がありません: {key}")
        value = np.asarray(predictions[key])
        if value.shape != (count,):
            raise AdvantageM1AnalysisError(f"重大誤り診断のshapeが不正です: {key}")
        output[role] = value
    return output


def _severe_transition(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray], joined: Sequence[Mapping[str, Any]] | None,
    candidate_severe: np.ndarray, reference_severe: np.ndarray,
) -> dict[str, Any]:
    new_rows = _diagnostic_rows(
        samples, raw, probabilities, joined, candidate_severe & ~reference_severe,
    )
    resolved_rows = _diagnostic_rows(
        samples, raw, probabilities, joined, reference_severe & ~candidate_severe,
    )
    new_episodes = _severe_episodes(new_rows)
    resolved_episodes = _severe_episodes(resolved_rows)
    return {
        "new_severe_row_count": len(new_rows), "new_severe_rows": new_rows,
        "new_severe_episode_count": len(new_episodes),
        "new_severe_episodes": new_episodes,
        "resolved_severe_row_count": len(resolved_rows),
        "resolved_severe_rows": resolved_rows,
        "resolved_severe_episode_count": len(resolved_episodes),
        "resolved_severe_episodes": resolved_episodes,
    }


def _diagnostic_rows(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray], joined: Sequence[Mapping[str, Any]] | None,
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = [
        _diagnostic_row(samples, raw, probabilities, joined, index)
        for index in np.flatnonzero(mask)
    ]
    return sorted(rows, key=_diagnostic_sort_key)


def _diagnostic_row(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray], joined: Sequence[Mapping[str, Any]] | None,
    index: int,
) -> dict[str, Any]:
    label = float(samples.labels[index])
    values = {name: float(value[index]) for name, value in probabilities.items()}
    output: dict[str, Any] = {
        "source": str(samples.source_groups[index]), "game": str(samples.game_keys[index]),
        "segment": int(raw["online_segment_index"][index]),
        "time_ms": int(raw["available_ms"][index]), "label": label,
        "fold": int(samples.folds[index]), "ledger_usable": bool(raw["ledger_usable"][index]),
        "candidate_probability": values["candidate"], "m0_probability": values["m0"],
        "matched_random_probability": values.get("matched_random"),
        "candidate_winner_probability": _winner_probability(label, values["candidate"]),
        "m0_winner_probability": _winner_probability(label, values["m0"]),
        "matched_random_winner_probability": (
            _winner_probability(label, values["matched_random"])
            if "matched_random" in values else None
        ),
    }
    output["state_context"] = (
        _diagnostic_state_context(joined[index]) if joined is not None else None
    )
    return output


def _winner_probability(label: float, probability: float) -> float:
    return probability if label == 1.0 else 1.0 - probability


def _diagnostic_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return str(row["source"]), str(row["game"]), int(row["segment"]), int(row["time_ms"])


def _severe_episodes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    runs: list[list[Mapping[str, Any]]] = []
    for row in sorted(rows, key=_diagnostic_sort_key):
        if not runs or _starts_new_severe_episode(runs[-1][-1], row):
            runs.append([])
        runs[-1].append(row)
    return [_severe_episode_summary(run) for run in runs]


def _starts_new_severe_episode(
    previous: Mapping[str, Any], current: Mapping[str, Any],
) -> bool:
    identity = ("source", "game", "segment")
    if any(previous[name] != current[name] for name in identity):
        return True
    return int(current["time_ms"]) - int(previous["time_ms"]) > SEVERE_EPISODE_GAP_MS


def _severe_episode_summary(run: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": run[0]["source"], "game": run[0]["game"],
        "segment": run[0]["segment"], "start_ms": run[0]["time_ms"],
        "end_ms": run[-1]["time_ms"], "row_count": len(run),
        "fold": run[0]["fold"], "label": run[0]["label"],
        "min_candidate_winner_probability": min(
            float(row["candidate_winner_probability"]) for row in run
        ),
        "min_m0_winner_probability": min(
            float(row["m0_winner_probability"]) for row in run
        ),
        "min_matched_random_winner_probability": _minimum_optional(
            row["matched_random_winner_probability"] for row in run
        ),
    }


def _minimum_optional(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return min(finite) if finite else None


def _losses(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    def loss(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability.astype(np.float64), 1e-7, 1.0 - 1e-7)
        return -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return loss(baseline), loss(candidate)


def _group_deltas(
    groups: np.ndarray, weights: np.ndarray,
    losses: tuple[np.ndarray, np.ndarray], mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for group in np.unique(groups[mask]):
        selected = mask & (groups == group)
        first = float(np.average(losses[0][selected], weights=weights[selected]))
        second = float(np.average(losses[1][selected], weights=weights[selected]))
        rows.append({"group": str(group), "m0_log_loss": first,
                     "candidate_log_loss": second,
                     "candidate_minus_m0_log_loss": second - first})
    return rows


def _bootstrap(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, float | int]:
    values = np.asarray([row["candidate_minus_m0_log_loss"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "repeats": BOOTSTRAP_REPEATS, "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "probability_candidate_better": float(np.mean(means < 0.0)),
    }


def _pending_audit_if_requested(
    args: argparse.Namespace, samples: AnalysisSamples,
    raw: Mapping[str, np.ndarray], predictions: Mapping[str, np.ndarray],
    dataset: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    roots = getattr(args, "table_roots", None)
    if not roots:
        return {"performed": False}
    rows = state_rows if state_rows is not None else _load_pending_state_rows(roots, dataset)
    result = audit_pending_disagreement_fallback(samples, raw, predictions, rows)
    result["table_roots"] = [str(Path(root).resolve()) for root in roots]
    return result


def _state_rows_if_requested(
    args: argparse.Namespace, dataset: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    roots = getattr(args, "table_roots", None)
    return _load_pending_state_rows(roots, dataset) if roots else None


def _load_pending_state_rows(
    roots: Sequence[Path], dataset: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    columns = (
        "source_group_id", "online_segment_index", "available_ms",
        PENDING_DISAGREEMENT_FIELD, *STATE_DIAGNOSTIC_COLUMNS,
    )
    paths = _validated_table_paths(roots, dataset)
    return [row for root in paths
            for row in parquet.read_table(root / "states.parquet", columns=list(columns)).to_pylist()]


def _validated_table_paths(
    roots: Sequence[Path], dataset: Mapping[str, Any],
) -> list[Path]:
    sources = dataset.get("sources")
    if not isinstance(sources, list) or not sources:
        raise AdvantageM1AnalysisError("dataset manifestにsource receiptがありません")
    expected = [Path(str(source.get("table_root", ""))).resolve() for source in sources]
    provided = _select_expected_table_roots(roots, expected)
    if len(expected) != len(set(expected)) or len(provided) != len(set(provided)):
        raise AdvantageM1AnalysisError("table rootが重複しています")
    if set(provided) != set(expected):
        raise AdvantageM1AnalysisError("table rootsがdataset receiptと完全一致しません")
    receipts = {Path(str(source["table_root"])).resolve(): source for source in sources}
    for root in expected:
        _validate_table_receipt(root, receipts[root])
    return expected


def _select_expected_table_roots(
    roots: Sequence[Path], expected: Sequence[Path],
) -> list[Path]:
    selected: list[Path] = []
    direct_only = True
    for value in roots:
        root = Path(value).resolve()
        if _is_video_table_root(root):
            selected.append(root)
            continue
        direct_only = False
        schema_root = (root / "schema=v1").resolve()
        matches = [path for path in expected if path.parent == schema_root and path.is_dir()]
        if not matches:
            raise AdvantageM1AnalysisError(f"table rootをvideo dirへ展開できません: {root}")
        selected.extend(matches)
    if direct_only and set(selected) != set(expected):
        raise AdvantageM1AnalysisError("direct video rootsがdataset receiptと完全一致しません")
    return selected


def _is_video_table_root(root: Path) -> bool:
    """group COMPLETEをvideo receiptと誤認せず、実表rootだけを判定する。"""

    return all(
        (root / name).is_file()
        for name in ("manifest.json", "states.parquet", "COMPLETE")
    )


def _validate_table_receipt(root: Path, source: Mapping[str, Any]) -> None:
    manifest_path, complete_path = root / "manifest.json", root / "COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise AdvantageM1AnalysisError(f"table receiptが不足しています: {root}")
    actual = base.file_sha256(manifest_path)
    if actual != source.get("table_manifest_sha256"):
        raise AdvantageM1AnalysisError(f"dataset記録のtable manifest SHAが不一致です: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("manifest_sha256") != actual:
        raise AdvantageM1AnalysisError(f"table COMPLETEのmanifest SHAが不一致です: {root}")
    state_receipt = manifest.get("tables", {}).get("states", {})
    state_path = root / str(state_receipt.get("name", ""))
    if state_path != root / "states.parquet" or not state_path.is_file():
        raise AdvantageM1AnalysisError(f"states.parquet receiptが不正です: {root}")
    if state_receipt.get("sha256") != base.file_sha256(state_path):
        raise AdvantageM1AnalysisError(f"states.parquet SHAが不一致です: {root}")


def audit_pending_disagreement_fallback(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray], state_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mismatch = _pending_disagreement_mask(samples, raw, state_rows)
    pairs = _candidate_prediction_pairs(predictions)
    maximum = 0.0
    evaluated_counts: dict[str, int] = {}
    evaluated_masks: list[np.ndarray] = []
    for candidate_key, baseline_key in pairs:
        candidate = np.asarray(predictions[candidate_key])
        baseline = np.asarray(predictions[baseline_key])
        _validate_prediction_shape(candidate, baseline, mismatch, candidate_key)
        candidate_finite, baseline_finite = np.isfinite(candidate), np.isfinite(baseline)
        if not np.array_equal(candidate_finite, baseline_finite):
            raise AdvantageM1AnalysisError(
                f"candidateとM0のfinite maskが一致しません: {candidate_key}"
            )
        evaluated = mismatch & candidate_finite
        evaluated_counts[candidate_key] = int(evaluated.sum())
        evaluated_masks.append(evaluated)
        if not evaluated.any():
            raise AdvantageM1AnalysisError(
                f"評価済みpending disagreement行がありません: {candidate_key}"
            )
        difference = np.abs(candidate[evaluated] - baseline[evaluated])
        maximum = max(maximum, float(difference.max(initial=0.0)))
        if not np.array_equal(candidate[evaluated], baseline[evaluated]):
            raise AdvantageM1AnalysisError(
                f"pending disagreement行がM0へ完全fallbackしていません: {candidate_key}"
            )
    common_count, masks_common = _common_evaluated_count(evaluated_masks)
    return {
        "performed": True, "joined_state_count": len(mismatch),
        "dataset_pending_disagreement_state_count": int(mismatch.sum()),
        "evaluated_pending_disagreement_state_count": common_count,
        "evaluated_pending_disagreement_state_count_by_prediction": evaluated_counts,
        "evaluated_mismatch_mask_common_across_predictions": masks_common,
        "candidate_prediction_array_count": len(pairs),
        "max_abs_diff": maximum,
        "all_candidates_exact_matched_m0": True,
    }


def _common_evaluated_count(masks: Sequence[np.ndarray]) -> tuple[int | None, bool]:
    common = all(np.array_equal(masks[0], mask) for mask in masks[1:])
    return (int(masks[0].sum()) if common else None), common


def _pending_disagreement_mask(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    lookup = _state_row_lookup(samples, raw, rows)
    output: list[bool] = []
    for key in _dataset_state_keys(samples, raw):
        row = lookup[key]
        amount = row.get(PENDING_DISAGREEMENT_FIELD)
        if isinstance(amount, bool) or not isinstance(amount, (int, np.integer)) or amount < 0:
            raise AdvantageM1AnalysisError(f"pending disagreementが不正です: {key}")
        output.append(amount > 0)
    return np.asarray(output, dtype=bool)


def _joined_state_rows(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]] | None:
    _validate_diagnostic_identity(samples, raw)
    if rows is None:
        return None
    lookup = _state_row_lookup(samples, raw, rows)
    return [lookup[key] for key in _dataset_state_keys(samples, raw)]


def _state_row_lookup(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    lookup: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        key = _state_join_key(
            row["source_group_id"], row["online_segment_index"], row["available_ms"],
        )
        if key in lookup:
            raise AdvantageM1AnalysisError(f"元state join keyが重複しています: {key}")
        lookup[key] = row
    keys = _dataset_state_keys(samples, raw)
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise AdvantageM1AnalysisError(f"元stateへjoinできないdataset行があります: {missing[0]}")
    return lookup


def _dataset_state_keys(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
) -> list[tuple[str, int, int]]:
    segments = np.asarray(raw["online_segment_index"])
    times = np.asarray(raw["available_ms"])
    if len(segments) != len(samples.labels) or len(times) != len(samples.labels):
        raise AdvantageM1AnalysisError("datasetのstate join列の行数が一致しません")
    keys = [
        _state_join_key(group, segment, milliseconds)
        for group, segment, milliseconds in zip(
            samples.source_groups, segments, times, strict=True,
        )
    ]
    if len(keys) != len(set(keys)):
        raise AdvantageM1AnalysisError("datasetのstate join keyが重複しています")
    return keys


def _validate_diagnostic_identity(
    samples: AnalysisSamples, raw: Mapping[str, np.ndarray],
) -> None:
    count = len(samples.labels)
    for name in ("available_ms", "online_segment_index", "ledger_usable"):
        if name not in raw or np.asarray(raw[name]).shape != (count,):
            raise AdvantageM1AnalysisError(f"重大誤り診断の{name}が不正です")
    if samples.game_keys.shape != (count,) or samples.source_groups.shape != (count,):
        raise AdvantageM1AnalysisError("重大誤り診断のidentity列が不正です")


def _diagnostic_state_context(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in STATE_DIAGNOSTIC_COLUMNS if name not in row]
    if missing:
        raise AdvantageM1AnalysisError(f"診断用state列が不足しています: {missing[0]}")
    output = {
        name: _plain_scalar(row[name])
        for name in STATE_DIAGNOSTIC_COLUMNS if not name.endswith("_grid")
    }
    output[PENDING_DISAGREEMENT_FIELD] = _plain_scalar(
        row.get(PENDING_DISAGREEMENT_FIELD)
    )
    for side in ("p1", "p2"):
        grid = row[f"a_{side}_grid"]
        if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)):
            raise AdvantageM1AnalysisError(f"{side} state gridが配列ではありません")
        if len(grid) != BOARD_CELL_COUNT:
            raise AdvantageM1AnalysisError(f"{side} state gridが78セルではありません")
        cell = int(grid[DEATH_CELL_FLAT_INDEX])
        output[f"a_{side}_death_cell_value"] = cell
        output[f"a_{side}_death_cell_occupied"] = cell != 0
    return output


def _plain_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _state_join_key(group: Any, segment: Any, milliseconds: Any) -> tuple[str, int, int]:
    return str(group), int(segment), int(milliseconds)


def _candidate_prediction_pairs(
    predictions: Mapping[str, np.ndarray],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key in sorted(predictions):
        kind = next((value for value in ("raw", "calibrated")
                     if key.endswith(f"__{value}")), None)
        if kind is None:
            continue
        model_seed = key[:-(len(kind) + 2)]
        variant, separator, seed_text = model_seed.rpartition("__seed_")
        if not separator or variant == "m0":
            continue
        baseline = f"m0__seed_{seed_text}__{kind}"
        if baseline not in predictions:
            raise AdvantageM1AnalysisError(f"candidateと同一seed M0がありません: {key}")
        pairs.append((key, baseline))
    if not pairs:
        raise AdvantageM1AnalysisError("fallback監査対象candidateがありません")
    return pairs


def _validate_prediction_shape(
    candidate: np.ndarray, baseline: np.ndarray,
    mask: np.ndarray, candidate_key: str,
) -> None:
    expected = (len(mask),)
    if candidate.shape != expected or baseline.shape != expected:
        raise AdvantageM1AnalysisError(f"OOF prediction shapeが不正です: {candidate_key}")


def _report(
    args: argparse.Namespace, dataset: Mapping[str, Any], training: Mapping[str, Any],
    strata: Mapping[str, np.ndarray], comparisons: Mapping[str, Any],
    matched_random: Mapping[str, Any], pending_audit: Mapping[str, Any],
    severe_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": ANALYSIS_VERSION, "not_production": True,
        "dataset_sha256": dataset["dataset"]["sha256"],
        "training_results_sha256": base.file_sha256(args.training_root / "results.json"),
        "training_plan": training["plan"],
        "stratum_counts": {name: int(mask.sum()) for name, mask in strata.items()},
        "comparisons": dict(comparisons),
        "matched_random_direct_comparisons": dict(matched_random),
        "pending_disagreement_fallback_audit": dict(pending_audit),
        "severe_error_diagnostics": dict(severe_diagnostics or {}),
        "interpretation_contract": {
            "lower_log_loss_is_better": True,
            "large_nonfill_definition": "incoming>=12 and incoming<visible empty cells and recipient death cell empty",
            "elapsed_ge_60s_definition": "available_ms minus first observed game available_ms >= 60000",
            "causal_outcome_hard_override": False,
            "winner_label_used_for_diagnosis_only": True,
            "diagnostic_rows_must_not_reenter_training_inputs": True,
        },
    }


def _write(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    path = base._write_json_exclusive(root / "analysis.json", report)
    base._write_json_exclusive(root / "COMPLETE", {
        "format_version": "advantage-m1-fixed-oof-analysis-complete/v1",
        "analysis_sha256": base.file_sha256(path),
    })
    return dict(report)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--table-roots", type=Path, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = analyze(parse_args(argv))
    print(json.dumps({"comparison_count": len(report["comparisons"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
