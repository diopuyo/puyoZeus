"""最終M1レビュー5窓と正式live death observerの因果差を監査する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq

from src.event_death_adapter_v1 import (
    DeathObservationRow,
    EventDeathSidecar,
    load_event_death_sidecar,
)
from src.event_death_observer_v1 import OBSERVATION_RATE_TOLERANCE_HZ


FORMAT_VERSION = "phase-j-review-live-death-equivalence-audit/v1"
TARGETS = ("c138", "c28", "c42", "c76", "c8")
SIDES = ("p1", "p2")
MATCH_TOLERANCE_MS = 250
TARGET_RATE_HZ = 30.0
WINDOWS_MS = {
    "c138": (1_477_166, 1_497_166),
    "c28": (216_267, 238_000),
    "c42": (2_043_166, 2_064_033),
    "c76": (1_719_133, 1_741_700),
    "c8": (3_424_933, 3_444_933),
}
OUTPUT_NAMES = ("REPORT.json", "validation.json", "manifest.json", "COMPLETE")
REVIEW_COLUMNS = (
    "available_frame", "available_ms", "death_gate_transition_p1",
    "death_gate_transition_p2",
)


class PhaseJDeathAuditError(ValueError):
    """入力の同一性または監査契約が成立しない。"""


def file_sha256(path: Path) -> str:
    """ファイル実体のSHA-256を返す。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extract_review_transitions(
    rows: Iterable[Mapping[str, Any]], start_ms: int, end_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """review gateのarm/releaseを対象窓から取り出す。"""

    candidates: list[dict[str, Any]] = []
    releases: list[dict[str, Any]] = []
    for row in rows:
        available_ms = int(row["available_ms"])
        if not start_ms <= available_ms <= end_ms:
            continue
        for side in SIDES:
            transition = row.get(f"death_gate_transition_{side}")
            item = _transition_item(row, side, transition)
            if transition == "armed" or str(transition).endswith("then_rearmed"):
                candidates.append(item)
            if str(transition).startswith("released_"):
                releases.append(item)
    return candidates, releases


def extract_live_transitions(
    rows: Iterable[DeathObservationRow], start_ms: int, end_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """dense observerのcandidate/release/confirmを対象窓から取り出す。"""

    candidates: list[dict[str, Any]] = []
    releases: list[dict[str, Any]] = []
    confirmations: list[dict[str, Any]] = []
    for row in rows:
        if not start_ms <= row.available_ms <= end_ms:
            continue
        for side in SIDES:
            state = getattr(row, side)
            item = _live_transition_item(row, side, state.transition)
            if str(state.transition).startswith("candidate_"):
                candidates.append(item)
            elif str(state.transition).startswith("released_"):
                releases.append(item)
            elif str(state.transition).startswith("confirmed_"):
                confirmations.append(item)
    return candidates, releases, confirmations


def match_transitions(
    review: Sequence[Mapping[str, Any]], live: Sequence[Mapping[str, Any]],
    tolerance_ms: int = MATCH_TOLERANCE_MS,
) -> dict[str, Any]:
    """同じsideの最近傍を一対一対応し、残差も明示する。"""

    unmatched_live = set(range(len(live)))
    matched: list[dict[str, Any]] = []
    unmatched_review: list[dict[str, Any]] = []
    for review_item in review:
        candidates = _candidate_indices(review_item, live, unmatched_live, tolerance_ms)
        if not candidates:
            unmatched_review.append(dict(review_item))
            continue
        live_index = min(candidates, key=lambda index: _distance(review_item, live[index]))
        unmatched_live.remove(live_index)
        matched.append(_match_item(review_item, live[live_index]))
    return {
        "matched": matched,
        "review_only": unmatched_review,
        "live_only": [dict(live[index]) for index in sorted(unmatched_live)],
    }


def audit_target(
    target: str, project_root: Path, live_root: Path, review_root: Path,
    build_results_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """1対象のidentity、範囲、遷移差を監査する。"""

    manifest_path, gate_path, sidecar_path, source_path = _target_paths(
        target, project_root, live_root, review_root,
    )
    manifest = _load_json(manifest_path)
    source_sha = file_sha256(source_path)
    expected_sha = str(manifest["source"]["source_video_sha256"])
    if source_sha != expected_sha:
        raise PhaseJDeathAuditError(f"{target}: 元映像SHAがreview manifestと不一致")
    sidecar = load_event_death_sidecar(sidecar_path, require_decode_contract=True)
    _validate_sidecar_identity(target, sidecar, expected_sha)
    start_ms, end_ms = WINDOWS_MS[target]
    _validate_coverage(target, sidecar, start_ms, end_ms)
    rows = pq.read_table(gate_path, columns=list(REVIEW_COLUMNS)).to_pylist()
    review_candidates, review_releases = extract_review_transitions(rows, start_ms, end_ms)
    live_candidates, live_releases, live_confirmations = extract_live_transitions(
        sidecar.rows, start_ms, end_ms,
    )
    result = _target_result(
        target, sidecar, review_candidates, review_releases,
        live_candidates, live_releases, live_confirmations,
    )
    review_gate_sha = file_sha256(project_root / "src/review_display_death_gate_v1.py")
    build_binding = _optional_build_binding(
        target, build_results_root, sidecar, expected_sha,
    )
    result.update({
        "source_build_id": manifest["source"]["source_build_id"],
        "review_gate_code_matches_manifest": (
            review_gate_sha == manifest["display_gate"]["state_machine_code_sha256"]
        ),
        "live_sidecar_build_id": (
            None if build_binding is None else build_binding["build_id"]
        ),
        "live_build_binding": build_binding,
    })
    hashes = {
        "review_manifest": file_sha256(manifest_path),
        "review_display_gate": file_sha256(gate_path),
        "live_death_sidecar": sidecar.content_sha256,
        "source_video": source_sha,
    }
    if build_binding is not None:
        hashes.update(build_binding["artifact_sha256"])
    return result, hashes


def _target_paths(
    target: str, project_root: Path, live_root: Path, review_root: Path,
) -> tuple[Path, Path, Path, Path]:
    return (
        review_root / f"video_{target}" / "manifest.json",
        review_root / f"video_{target}" / "display_gate.parquet",
        live_root / f"{target}_event_death_v1.json",
        project_root / "data" / "frames" / f"video_{target}.mp4",
    )


def _optional_build_binding(
    target: str, results_root: Path | None, sidecar: EventDeathSidecar,
    source_sha: str,
) -> dict[str, Any] | None:
    if results_root is None:
        return None
    return _validate_build_run(target, results_root, sidecar, source_sha)


def build_report(
    project_root: Path, live_root: Path, review_root: Path,
    build_results_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """5対象を集約し、再利用可否とhard contractを判定する。"""

    results: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {}
    for target in TARGETS:
        result, target_hashes = audit_target(
            target, project_root, live_root, review_root, build_results_root,
        )
        results.append(result)
        hashes[target] = target_hashes
    totals = _sum_counts(results)
    exact = totals["candidate_review_only"] == 0 and totals["candidate_live_only"] == 0
    exact = exact and totals["release_review_only"] == 0 and totals["release_live_only"] == 0
    checks = {
        "all_five_source_identities_match": len(results) == len(TARGETS),
        "all_sidecars_decode_complete": all(row["decode_complete"] for row in results),
        "all_review_windows_covered": all(row["review_window_covered"] for row in results),
        "all_observation_rates_are_30hz": all(row["observation_rate_30hz"] for row in results),
        "all_review_gate_code_matches_manifests": all(
            row["review_gate_code_matches_manifest"] for row in results
        ),
        "live_sidecars_record_build_identity": False,
        "all_live_runs_build_bound": all(
            row["live_build_binding"] is not None for row in results
        ),
        "same_recognition_build_proven": False,
        "no_live_confirmation_in_review_windows": totals["live_confirmations"] == 0,
        "review_gate_exactly_equivalent_to_live_observer": exact,
        "review_gate_reuse_must_remain_forbidden": not exact,
    }
    report = {
        "format": FORMAT_VERSION, "decision": _decision(checks),
        "match_tolerance_ms": MATCH_TOLERANCE_MS, "checks": checks,
        "totals": totals, "targets": results,
        "comparison_scope": (
            "same_source_sha_and_absolute_time; old_review_build_and_current_live_run_"
            "are_independently_hash_bound; pure_same-recognition-build_parity_not_claimed"
        ),
        "terminal_outcome_contract": {
            "death_observer_can_emit_exact_probability": False,
            "exact_zero_or_one_requires": "visual_result_logo_bilateral_2x2",
            "statistical_model_role": "non_terminal_calibrated_prediction_only",
        },
    }
    return report, hashes


def _validate_build_run(
    target: str, results_root: Path, sidecar: EventDeathSidecar,
    source_sha: str,
) -> dict[str, Any]:
    """runner成果物からlive sidecarとevent runのbuild束縛を検証する。"""

    result_path = results_root / f"{target}.json"
    result = _load_json(result_path)
    build_id = str(result.get("build_id", ""))
    run_dir = _resolve_run_dir(results_root, target, build_id)
    manifest_path = run_dir / "manifest.json"
    validation_path = run_dir / "validation.json"
    complete_path = run_dir / "COMPLETE"
    config_path = run_dir / "recognition-config.json"
    manifest = _load_json(manifest_path)
    validation = _load_json(validation_path)
    complete = _load_json(complete_path)
    config = _load_json(config_path)
    _validate_run_receipts(target, result, manifest, validation, complete, run_dir)
    _validate_run_source(target, manifest, config, sidecar, source_sha)
    _validate_embedded_death(target, result, manifest, config, sidecar)
    return {
        "build_bound": True, "build_id": build_id, "run_dir": str(run_dir),
        "artifact_sha256": {
            "build_result": file_sha256(result_path),
            "event_run_manifest": file_sha256(manifest_path),
            "event_run_validation": file_sha256(validation_path),
            "event_run_complete": file_sha256(complete_path),
            "recognition_config": file_sha256(config_path),
        },
    }


def _resolve_run_dir(results_root: Path, target: str, build_id: str) -> Path:
    if not build_id.startswith("build-"):
        raise PhaseJDeathAuditError(f"{target}: build IDが不正です")
    build_root = (
        results_root.parent / "runs" / "schema=v1"
        / f"video=video_{target}" / f"build={build_id}"
    )
    candidates = sorted(path for path in build_root.glob("attempt=*") if path.is_dir())
    if len(candidates) != 1:
        raise PhaseJDeathAuditError(f"{target}: event run attemptを一意に解決できません")
    return candidates[0]


def _validate_run_receipts(
    target: str, result: Mapping[str, Any], manifest: Mapping[str, Any],
    validation: Mapping[str, Any], complete: Mapping[str, Any], run_dir: Path,
) -> None:
    checks = {
        "result/manifest build ID": result.get("build_id") == manifest.get("build_id"),
        "result/manifest event count": result.get("event_count") == manifest.get("adopted_event_count"),
        "validation valid": validation.get("valid") is True,
        "COMPLETE manifest hash": complete.get("manifest_sha256") == file_sha256(run_dir / "manifest.json"),
        "COMPLETE validation hash": complete.get("validation_sha256") == file_sha256(run_dir / "validation.json"),
        "manifest validation hash": manifest.get("validation", {}).get("sha256") == file_sha256(run_dir / "validation.json"),
        "manifest config hash": manifest.get("recognition_config", {}).get("sha256") == file_sha256(run_dir / "recognition-config.json"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PhaseJDeathAuditError(f"{target}: event run receipt不整合: {failed}")


def _validate_run_source(
    target: str, manifest: Mapping[str, Any], config: Mapping[str, Any],
    sidecar: EventDeathSidecar, source_sha: str,
) -> None:
    source = manifest.get("source", {})
    checks = {
        "video ID": source.get("source_video_id") == f"video_{target}",
        "video SHA": source.get("source_video_sha256") == source_sha,
        "start frame": source.get("processing_start_frame") == sidecar.processing_start_frame,
        "end frame": source.get("processing_end_frame_exclusive") == sidecar.processing_end_frame_exclusive,
        "config video ID": config.get("source_video_id") == f"video_{target}",
        "config start frame": config.get("processing_start_frame") == sidecar.processing_start_frame,
        "config end frame": config.get("processing_end_frame_exclusive") == sidecar.processing_end_frame_exclusive,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PhaseJDeathAuditError(f"{target}: event run source不整合: {failed}")


def _validate_embedded_death(
    target: str, result: Mapping[str, Any], manifest: Mapping[str, Any],
    config: Mapping[str, Any], sidecar: EventDeathSidecar,
) -> None:
    death_input = config.get("event_death_input", {})
    embedded = str(death_input.get("content_utf8", "")).encode("utf-8")
    tokens = config.get("collection_tokens", [])
    code_paths = {row.get("relative_path") for row in manifest.get("code_artifacts", [])}
    checks = {
        "result sidecar hash": result.get("death_sidecar_sha256") == sidecar.content_sha256,
        "embedded sidecar hash": death_input.get("sha256") == sidecar.content_sha256,
        "embedded content hash": hashlib.sha256(embedded).hexdigest() == sidecar.content_sha256,
        "death collection flag": "--enable-event-death-sidecar" in tokens,
        "precise seek flag": "--precise-seek" in tokens,
        "observer code receipt": "src/event_death_observer_v1.py" in code_paths,
        "adapter code receipt": "src/event_death_adapter_v1.py" in code_paths,
        "production config receipt": "src/production_config.py" in code_paths,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PhaseJDeathAuditError(f"{target}: death build束縛不整合: {failed}")


def write_outputs(
    output: Path, report: Mapping[str, Any], input_hashes: Mapping[str, Any],
    project_root: Path,
) -> None:
    """監査4成果物を排他的かつhash付きで保存する。"""

    output.mkdir(parents=True, exist_ok=True)
    _reject_existing_outputs(output)
    report_path = output / "REPORT.json"
    _write_json_exclusive(report_path, report)
    validation = {
        "format": f"{FORMAT_VERSION}-validation",
        "checks": report["checks"], "decision": report["decision"],
    }
    validation_path = output / "validation.json"
    _write_json_exclusive(validation_path, validation)
    manifest = {
        "format": f"{FORMAT_VERSION}-manifest", "inputs": input_hashes,
        "code_sha256": {
            "audit": file_sha256(Path(__file__).resolve()),
            "death_observer": file_sha256(project_root / "src/event_death_observer_v1.py"),
            "death_adapter": file_sha256(project_root / "src/event_death_adapter_v1.py"),
            "review_gate": file_sha256(project_root / "src/review_display_death_gate_v1.py"),
            "match_end_detector": file_sha256(project_root / "src/match_end_detector.py"),
            "terminal_detector": file_sha256(project_root / "src/review_terminal_outcome.py"),
            "phase_j_package": file_sha256(project_root / "src/phase_j/__init__.py"),
            "phase_j_terminal_adapter": file_sha256(project_root / "src/phase_j/terminal_adapter.py"),
            "phase_j_reducer": file_sha256(project_root / "src/phase_j/reducer.py"),
            "phase_j_display_projector": file_sha256(project_root / "src/phase_j/display_projector.py"),
            "renderer": file_sha256(project_root / "scripts/render_provisional_oof_review_v1.py"),
            "production_config": file_sha256(project_root / "src/production_config.py"),
            "match_end_batan": file_sha256(project_root / "models/ui_templates/match_end_batan.png"),
            "match_end_yatta": file_sha256(project_root / "models/ui_templates/match_end_yatta.png"),
        },
        "outputs": {
            "REPORT.json": file_sha256(report_path),
            "validation.json": file_sha256(validation_path),
        },
    }
    manifest_path = output / "manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    complete = {
        "format": f"{FORMAT_VERSION}-complete", "decision": report["decision"],
        "manifest_sha256": file_sha256(manifest_path),
        "report_sha256": file_sha256(report_path),
        "validation_sha256": file_sha256(validation_path),
    }
    _write_json_exclusive(output / "COMPLETE", complete)


def _transition_item(
    row: Mapping[str, Any], side: str, transition: Any,
) -> dict[str, Any]:
    return {
        "side": side, "frame_idx": int(row["available_frame"]),
        "available_ms": int(row["available_ms"]), "transition": str(transition),
    }


def _live_transition_item(
    row: DeathObservationRow, side: str, transition: str | None,
) -> dict[str, Any]:
    return {
        "side": side, "frame_idx": row.frame_idx,
        "available_ms": row.available_ms, "transition": str(transition),
    }


def _candidate_indices(
    review: Mapping[str, Any], live: Sequence[Mapping[str, Any]],
    indices: set[int], tolerance_ms: int,
) -> list[int]:
    return [
        index for index in indices
        if live[index]["side"] == review["side"]
        and _distance(review, live[index]) <= tolerance_ms
    ]


def _distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> int:
    return abs(int(first["available_ms"]) - int(second["available_ms"]))


def _match_item(
    review: Mapping[str, Any], live: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "side": review["side"], "delta_ms": int(live["available_ms"]) - int(review["available_ms"]),
        "review": dict(review), "live": dict(live),
    }


def _target_result(
    target: str, sidecar: EventDeathSidecar,
    review_candidates: Sequence[Mapping[str, Any]], review_releases: Sequence[Mapping[str, Any]],
    live_candidates: Sequence[Mapping[str, Any]], live_releases: Sequence[Mapping[str, Any]],
    live_confirmations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_match = match_transitions(review_candidates, live_candidates)
    release_match = match_transitions(review_releases, live_releases)
    return {
        "target": target, "window_ms": list(WINDOWS_MS[target]),
        "decode_complete": sidecar.decode_status == "requested_range_complete",
        "review_window_covered": _is_covered(sidecar, *WINDOWS_MS[target]),
        "observation_rate_30hz": (
            abs(sidecar.observation_rate_hz - TARGET_RATE_HZ)
            <= OBSERVATION_RATE_TOLERANCE_HZ
        ),
        "observed_frame_count": sidecar.observed_frame_count,
        "inspected_side_count": sidecar.inspected_side_count,
        "candidate_match": candidate_match, "release_match": release_match,
        "live_confirmations": [dict(item) for item in live_confirmations],
    }


def _sum_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {
        "candidate_matched": 0, "candidate_review_only": 0,
        "candidate_live_only": 0, "release_matched": 0,
        "release_review_only": 0, "release_live_only": 0,
        "live_confirmations": 0, "observed_frames": 0, "inspected_sides": 0,
    }
    for row in results:
        for prefix in ("candidate", "release"):
            match = row[f"{prefix}_match"]
            totals[f"{prefix}_matched"] += len(match["matched"])
            totals[f"{prefix}_review_only"] += len(match["review_only"])
            totals[f"{prefix}_live_only"] += len(match["live_only"])
        totals["live_confirmations"] += len(row["live_confirmations"])
        totals["observed_frames"] += int(row["observed_frame_count"])
        totals["inspected_sides"] += int(row["inspected_side_count"])
    return totals


def _validate_sidecar_identity(
    target: str, sidecar: EventDeathSidecar, expected_sha: str,
) -> None:
    if sidecar.source_video_id != f"video_{target}":
        raise PhaseJDeathAuditError(f"{target}: death sidecarのvideo IDが不一致")
    if sidecar.source_video_sha256 != expected_sha:
        raise PhaseJDeathAuditError(f"{target}: death sidecarの元映像SHAが不一致")


def _validate_coverage(
    target: str, sidecar: EventDeathSidecar, start_ms: int, end_ms: int,
) -> None:
    if not _is_covered(sidecar, start_ms, end_ms):
        raise PhaseJDeathAuditError(f"{target}: death sidecarがreview窓を包含しません")
    if abs(sidecar.observation_rate_hz - TARGET_RATE_HZ) > OBSERVATION_RATE_TOLERANCE_HZ:
        raise PhaseJDeathAuditError(f"{target}: death sidecarが30Hzではありません")


def _is_covered(sidecar: EventDeathSidecar, start_ms: int, end_ms: int) -> bool:
    start = sidecar.processing_start_frame * 1000 * sidecar.time_base_numerator
    start //= sidecar.time_base_denominator
    end = sidecar.processing_end_frame_exclusive * 1000 * sidecar.time_base_numerator
    end //= sidecar.time_base_denominator
    return start <= start_ms and end >= end_ms


def _decision(checks: Mapping[str, bool]) -> str:
    required = (
        "all_five_source_identities_match", "all_sidecars_decode_complete",
        "all_review_windows_covered", "all_observation_rates_are_30hz",
        "all_review_gate_code_matches_manifests",
    )
    integrity = all(checks[key] for key in required)
    if not integrity:
        return "BLOCKED_INPUT_INTEGRITY"
    if checks.get("all_live_runs_build_bound", False):
        return "NO_GO_REVIEW_GATE_REUSE__FORMAL_LIVE_OBSERVER_IS_AUTHORITY"
    if (
        checks["same_recognition_build_proven"]
        and checks["review_gate_exactly_equivalent_to_live_observer"]
    ):
        return "GO_EQUIVALENT"
    return "NO_GO_REVIEW_GATE_REUSE__REQUIRE_BUILD_BOUND_LIVE_VALIDATION"


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PhaseJDeathAuditError(f"JSON objectではありません: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.write("\n")


def _reject_existing_outputs(output: Path) -> None:
    existing = [name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"監査成果物を上書きしません: {existing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--build-results-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    report, hashes = build_report(
        project_root, args.live_root.resolve(), args.review_root.resolve(),
        None if args.build_results_root is None else args.build_results_root.resolve(),
    )
    write_outputs(args.output.resolve(), report, hashes, project_root)
    print(json.dumps({"decision": report["decision"], "totals": report["totals"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
