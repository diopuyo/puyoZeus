"""Phase J review/live死亡監査器の単体検査。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.event_death_adapter_v1 import (
    DeathBoundaryState,
    DeathObservationRow,
    DeathSideState,
)
from scripts.audit_phase_j_review_live_death_v1 import (
    PhaseJDeathAuditError,
    _decision,
    _resolve_run_dir,
    _validate_embedded_death,
    _validate_run_receipts,
    _validate_run_source,
    extract_live_transitions,
    extract_review_transitions,
    match_transitions,
)


def _side(transition: str | None = None) -> DeathSideState:
    return DeathSideState("clear", transition, None, None, None)


def _live_row(
    frame: int, transition_p2: str | None,
) -> DeathObservationRow:
    return DeathObservationRow(
        frame_idx=frame, available_frame=frame, available_ms=frame * 10,
        game_idx=0, death_generation=0, p1=_side(), p2=_side(transition_p2),
        boundary=DeathBoundaryState(False, None, None, None, None),
    )


def test_extract_review_treats_release_then_rearm_as_both_events() -> None:
    rows = [{
        "available_frame": 100, "available_ms": 1000,
        "death_gate_transition_p1": "none",
        "death_gate_transition_p2": "released_tsumo_then_rearmed",
    }]

    candidates, releases = extract_review_transitions(rows, 900, 1100)

    assert [row["side"] for row in candidates] == ["p2"]
    assert [row["side"] for row in releases] == ["p2"]


def test_extract_review_excludes_rows_outside_closed_window() -> None:
    rows = [{
        "available_frame": 89, "available_ms": 899,
        "death_gate_transition_p1": "armed",
        "death_gate_transition_p2": "none",
    }]

    candidates, releases = extract_review_transitions(rows, 900, 1100)

    assert candidates == []
    assert releases == []


def test_extract_live_separates_candidate_release_and_confirmation() -> None:
    rows = [
        _live_row(100, "candidate_placement"),
        _live_row(110, "released_placement"),
        _live_row(120, "confirmed_placement"),
    ]

    candidates, releases, confirmations = extract_live_transitions(rows, 900, 1300)

    assert [row["frame_idx"] for row in candidates] == [100]
    assert [row["frame_idx"] for row in releases] == [110]
    assert [row["frame_idx"] for row in confirmations] == [120]


def test_match_transitions_is_side_scoped_and_one_to_one() -> None:
    review = [
        {"side": "p1", "available_ms": 1000, "frame_idx": 100},
        {"side": "p2", "available_ms": 1010, "frame_idx": 101},
    ]
    live = [
        {"side": "p2", "available_ms": 1000, "frame_idx": 100},
        {"side": "p1", "available_ms": 1020, "frame_idx": 102},
    ]

    result = match_transitions(review, live, tolerance_ms=25)

    assert [row["delta_ms"] for row in result["matched"]] == [20, -10]
    assert result["review_only"] == []
    assert result["live_only"] == []


def test_match_transitions_reports_static_review_only_candidate() -> None:
    review = [{"side": "p2", "available_ms": 1000, "frame_idx": 100}]

    result = match_transitions(review, [], tolerance_ms=250)

    assert result["matched"] == []
    assert result["review_only"] == review
    assert result["live_only"] == []


def test_match_transitions_reports_dense_live_only_candidate() -> None:
    live = [{"side": "p2", "available_ms": 1000, "frame_idx": 100}]

    result = match_transitions([], live, tolerance_ms=250)

    assert result["matched"] == []
    assert result["review_only"] == []
    assert result["live_only"] == live


def test_match_transitions_rejects_event_beyond_tolerance() -> None:
    review = [{"side": "p2", "available_ms": 1000, "frame_idx": 100}]
    live = [{"side": "p2", "available_ms": 1251, "frame_idx": 125}]

    result = match_transitions(review, live, tolerance_ms=250)

    assert result["matched"] == []
    assert len(result["review_only"]) == 1
    assert len(result["live_only"]) == 1


def test_decision_makes_build_bound_live_observer_authoritative() -> None:
    checks = {
        "all_five_source_identities_match": True,
        "all_sidecars_decode_complete": True,
        "all_review_windows_covered": True,
        "all_observation_rates_are_30hz": True,
        "all_review_gate_code_matches_manifests": True,
        "all_live_runs_build_bound": True,
        "same_recognition_build_proven": False,
        "review_gate_exactly_equivalent_to_live_observer": False,
    }

    assert _decision(checks) == (
        "NO_GO_REVIEW_GATE_REUSE__FORMAL_LIVE_OBSERVER_IS_AUTHORITY"
    )


def test_resolve_run_dir_requires_exactly_one_attempt(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    build = "build-" + "a" * 64
    root = tmp_path / "runs/schema=v1/video=video_c8" / f"build={build}"
    (root / "attempt=one").mkdir(parents=True)

    assert _resolve_run_dir(results, "c8", build) == root / "attempt=one"
    (root / "attempt=two").mkdir()
    with pytest.raises(PhaseJDeathAuditError, match="一意"):
        _resolve_run_dir(results, "c8", build)


def test_validate_build_receipts_and_death_binding(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "validation.json", {"valid": True})
    content = b'{"death":true}\n'
    content_sha = hashlib.sha256(content).hexdigest()
    config = _minimal_config(content.decode(), content_sha)
    _write_json(run_dir / "recognition-config.json", config)
    manifest = _minimal_manifest(run_dir, content_sha)
    _write_json(run_dir / "manifest.json", manifest)
    complete = {
        "manifest_sha256": _sha(run_dir / "manifest.json"),
        "validation_sha256": _sha(run_dir / "validation.json"),
    }
    _write_json(run_dir / "COMPLETE", complete)
    result = {"build_id": "build-x", "event_count": 1,
              "death_sidecar_sha256": content_sha}
    sidecar = SimpleNamespace(content_sha256=content_sha,
                              processing_start_frame=10,
                              processing_end_frame_exclusive=20)

    _validate_run_receipts(
        "c8", result, manifest, {"valid": True}, complete, run_dir,
    )
    _validate_run_source("c8", manifest, config, sidecar, "f" * 64)
    _validate_embedded_death("c8", result, manifest, config, sidecar)


def _minimal_config(content: str, content_sha: str) -> dict[str, object]:
    return {
        "source_video_id": "video_c8", "processing_start_frame": 10,
        "processing_end_frame_exclusive": 20,
        "event_death_input": {"content_utf8": content, "sha256": content_sha},
        "collection_tokens": ["--enable-event-death-sidecar", "--precise-seek"],
    }


def _minimal_manifest(run_dir: Path, _content_sha: str) -> dict[str, object]:
    code_paths = (
        "src/event_death_observer_v1.py", "src/event_death_adapter_v1.py",
        "src/production_config.py",
    )
    return {
        "build_id": "build-x", "adopted_event_count": 1,
        "source": {"source_video_id": "video_c8", "source_video_sha256": "f" * 64,
                   "processing_start_frame": 10, "processing_end_frame_exclusive": 20},
        "validation": {"sha256": _sha(run_dir / "validation.json")},
        "recognition_config": {"sha256": _sha(run_dir / "recognition-config.json")},
        "code_artifacts": [{"relative_path": path} for path in code_paths],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
