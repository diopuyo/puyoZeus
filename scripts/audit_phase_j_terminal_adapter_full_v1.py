"""Phase J terminal adapterを固定57試合の実フレーム終局窓で全走査する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from scripts.audit_review_direct_terminal_v1 import VIDEO, WINDOW_FRAMES, _read_events
from src.match_end_detector import DEFAULT_TEMPLATE_DIR
from src.phase_j.terminal_adapter import PhaseJTerminalAdapter, TerminalFrameContext


FORMAT_VERSION = "phase-j-terminal-adapter-full-game-audit/v2"
EXPECTED_GAME_COUNT = 57
MINIMUM_SAFE_DETECTION_COUNT = 12
WILSON_Z_95 = 1.959963984540054
TERMINAL_WINDOW_KIND = "terminal"
NEGATIVE_CONTROL_WINDOW_KIND = "pre_terminal_negative_control"
CAPTURE_SESSION_ID = "phase-j-terminal-full-audit"
ASSET_BUNDLE_ID = "phase-j-terminal-full-audit-assets"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scan_games(
    video: Path, games: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return _scan_windows(video, games, terminal=True)


def scan_negative_control_games(
    video: Path, games: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return _scan_windows(video, games, terminal=False)


def _scan_windows(
    video: Path, games: Sequence[Mapping[str, Any]], *, terminal: bool,
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"動画を開けません: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise RuntimeError("動画FPSを取得できません")
    rows: list[dict[str, Any]] = []
    try:
        for game in games:
            rows.append(_scan_game(capture, fps, game, terminal=terminal))
    finally:
        capture.release()
    return rows


def _scan_game(
    capture: cv2.VideoCapture, fps: float, game: Mapping[str, Any], *, terminal: bool = True,
) -> dict[str, Any]:
    start, end = _window_bounds(int(game["end_frame"]), terminal=terminal)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    adapter = PhaseJTerminalAdapter.load_default()
    event = None
    for frame_index in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        available_ms = int(round(frame_index * 1000.0 / fps))
        event = adapter.update(frame, _context(game, frame_index, available_ms))
        if event is not None:
            break
    expected = "1P" if bool(game["p1_won"]) else "2P"
    return {
        "game_key": str(game["game_key"]), "game_number": int(game["number"]),
        "window_kind": TERMINAL_WINDOW_KIND if terminal else NEGATIVE_CONTROL_WINDOW_KIND,
        "start_frame": start, "end_frame": end,
        "expected_winner": expected, "detected": event is not None,
        "detected_winner": None if event is None else event.payload["winner"],
        "detected_frame": None if event is None else event.payload["source_available_frame"],
        "evidence_kind": None if event is None else event.payload["evidence_kind"],
        "event_digest": None if event is None else event.content_digest,
    }


def _window_bounds(end_frame: int, *, terminal: bool) -> tuple[int, int]:
    if terminal:
        return max(0, end_frame - WINDOW_FRAMES), end_frame
    control_end = max(0, end_frame - WINDOW_FRAMES - 1)
    return max(0, control_end - WINDOW_FRAMES + 1), control_end


def _context(
    game: Mapping[str, Any], frame_index: int, available_ms: int,
) -> TerminalFrameContext:
    return TerminalFrameContext(
        event_seq=int(game["number"]), match_id=str(game["game_key"]),
        capture_session_id=CAPTURE_SESSION_ID, asset_bundle_id=ASSET_BUNDLE_ID,
        source_available_frame=frame_index, source_available_ms=available_ms,
        captured_monotonic_ms=available_ms, observed_monotonic_ms=available_ms,
    )


def build_report(
    rows: Sequence[Mapping[str, Any]],
    negative_control_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    detected = [row for row in rows if row["detected"]]
    controls = [] if negative_control_rows is None else list(negative_control_rows)
    false_positives = [row for row in controls if row["detected"]]
    mismatches = [
        row for row in detected if row["detected_winner"] != row["expected_winner"]
    ]
    direction_coverage = _direction_coverage(rows)
    coverage_gap = abs(direction_coverage["1P"] - direction_coverage["2P"])
    direction_intervals = _direction_intervals(rows)
    checks = {
        "official_game_count_57": len(rows) == EXPECTED_GAME_COUNT,
        "negative_control_game_count_57": len(controls) == EXPECTED_GAME_COUNT,
        "negative_control_false_positive_zero": not false_positives,
        "safe_detection_count_at_least_12": len(detected) >= MINIMUM_SAFE_DETECTION_COUNT,
        "winner_direction_wilson95_intervals_overlap": _intervals_overlap(
            direction_intervals["1P"], direction_intervals["2P"],
        ),
        "winner_direction_mismatch_zero": len(mismatches) == 0,
        "all_evidence_is_bilateral_allowlisted": all(
            row["evidence_kind"] == "visual_result_logo_bilateral_2x2"
            for row in detected
        ),
        "all_events_are_digest_bound": all(
            str(row["event_digest"]).startswith("sha256:") for row in detected
        ),
    }
    return {
        "format": FORMAT_VERSION, "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "official_game_count": len(rows), "detected_count": len(detected),
        "direction_mismatch_count": len(mismatches),
        "direction_coverage": direction_coverage,
        "direction_coverage_wilson95": direction_intervals,
        "direction_coverage_gap": coverage_gap,
        "negative_control_false_positive_count": len(false_positives),
        "rows": list(rows), "negative_control_rows": controls,
    }


def _direction_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    coverage: dict[str, float] = {}
    for winner in ("1P", "2P"):
        matching = [row for row in rows if row["expected_winner"] == winner]
        detected = sum(bool(row["detected"]) for row in matching)
        coverage[winner] = detected / len(matching) if matching else 0.0
    return coverage


def _direction_intervals(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, float]]:
    intervals: dict[str, tuple[float, float]] = {}
    for winner in ("1P", "2P"):
        matching = [row for row in rows if row["expected_winner"] == winner]
        hits = sum(bool(row["detected"]) for row in matching)
        intervals[winner] = _wilson_interval(hits, len(matching))
    return intervals


def _wilson_interval(hits: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    rate = hits / total
    squared = WILSON_Z_95 * WILSON_Z_95
    denominator = 1.0 + squared / total
    center = (rate + squared / (2.0 * total)) / denominator
    margin = WILSON_Z_95 * (
        (rate * (1.0 - rate) / total + squared / (4.0 * total * total)) ** 0.5
    ) / denominator
    return (center - margin, center + margin)


def _intervals_overlap(
    first: tuple[float, float], second: tuple[float, float],
) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


def write_outputs(
    output: Path, report: Mapping[str, Any], video: Path, project_root: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "REPORT.json"
    _write_json(report_path, report)
    manifest = {
        "format": f"{FORMAT_VERSION}-manifest",
        "inputs": {
            "video": file_sha256(video),
            "match_end_batan": file_sha256(
                project_root / DEFAULT_TEMPLATE_DIR / "match_end_batan.png"
            ),
            "match_end_yatta": file_sha256(
                project_root / DEFAULT_TEMPLATE_DIR / "match_end_yatta.png"
            ),
        },
        "code": {
            "audit": file_sha256(Path(__file__).resolve()),
            "phase_j_package": file_sha256(project_root / "src/phase_j/__init__.py"),
            "terminal_adapter": file_sha256(project_root / "src/phase_j/terminal_adapter.py"),
            "terminal_detector": file_sha256(project_root / "src/review_terminal_outcome.py"),
            "match_end_detector": file_sha256(project_root / "src/match_end_detector.py"),
            "terminal_contracts": file_sha256(project_root / "src/phase_j/contracts.py"),
            "terminal_reducer": file_sha256(project_root / "src/phase_j/reducer.py"),
            "reference_audit": file_sha256(project_root / "scripts/audit_review_direct_terminal_v1.py"),
        },
        "outputs": {"REPORT.json": file_sha256(report_path)},
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(output / "COMPLETE", {
        "decision": report["decision"], "report_sha256": file_sha256(report_path),
        "manifest_sha256": file_sha256(manifest_path),
    })


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    _boundaries, games = _read_events()
    rows = scan_games(args.video.resolve(), games)
    negative_rows = scan_negative_control_games(args.video.resolve(), games)
    report = build_report(rows, negative_rows)
    write_outputs(args.output.resolve(), report, args.video.resolve(), args.project_root.resolve())
    print(json.dumps({
        "decision": report["decision"], "games": report["official_game_count"],
        "detected": report["detected_count"], "mismatches": report["direction_mismatch_count"],
        "negative_control_false_positives": report["negative_control_false_positive_count"],
    }))
    return int(report["decision"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
