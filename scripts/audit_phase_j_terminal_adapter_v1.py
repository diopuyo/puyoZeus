"""既存の実フレーム決着13件をPhase J terminal adapterで再検証する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from src.match_end_detector import DEFAULT_TEMPLATE_DIR
from src.phase_j.terminal_adapter import PhaseJTerminalAdapter, TerminalFrameContext


FORMAT_VERSION = "phase-j-terminal-adapter-positive-replay/v1"
FRAME_MARGIN = 12
CAPTURE_SESSION_ID = "phase-j-terminal-audit-capture"
ASSET_BUNDLE_ID = "phase-j-terminal-audit-assets"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def replay_reference_rows(
    video: Path, rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"動画を開けません: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise RuntimeError("動画FPSを取得できません")
    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            results.append(_replay_one(capture, fps, row))
    finally:
        capture.release()
    return results


def _replay_one(
    capture: cv2.VideoCapture, fps: float, row: Mapping[str, Any],
) -> dict[str, Any]:
    reference_frame = int(row["frame"])
    start = max(0, reference_frame - FRAME_MARGIN)
    end = reference_frame + FRAME_MARGIN
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    adapter = PhaseJTerminalAdapter.load_default()
    event = None
    for frame_index in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        available_ms = int(round(frame_index * 1000.0 / fps))
        context = _context(row, frame_index, available_ms)
        event = adapter.update(frame, context)
        if event is not None:
            break
    expected = "1P" if bool(row["official_p1_win"]) else "2P"
    return {
        "game_key": str(row["game_key"]), "reference_frame": reference_frame,
        "expected_winner": expected, "detected": event is not None,
        "detected_frame": None if event is None else event.payload["source_available_frame"],
        "detected_winner": None if event is None else event.payload["winner"],
        "result_code": None if event is None else event.payload["result_code"],
        "evidence_kind": None if event is None else event.payload["evidence_kind"],
        "event_digest": None if event is None else event.content_digest,
    }


def _context(
    row: Mapping[str, Any], frame_index: int, available_ms: int,
) -> TerminalFrameContext:
    return TerminalFrameContext(
        event_seq=int(row["game_number"]), match_id=str(row["game_key"]),
        capture_session_id=CAPTURE_SESSION_ID, asset_bundle_id=ASSET_BUNDLE_ID,
        source_available_frame=frame_index, source_available_ms=available_ms,
        captured_monotonic_ms=available_ms, observed_monotonic_ms=available_ms,
    )


def build_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    detected = [row for row in rows if row["detected"]]
    checks = {
        "all_reference_detections_reproduced": len(detected) == len(rows),
        "all_winner_directions_match": all(
            row["detected_winner"] == row["expected_winner"] for row in detected
        ),
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
        "reference_count": len(rows), "detected_count": len(detected),
        "direction_mismatch_count": sum(
            row["detected"] and row["detected_winner"] != row["expected_winner"]
            for row in rows
        ),
        "rows": list(rows),
    }


def write_outputs(
    output: Path, report: Mapping[str, Any], reference: Path, video: Path,
    project_root: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "REPORT.json"
    _write_json(report_path, report)
    manifest = {
        "format": f"{FORMAT_VERSION}-manifest",
        "inputs": {
            "reference_report": file_sha256(reference), "video": file_sha256(video),
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
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    reference = args.reference_report.resolve()
    value = json.loads(reference.read_text(encoding="utf-8"))
    rows = replay_reference_rows(args.video.resolve(), value["rows"])
    report = build_report(rows)
    write_outputs(
        args.output.resolve(), report, reference, args.video.resolve(),
        args.project_root.resolve(),
    )
    print(json.dumps({"decision": report["decision"], "detected": report["detected_count"]}))
    return int(report["decision"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
