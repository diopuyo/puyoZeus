"""左右の勝ち・負け表示を全試合で独立走査し、確定方向を監査する。"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as parquet


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
RUN_DIR = ROOT / (
    "data/verify/zenchi_two_sets_review_source_2026-08-31/runs/schema=v1/"
    "video=video_zenchi_c0BQoMJwwQU/"
    "build=build-a2d3d70a5d2ebfb316f4eb993fab9a8ddcf079045917330aea65abfc59feaa24/"
    "attempt=zenchi-review-set1-20260831"
)
TEMPLATE_DIR = ROOT / "models/ui_templates"
SEARCH_AREAS = {"p1": (100, 100, 800, 600), "p2": (1150, 200, 700, 500)}
SCAN_INTERVAL = 2
REQUIRED_HITS = 2
WINDOW_FRAMES = 450
NCC_THRESHOLD = 0.55
MINIMUM_SYMMETRIC_TERMINAL_COUNT = 12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_events() -> tuple[list[int], list[dict[str, Any]]]:
    boundaries: list[int] = []
    games: dict[int, dict[str, Any]] = {}
    batch_ms: dict[str, int] = {}
    for path in sorted((RUN_DIR / "events").glob("part-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                batch = str(row.get("availability_batch_id"))
                if row.get("record_kind") == "batch_begin":
                    batch_ms[batch] = int(row["available_ms"])
                if row.get("event_type") == "match_boundary_evidence":
                    boundaries.append(batch_ms[batch])
                if row.get("event_type") == "winner_observed":
                    game = _official_game(row)
                    games[int(game["number"])] = game
    return sorted(set(boundaries)), [games[number] for number in sorted(games)]


def _official_game(event: Mapping[str, Any]) -> dict[str, Any]:
    number = int(event["payload"]["game_index_unverified"]) + 1
    return {
        "number": number,
        "game_key": f"video_zenchi_c0BQoMJwwQU:game-{number:04d}",
        "p1_won": event["payload"]["winner_side"] == "1P",
        "end_frame": int(event["timing"]["occurred_latest_frame"]),
    }


def _templates() -> dict[str, np.ndarray]:
    output = {}
    for name in ("match_end_batan", "match_end_yatta"):
        image = cv2.imread(str(TEMPLATE_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"決着画像を読めません: {name}")
        output[name] = image
    return output


def _score(
    frame: np.ndarray, templates: Mapping[str, np.ndarray], name: str, side: str,
) -> float:
    x, y, width, height = SEARCH_AREAS[side]
    gray = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    return float(cv2.matchTemplate(gray, templates[name], cv2.TM_CCOEFF_NORMED).max())


def _outcome(
    frame: np.ndarray, templates: Mapping[str, np.ndarray],
) -> tuple[str, float] | None:
    scores = {
        (name, side): _score(frame, templates, name, side)
        for name in templates for side in ("p1", "p2")
    }
    candidates = []
    for loser in ("p1", "p2"):
        winner = "p2" if loser == "p1" else "p1"
        support = min(scores[("match_end_batan", loser)], scores[("match_end_yatta", winner)])
        contradiction = max(scores[("match_end_batan", winner)], scores[("match_end_yatta", loser)])
        if support >= NCC_THRESHOLD and contradiction < NCC_THRESHOLD:
            candidates.append((loser, support))
    return candidates[0] if len(candidates) == 1 else None


def _current_basis(
    rows: Sequence[Mapping[str, Any]], times: Sequence[int], boundaries: Sequence[int], now_ms: int,
) -> Mapping[str, Any] | None:
    index = bisect.bisect_right(times, now_ms) - 1
    if index < 0 or rows[index].get("input_usable") is not True:
        return None
    boundary_index = bisect.bisect_right(boundaries, now_ms) - 1
    boundary = None if boundary_index < 0 else boundaries[boundary_index]
    return None if boundary is not None and times[index] <= boundary else rows[index]


def _scan_game(
    capture: cv2.VideoCapture, fps: float, game: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]], times: Sequence[int], boundaries: Sequence[int],
    templates: Mapping[str, np.ndarray],
) -> dict[int, dict[str, Any]]:
    end = int(game["end_frame"])
    start = max(0, end - WINDOW_FRAMES)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    hits = {phase: (None, 0) for phase in range(SCAN_INTERVAL)}
    found: dict[int, dict[str, Any]] = {}
    for frame_number in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        signal = _outcome(frame, templates)
        now_ms = int(round(frame_number * 1000.0 / fps))
        basis = _current_basis(rows, times, boundaries, now_ms)
        _update_phase(frame_number, signal if basis is not None else None, game, hits, found)
    return found


def _update_phase(
    frame: int, signal: tuple[str, float] | None, game: Mapping[str, Any],
    hits: dict[int, tuple[str | None, int]], found: dict[int, dict[str, Any]],
) -> None:
    phase = frame % SCAN_INTERVAL
    loser = None if signal is None else signal[0]
    previous_loser, previous_count = hits[phase]
    count = previous_count + 1 if loser is not None and loser == previous_loser else int(loser is not None)
    hits[phase] = (loser, count)
    if signal is not None and count >= REQUIRED_HITS and phase not in found:
        found[phase] = {
            "game_key": game["game_key"], "game_number": game["number"],
            "terminal_loser": loser, "displayed_p1_win": loser == "p2",
            "official_p1_win": game["p1_won"], "frame": frame, "score": signal[1],
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise RuntimeError("出力先は新規でなければなりません")
    rows = parquet.read_table(args.predictions).to_pylist()
    rows.sort(key=lambda row: (int(row["available_ms"]), str(row["state_id"])))
    times = [int(row["available_ms"]) for row in rows]
    boundaries, games = _read_events()
    capture = cv2.VideoCapture(str(VIDEO))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 60.0)
    templates = _templates()
    detections = {phase: [] for phase in range(SCAN_INTERVAL)}
    try:
        for game in games:
            for phase, row in _scan_game(
                capture, fps, game, rows, times, boundaries, templates,
            ).items():
                detections[phase].append(row)
    finally:
        capture.release()
    return _report(args, games, detections)


def _report(
    args: argparse.Namespace, games: Sequence[Mapping[str, Any]],
    detections: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    signatures = {
        phase: sorted((row["game_key"], row["terminal_loser"]) for row in rows)
        for phase, rows in detections.items()
    }
    primary = list(detections[0])
    checks = {
        "official_result_count_57": len(games) == 57,
        # 旧23件は片側ロゴ+死亡候補の検出数であり、左右ロゴ同時一致とは
        # 母集団が異なる。厳格な左右一致方式の固定57試合基準は12件。
        "symmetric_terminal_count_at_least_12": (
            len(primary) >= MINIMUM_SYMMETRIC_TERMINAL_COUNT
        ),
        "all_directions_match": all(row["displayed_p1_win"] == row["official_p1_win"] for row in primary),
        "game13_detected": any(row["game_number"] == 13 for row in primary),
        "all_2_scan_phases_match": len({json.dumps(value) for value in signatures.values()}) == 1,
    }
    report = {
        "format": "review-direct-terminal-independent-audit/v1", "checks": checks,
        "terminal_count": len(primary),
        "direction_mismatch_count": sum(row["displayed_p1_win"] != row["official_p1_win"] for row in primary),
        "phase_counts": {str(phase): len(rows) for phase, rows in detections.items()},
        "rows": primary,
        "input_sha256": {"video": _sha256(VIDEO), "predictions": _sha256(args.predictions),
                         "audit_code": _sha256(Path(__file__))},
    }
    if not all(checks.values()):
        missing_by_phase = {
            str(phase): sorted(set(signatures[0]) - set(rows))
            for phase, rows in signatures.items()
        }
        raise RuntimeError(
            f"直接決着監査に失敗しました: {checks}; "
            f"phase_counts={report['phase_counts']}; "
            f"missing_from_phase0={missing_by_phase}"
        )
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = args.output_root / "REPORT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "COMPLETE").write_text(
        json.dumps({"report_sha256": _sha256(path)}, sort_keys=True) + "\n", encoding="utf-8",
    )
    return report


def main() -> int:
    report = run(_parse_args())
    print(json.dumps({"checks": report["checks"], "terminal_count": report["terminal_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
