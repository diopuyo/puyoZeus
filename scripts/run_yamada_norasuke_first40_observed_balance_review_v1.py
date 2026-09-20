"""ヤマダvsのらすけ先頭40試合を新判定・新配置の2本へ安全に書き出す。"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from scripts.run_zenchi_two_sets_review_delivery_v1 import (
    _audio_has_packets,
    _decode_entire_media,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "venv" / "bin" / "python"
VIDEO_ID = "video_zenchi_c0BQoMJwwQU"
SOURCE = ROOT / "data" / "frames" / f"{VIDEO_ID}.mp4"
RUN_ROOT = ROOT / "data" / "verify" / "zenchi_two_sets_review_source_2026-08-31" / "runs"
ATTEMPT = "zenchi-review-set1-20260831"
OOF = ROOT / "data" / "verify" / "late_observed_attack_balance_monotonic_oof_v1_2026-09-01"
SELECTION = ROOT / "data" / "verify" / "observed_attack_balance_residual_review_selection_v1_2026-09-01"
PREDICTIONS = ROOT / "data" / "verify" / "yamada_norasuke_set1_observed_balance_predictions_v4_2026-09-01"
DELIVERY = Path(
    "/mnt/d/puyo_analyzer/videos/review/"
    "yamada_vs_norasuke_first40_observed_balance_residual_dashboard_2026-09-01"
)
VERIFY = ROOT / "data" / "verify" / "yamada_norasuke_first40_observed_balance_review_v2_2026-09-01"
LOGS = ROOT / "logs" / "yamada_norasuke_first40_observed_balance_review_v2_2026-09-01"


@dataclass(frozen=True)
class ReviewPart:
    start: int
    end: int
    output: Path


def _run_dir() -> Path:
    matches = list(RUN_ROOT.glob(f"schema=v1/video={VIDEO_ID}/build=*/attempt={ATTEMPT}"))
    if len(matches) != 1 or not (matches[0] / "COMPLETE").is_file():
        raise RuntimeError("set1の完成済み出来事原本が一意ではありません")
    return matches[0]


def _run_logged(command: Sequence[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command), cwd=ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"処理に失敗しました: {log}")


def _build_predictions(run_dir: Path) -> None:
    if (PREDICTIONS / "COMPLETE").is_file():
        return
    _run_logged((
        str(PYTHON), "scripts/build_heldout_redesign_predictions_v1.py",
        "--event-source-run", str(run_dir), "--selection-root", str(SELECTION),
        "--oof-root", str(OOF), "--output-root", str(PREDICTIONS),
        "--source-group-id", "c0BQoMJwwQU", "--tier", "S級",
    ), LOGS / "build_predictions.log")


def _parts() -> tuple[ReviewPart, ...]:
    return (
        ReviewPart(
            1, 20,
            DELIVERY / "yamada_vs_norasuke_set1_games01-20_observed_balance_residual_dashboard.mp4",
        ),
        ReviewPart(
            21, 40,
            DELIVERY / "yamada_vs_norasuke_set1_games21-40_observed_balance_residual_dashboard.mp4",
        ),
    )


def _render_command(run_dir: Path, part: ReviewPart) -> tuple[str, ...]:
    return (
        str(PYTHON), "scripts/render_provisional_oof_review_v1.py",
        "--oof-root", str(OOF), "--provisional-selection-root", str(SELECTION),
        "--heldout-prediction-root", str(PREDICTIONS),
        "--source-video", str(SOURCE), "--source-video-id", VIDEO_ID,
        "--event-source-run", str(run_dir), "--output", str(part.output),
        "--family", "tree", "--config", "B", "--probability", "raw",
        "--start-game-position", str(part.start), "--end-game-position", str(part.end),
        "--show-probability-graph", "--review-dashboard-layout",
    )


def _render_parallel(run_dir: Path, parts: Sequence[ReviewPart]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."
    running = []
    for part in parts:
        log = LOGS / f"render_games{part.start:02d}-{part.end:02d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("x", encoding="utf-8")
        process = subprocess.Popen(
            _render_command(run_dir, part), cwd=ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT,
        )
        running.append((process, handle, log))
    failures = []
    for process, handle, log in running:
        code = process.wait()
        handle.close()
        if code:
            failures.append(f"{log}: rc={code}")
    if failures:
        raise RuntimeError(" / ".join(failures))


def _layout_pixels(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame.shape[:2] != (1080, 1920):
        raise RuntimeError(f"中央フレームを検査できません: {path}")
    video_std = float(np.std(frame[:810, :1440]))
    probability_non_dark = int(np.count_nonzero(frame[:810, 1440:].max(axis=2) > 80))
    graph_non_dark = int(np.count_nonzero(frame[810:].max(axis=2) > 80))
    if video_std < 10 or probability_non_dark < 100 or graph_non_dark < 100:
        raise RuntimeError(f"新配置の画素検査に失敗しました: {path}")
    return {"video_std": video_std, "probability_pixels": probability_non_dark,
            "graph_pixels": graph_non_dark}


def _validate(part: ReviewPart) -> dict[str, Any]:
    manifest_path = part.output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segment = manifest.get("game_segment", {})
    checks = {
        "audio_muxed": manifest.get("audio_muxed") is True,
        "audio_packets": _audio_has_packets(part.output),
        "dashboard_layout": manifest.get("review_layout") == "dashboard_v1",
        "probability_graph": manifest.get("probability_graph") is True,
        "game_range": (segment.get("start_position"), segment.get("end_position"))
        == (part.start, part.end),
        "total_games": segment.get("total_games") == 57,
        "training_excluded": manifest.get("heldout_prediction_source", {}).get(
            "training_excluded"
        ) is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"成果物検査に失敗しました: {checks}")
    return {"path": str(part.output), "checks": checks,
            "layout": _layout_pixels(part.output), "decode": _decode_entire_media(part.output)}


def main() -> int:
    if VERIFY.exists() or DELIVERY.exists():
        raise RuntimeError("出力先は新規でなければなりません")
    run_dir = _run_dir()
    _build_predictions(run_dir)
    parts = _parts()
    _render_parallel(run_dir, parts)
    results = [_validate(part) for part in parts]
    VERIFY.mkdir(parents=True, exist_ok=False)
    report = {"format": "first40-observed-balance-review/v1", "results": results}
    with (VERIFY / "REPORT.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(
            report, ensure_ascii=False, sort_keys=True, indent=2,
        ) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
