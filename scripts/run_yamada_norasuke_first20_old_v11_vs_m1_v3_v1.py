"""先頭20試合の旧V11とM1 V3を同一時刻・左右並列で比較する。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from scripts.run_yamada_norasuke_first40_observed_balance_review_v1 import _run_dir
from scripts.run_zenchi_two_sets_review_delivery_v1 import (
    _audio_has_packets, _decode_entire_media,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "venv/bin/python"
VIDEO_ID = "video_zenchi_c0BQoMJwwQU"
SOURCE = ROOT / f"data/frames/{VIDEO_ID}.mp4"
M1_PREDICTIONS = (
    ROOT / "data/verify/zenchi_set1_m1_v3_external_18model_predictions_2026-09-05_v1"
)
OLD_VIDEO = Path(
    "/mnt/d/puyo_analyzer/videos/review/"
    "yamada_vs_norasuke_first20_visible_receipt_fix_dashboard_2026-09-01/"
    "yamada_vs_norasuke_set1_games01-20_visible_receipt_fix_dashboard.mp4"
)
DELIVERY = Path(
    "/mnt/d/puyo_analyzer/videos/review/"
    "yamada_vs_norasuke_first20_old_v11_vs_m1_v3_2026-09-05_v1"
)
VERIFY = ROOT / "data/verify/yamada_norasuke_first20_old_v11_vs_m1_v3_2026-09-05_v1"
LOGS = ROOT / "logs/yamada_norasuke_first20_old_v11_vs_m1_v3_2026-09-05_v1"
M1_FINAL = DELIVERY / "yamada_vs_norasuke_set1_games01-20_m1_v3_external.mp4"
COMPARISON = DELIVERY / "yamada_vs_norasuke_set1_games01-20_old_v11_vs_m1_v3.mp4"
EXPECTED_OLD_SHA256 = "9c37a351293f54502adb6c87f951e74c58246b6b038f03ccca58f780e935ae87"
M1_METHOD = "three_seed_six_fold_equal_probability_mean_external"
FRAME_STRIDE = 2
OUTPUT_FPS = 30
COMPARISON_CRF = 22


@dataclass(frozen=True)
class Part:
    start: int
    end: int

    @property
    def output(self) -> Path:
        return DELIVERY / f"m1_part_games{self.start:02d}-{self.end:02d}.mp4"


def _parts() -> tuple[Part, ...]:
    return Part(1, 7), Part(8, 14), Part(15, 20)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _render_command(part: Part) -> tuple[str, ...]:
    return (
        str(PYTHON), "scripts/render_provisional_oof_review_v1.py",
        "--heldout-prediction-root", str(M1_PREDICTIONS),
        "--source-video", str(SOURCE), "--source-video-id", VIDEO_ID,
        "--event-source-run", str(_run_dir()), "--output", str(part.output),
        "--family", "linear", "--config", "C", "--probability", "raw",
        "--start-game-position", str(part.start),
        "--end-game-position", str(part.end),
        "--show-probability-graph", "--review-dashboard-layout",
        "--frame-stride", str(FRAME_STRIDE),
    )


def _launch(parts: Sequence[Part]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."
    running: list[tuple[subprocess.Popen[str], Any, Path]] = []
    for part in parts:
        log = LOGS / f"render_m1_{part.start:02d}-{part.end:02d}.log"
        handle = log.open("x", encoding="utf-8")
        process = subprocess.Popen(
            _render_command(part), cwd=ROOT, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, text=True,
        )
        running.append((process, handle, log))
    failures = _wait(running)
    if failures:
        raise RuntimeError(" / ".join(failures))


def _wait(running: Sequence[tuple[subprocess.Popen[str], Any, Path]]) -> list[str]:
    failures: list[str] = []
    for process, handle, log in running:
        code = process.wait()
        handle.close()
        if code:
            failures.append(f"{log}: rc={code}")
    return failures


def _load_part_manifest(part: Part) -> dict[str, Any]:
    path = part.output.with_suffix(".manifest.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"分割manifestがobjectではありません: {path}")
    return value


def _validate_parts(parts: Sequence[Part]) -> list[dict[str, Any]]:
    manifests = [_load_part_manifest(part) for part in parts]
    for part, value in zip(parts, manifests, strict=True):
        segment = value.get("game_segment", {})
        source = value.get("external_prediction_source", {})
        checks = (
            value.get("audio_muxed") is True,
            value.get("review_layout") == "dashboard_v1",
            value.get("probability_graph") is True,
            value.get("frame_stride") == FRAME_STRIDE,
            value.get("m1_external_review") is True,
            (segment.get("start_position"), segment.get("end_position"))
            == (part.start, part.end),
            source.get("prediction_method") == M1_METHOD,
            source.get("model_count") == 18,
            source.get("training_excluded") is True,
            _audio_has_packets(part.output),
        )
        if not all(checks):
            raise RuntimeError(f"M1分割動画の検査に失敗しました: {part.output}")
    return manifests


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _concat(parts: Sequence[Part]) -> None:
    concat_path = VERIFY / "m1_concat.txt"
    concat_path.write_text(
        "".join(f"file '{part.output.as_posix()}'\n" for part in parts),
        encoding="utf-8", newline="\n",
    )
    command = (
        _ffmpeg(), "-n", "-f", "concat", "-safe", "0", "-i",
        str(concat_path), "-c", "copy", str(M1_FINAL),
    )
    _run_ffmpeg(command, M1_FINAL, "M1分割動画の結合")


def _comparison_command() -> tuple[str, ...]:
    graph = (
        f"[0:v]fps={OUTPUT_FPS},setpts=PTS-STARTPTS[left];"
        f"[1:v]fps={OUTPUT_FPS},setpts=PTS-STARTPTS[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    return (
        _ffmpeg(), "-n", "-i", str(OLD_VIDEO), "-i", str(M1_FINAL),
        "-filter_complex", graph, "-map", "[v]", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf",
        str(COMPARISON_CRF), "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-shortest", str(COMPARISON),
    )


def _compose_comparison() -> None:
    if _sha256(OLD_VIDEO) != EXPECTED_OLD_SHA256:
        raise RuntimeError("旧V11動画のSHA256が固定値と一致しません")
    _run_ffmpeg(_comparison_command(), COMPARISON, "新旧比較動画の合成")


def _run_ffmpeg(command: Sequence[str], output: Path, label: str) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode or not output.is_file():
        raise RuntimeError(f"{label}に失敗しました: {completed.stderr[-3000:]}")


def _video_properties(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    value = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return value


def _adjacent(manifests: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        left["game_segment"]["resolved_end_sec"]
        == right["game_segment"]["resolved_start_sec"]
        for left, right in zip(manifests, manifests[1:])
    )


def _report(parts: Sequence[Part], manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    properties = _video_properties(COMPARISON)
    checks = {
        "three_parallel_m1_parts": len(parts) == 3,
        "adjacent_half_open_segments": _adjacent(manifests),
        "m1_audio_packets": _audio_has_packets(M1_FINAL),
        "comparison_audio_packets": _audio_has_packets(COMPARISON),
        "comparison_resolution_3840x1080": (
            properties["width"], properties["height"]
        ) == (3840, 1080),
        "comparison_fps_review_safe": properties["fps"] >= 25.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"完成動画の検査に失敗しました: {checks}")
    return {
        "format": "first20-old-v11-vs-m1-v3-review/v1",
        "layout": "left_old_v11_right_m1_v3_synchronized",
        "checks": checks, "comparison_properties": properties,
        "old_v11": {"path": str(OLD_VIDEO), "sha256": EXPECTED_OLD_SHA256},
        "m1_v3": {"path": str(M1_FINAL), "sha256": _sha256(M1_FINAL)},
        "comparison": {"path": str(COMPARISON), "sha256": _sha256(COMPARISON)},
        "full_decode": _decode_entire_media(COMPARISON),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _prepare_output_directories() -> None:
    for path in (DELIVERY, VERIFY, LOGS):
        if path.exists():
            raise RuntimeError(f"出力先は新規でなければなりません: {path}")
    DELIVERY.mkdir(parents=True, exist_ok=False)
    VERIFY.mkdir(parents=True, exist_ok=False)
    LOGS.mkdir(parents=True, exist_ok=False)


def main() -> int:
    _prepare_output_directories()
    parts = _parts()
    _launch(parts)
    manifests = _validate_parts(parts)
    _concat(parts)
    _compose_comparison()
    report = _report(parts, manifests)
    _write_json(VERIFY / "REPORT.json", report)
    _write_json(VERIFY / "COMPLETE", {"report_sha256": _sha256(VERIFY / "REPORT.json")})
    _write_json(COMPARISON.with_suffix(".manifest.json"), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def finalize_existing_parts() -> int:
    """初回描画後の検査停止から、既存分割を変更せず結合だけ再開する。"""
    parts = _parts()
    required = [path for part in parts for path in (
        part.output, part.output.with_suffix(".manifest.json"),
    )]
    if not all(path.is_file() for path in required):
        raise RuntimeError("再開に必要なM1分割動画またはmanifestがありません")
    if M1_FINAL.exists() or COMPARISON.exists() or (VERIFY / "REPORT.json").exists():
        raise RuntimeError("再開先に結合済み成果物があり、上書きできません")
    manifests = _validate_parts(parts)
    _concat(parts)
    _compose_comparison()
    report = _report(parts, manifests)
    _write_json(VERIFY / "REPORT.json", report)
    _write_json(VERIFY / "COMPLETE", {"report_sha256": _sha256(VERIFY / "REPORT.json")})
    _write_json(COMPARISON.with_suffix(".manifest.json"), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def record_existing_outputs() -> int:
    """合成済み成果物を再生成せず、最終検収receiptだけ記録する。"""
    if not M1_FINAL.is_file() or not COMPARISON.is_file():
        raise RuntimeError("記録対象のM1動画または比較動画がありません")
    if (VERIFY / "REPORT.json").exists() or (VERIFY / "COMPLETE").exists():
        raise RuntimeError("検収receiptは既存のため上書きできません")
    report = _report(_parts(), [_load_part_manifest(part) for part in _parts()])
    _write_json(VERIFY / "REPORT.json", report)
    _write_json(VERIFY / "COMPLETE", {"report_sha256": _sha256(VERIFY / "REPORT.json")})
    _write_json(COMPARISON.with_suffix(".manifest.json"), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
