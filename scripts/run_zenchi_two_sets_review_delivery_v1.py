"""全知全能ぷよの30先2セットを、48本仮方式のグラフ付き6本へ自動納品する。

現行方式の出来事原本2本が完成するまで待機し、その後に未学習映像向け予測、
20試合単位の動画書き出し、成果物検証までを順番に行う。元動画と既存成果物は
変更せず、動画はDドライブの新規ディレクトリだけへ保存する。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / "venv" / "bin" / "python"
SOURCE_VIDEO_ID = "video_zenchi_c0BQoMJwwQU"
SOURCE_GROUP_ID = "c0BQoMJwwQU"
SOURCE_VIDEO = REPO_ROOT / "data" / "frames" / f"{SOURCE_VIDEO_ID}.mp4"
RUN_ROOT = REPO_ROOT / "data" / "verify" / "zenchi_two_sets_review_source_2026-08-31" / "runs"
PREDICTION_ROOT = REPO_ROOT / "data" / "verify" / "zenchi_two_sets_heldout_redesign_predictions_v3_2026-08-31"
OOF_ROOT = REPO_ROOT / "data" / "verify" / "indicator_redesign_48_oof_merged_v2_online_uncertain_scope_2026-08-31"
SELECTION_ROOT = REPO_ROOT / "data" / "verify" / "indicator_redesign_provisional_selection_v2_online_uncertain_scope_2026-08-31"
DELIVERY_ROOT = Path("/mnt/d/puyo_analyzer/videos/review/zenchi_two_30first_redesign48_graph_v2_2026-08-31")
VERIFY_ROOT = REPO_ROOT / "data" / "verify" / "zenchi_two_sets_redesign_review_delivery_v2_2026-08-31"
LOG_ROOT = REPO_ROOT / "logs" / "zenchi_two_sets_redesign_review_delivery_v2_2026-08-31"
STATUS_INTERVAL_SEC = 20 * 60
POLL_INTERVAL_SEC = 60
MAX_PARALLEL_JOBS = 2
PART_SIZE_GAMES = 20
EXPECTED_REVIEW_GAME_COUNT = 57
EXPECTED_MODEL_LABEL = "固定48本の暫定方式（100本以上で再学習予定）"
GRAPH_PIXEL_MIN_COUNT = 10


@dataclass(frozen=True)
class SetSpec:
    """1セット分の固定情報。"""

    number: int
    attempt_id: str


@dataclass(frozen=True)
class RenderJob:
    """1本のレビュー動画の書き出し条件。"""

    set_spec: SetSpec
    run_dir: Path
    prediction_root: Path
    start_game: int
    end_game: int
    output: Path


SET_SPECS = (
    SetSpec(1, "zenchi-review-set1-20260831"),
    SetSpec(2, "zenchi-review-set2-20260831"),
)


def _now_text() -> str:
    """UTCの状態記録用時刻を返す。"""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json_exclusive(path: Path, value: Any) -> None:
    """既存ファイルを上書きせずJSONを書き出す。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _find_run(spec: SetSpec) -> Path | None:
    """試行IDに一致する原本ディレクトリを一意に探す。"""

    matches = list(RUN_ROOT.glob(
        f"schema=v1/video={SOURCE_VIDEO_ID}/build=*/attempt={spec.attempt_id}"
    ))
    if len(matches) > 1:
        raise RuntimeError(f"セット{spec.number}の原本候補が複数あります")
    return matches[0] if matches else None


def _status_value(stage: str) -> dict[str, Any]:
    """現在の完了状態を人が追える小さな記録へする。"""

    states = []
    for spec in SET_SPECS:
        run_dir = _find_run(spec)
        states.append({
            "set": spec.number,
            "attempt_id": spec.attempt_id,
            "run_found": run_dir is not None,
            "complete": bool(run_dir and (run_dir / "COMPLETE").is_file()),
            "run_dir": str(run_dir) if run_dir else None,
        })
    return {"stage": stage, "recorded_at_utc": _now_text(), "sets": states}


def _record_status(stage: str) -> None:
    """20分間隔の状態を新規ファイルとして残す。"""

    value = _status_value(stage)
    safe_stage = stage.replace("/", "_")
    path = VERIFY_ROOT / "status" / f"status_{value['recorded_at_utc']}_{safe_stage}.json"
    _write_json_exclusive(path, value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _wait_for_runs() -> dict[int, Path]:
    """2セットの原本が検証済みで完成するまで待つ。"""

    last_status = 0.0
    while True:
        runs = {spec.number: _find_run(spec) for spec in SET_SPECS}
        if all(path and (path / "COMPLETE").is_file() for path in runs.values()):
            return {number: path for number, path in runs.items() if path is not None}
        now = time.monotonic()
        if now - last_status >= STATUS_INTERVAL_SEC:
            _record_status("waiting_for_event_sources")
            last_status = now
        time.sleep(POLL_INTERVAL_SEC)


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    """1コマンドを専用ログへ記録し、失敗を上位へ伝える。"""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."
    with log_path.open("x", encoding="utf-8") as log_handle:
        result = subprocess.run(
            list(command), cwd=REPO_ROOT, env=environment,
            stdout=log_handle, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"処理失敗 rc={result.returncode}: {log_path}")


def _run_parallel(commands: Sequence[tuple[Sequence[str], Path]]) -> None:
    """重い処理をPC上限の2本ずつ実行する。"""

    for offset in range(0, len(commands), MAX_PARALLEL_JOBS):
        batch = commands[offset:offset + MAX_PARALLEL_JOBS]
        processes: list[tuple[subprocess.Popen[str], Path, Any]] = []
        environment = os.environ.copy()
        environment["PYTHONPATH"] = "."
        for command, log_path in batch:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(
                list(command), cwd=REPO_ROOT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            processes.append((process, log_path, handle))
        _wait_parallel_batch(processes)


def _wait_parallel_batch(processes: Sequence[tuple[subprocess.Popen[str], Path, Any]]) -> None:
    """並列バッチを待ち、すべてのログを確実に閉じる。"""

    failures: list[str] = []
    last_status = time.monotonic()
    try:
        pending = list(processes)
        while pending:
            remaining = []
            for process, log_path, handle in pending:
                return_code = process.poll()
                if return_code is None:
                    remaining.append((process, log_path, handle))
                elif return_code != 0:
                    failures.append(f"{log_path}: rc={return_code}")
            pending = remaining
            now = time.monotonic()
            if pending and now - last_status >= STATUS_INTERVAL_SEC:
                _record_status("long_running_parallel_work")
                last_status = now
            if pending:
                time.sleep(min(POLL_INTERVAL_SEC, 30))
    finally:
        for _process, _log_path, handle in processes:
            handle.close()
    if failures:
        raise RuntimeError("並列処理に失敗しました: " + ", ".join(failures))


def _build_predictions(runs: dict[int, Path]) -> dict[int, Path]:
    """固定48本の候補18モデルと補完18モデルで仮勝率を作る。"""

    commands = []
    roots: dict[int, Path] = {}
    for spec in SET_SPECS:
        root = PREDICTION_ROOT / f"set{spec.number}"
        roots[spec.number] = root
        if (root / "COMPLETE").is_file():
            continue
        command = (
            str(PYTHON), "scripts/build_heldout_redesign_predictions_v1.py",
            "--event-source-run", str(runs[spec.number]),
            "--selection-root", str(SELECTION_ROOT),
            "--oof-root", str(OOF_ROOT), "--output-root", str(root),
            "--source-group-id", SOURCE_GROUP_ID, "--tier", "S級",
        )
        commands.append((command, LOG_ROOT / f"set{spec.number}_predictions.log"))
    if commands:
        _run_parallel(commands)
    return roots


def _prediction_game_count(root: Path) -> int:
    """予測結果の公式試合数を検証して返す。"""

    value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if value.get("training_excluded") is not True:
        raise RuntimeError("レビュー元映像が学習から除外されていません")
    count = int(value.get("review_game_count", 0))
    if count != EXPECTED_REVIEW_GAME_COUNT:
        raise RuntimeError(
            f"レビュー対象は57試合固定です: 実際={count}"
        )
    return count


def _render_jobs(runs: dict[int, Path], roots: dict[int, Path]) -> list[RenderJob]:
    """各セットを20試合・20試合・残りへ分ける。"""

    jobs = []
    for spec in SET_SPECS:
        count = _prediction_game_count(roots[spec.number])
        ranges = ((1, PART_SIZE_GAMES), (21, 40), (41, count))
        for part, (start, end) in enumerate(ranges, start=1):
            filename = f"yamada_vs_norasuke_set{spec.number}_part{part}_games{start:02d}-{end:02d}.mp4"
            jobs.append(RenderJob(
                spec, runs[spec.number], roots[spec.number], start, end,
                DELIVERY_ROOT / filename,
            ))
    return jobs


def _render_command(job: RenderJob) -> tuple[str, ...]:
    """グラフ・音声付きレビュー動画1本の実行引数を作る。"""

    return (
        str(PYTHON), "scripts/render_provisional_oof_review_v1.py",
        "--oof-root", str(OOF_ROOT),
        "--provisional-selection-root", str(SELECTION_ROOT),
        "--heldout-prediction-root", str(job.prediction_root),
        "--source-video", str(SOURCE_VIDEO), "--source-video-id", SOURCE_VIDEO_ID,
        "--event-source-run", str(job.run_dir), "--output", str(job.output),
        "--family", "tree", "--config", "B", "--probability", "raw",
        "--start-game-position", str(job.start_game),
        "--end-game-position", str(job.end_game), "--show-probability-graph",
    )


def _render(jobs: Sequence[RenderJob]) -> None:
    """6本を2本ずつ書き出す。"""

    commands = [
        (_render_command(job), LOG_ROOT / f"render_{job.output.stem}.log")
        for job in jobs
    ]
    _run_parallel(commands)


def _validate_render(job: RenderJob) -> dict[str, Any]:
    """人へ渡す動画の必須条件を1本ずつ確認する。"""

    manifest_path = job.output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segment = manifest.get("game_segment", {})
    heldout = manifest.get("heldout_prediction_source", {})
    stats = manifest.get("stats", {})
    checks = {
        "output_exists": job.output.is_file() and job.output.stat().st_size > 0,
        "audio_muxed": manifest.get("audio_muxed") is True,
        "probability_graph": manifest.get("probability_graph") is True,
        "graph_has_values": int(stats.get("graph_point_count", 0)) > 0,
        "training_excluded": heldout.get("training_excluded") is True,
        "provisional_48": manifest.get("provisional_48") is True,
        "selection_linked": heldout.get("selection_sha256") is not None,
        "raw_tree_b": (
            manifest.get("family"), manifest.get("config"), manifest.get("probability")
        ) == ("tree", "B", "raw"),
        "game_range": (
            segment.get("start_position"), segment.get("end_position")
        ) == (job.start_game, job.end_game),
        "total_games_57": segment.get("total_games") == EXPECTED_REVIEW_GAME_COUNT,
    }
    if not all(checks.values()):
        raise RuntimeError(f"動画検証に失敗しました: {job.output}: {checks}")
    media = _validate_completed_media(job, manifest)
    return {
        "path": str(job.output), "checks": checks, "media": media,
        "manifest": str(manifest_path),
    }


def _ffmpeg_executable() -> str:
    """全域デコードに使えるffmpegを返す。"""

    if path := shutil.which("ffmpeg"):
        return path
    try:
        import imageio_ffmpeg
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as error:
        raise RuntimeError("完成動画を検査するffmpegが見つかりません") from error


def _decode_entire_media(path: Path) -> dict[str, Any]:
    """映像と音声を最後まで復号し、破損と音声欠落を検出する。"""

    command = [
        _ffmpeg_executable(), "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"映像・音声の全域復号に失敗しました: {path}: {result.stderr}")
    return {"full_video_audio_decode": True}


def _audio_has_packets(path: Path) -> bool:
    """完成動画の音声ストリームに実データがあることを確認する。"""

    command = [
        _ffmpeg_executable(), "-v", "error", "-t", "0.5", "-i", str(path),
        "-map", "0:a:0", "-f", "framecrc", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    packets = [line for line in result.stdout.splitlines() if line and not line.startswith("#")]
    return result.returncode == 0 and bool(packets)


def _graph_sample_times(job: RenderJob) -> list[tuple[str, int]]:
    """指定範囲の各試合から、確率が表示できる代表時刻を1つ返す。"""

    import pyarrow.parquet as parquet
    rows = parquet.read_table(
        job.prediction_root / "predictions.parquet",
        columns=["game_key", "available_ms", "input_usable", "raw_probability"],
    ).to_pylist()
    by_game: dict[str, list[int]] = {}
    for row in rows:
        if row["game_key"] is None or not row["input_usable"] or row["raw_probability"] is None:
            continue
        by_game.setdefault(str(row["game_key"]), []).append(int(row["available_ms"]))
    ordered = sorted(by_game.items(), key=lambda item: min(item[1]))
    selected = ordered[job.start_game - 1:job.end_game]
    return [(key, sorted(times)[len(times) // 2]) for key, times in selected]


def _has_graph_pixels(frame: Any, source_height: int) -> tuple[bool, bool]:
    """完成画面から折れ線色と左上の黄色い説明文を検出する。"""

    import cv2
    panel = frame[source_height:]
    graph = panel[:, frame.shape[1] // 2:]
    blue, green, red = cv2.split(graph)
    line = (blue > 170) & (green > 110) & (green < 240) & (red > 30) & (red < 180)
    caption = panel[:45, :frame.shape[1] // 2]
    cb, cg, cr = cv2.split(caption)
    yellow = (cb > 30) & (cb < 150) & (cg > 150) & (cr > 170)
    return int(line.sum()) >= GRAPH_PIXEL_MIN_COUNT, int(yellow.sum()) >= GRAPH_PIXEL_MIN_COUNT


def _validate_graph_for_every_game(
    job: RenderJob, manifest: dict[str, Any],
) -> dict[str, Any]:
    """分割内の全試合で、完成動画にグラフが実描画されていることを確認する。"""

    import cv2
    source = cv2.VideoCapture(str(SOURCE_VIDEO))
    source_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source.release()
    capture = cv2.VideoCapture(str(job.output))
    start_sec = float(manifest["requested_start_sec"])
    passed, caption_passed = [], True
    for key, available_ms in _graph_sample_times(job):
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, available_ms - start_sec * 1000.0))
        ok, frame = capture.read()
        graph_ok, caption_ok = (False, False) if not ok else _has_graph_pixels(frame, source_height)
        if graph_ok:
            passed.append(key)
        caption_passed = caption_passed and caption_ok
    capture.release()
    expected = job.end_game - job.start_game + 1
    if len(passed) != expected or not caption_passed:
        raise RuntimeError(f"完成画面のグラフまたは説明表示が不足しています: {job.output}")
    return {"graph_games_with_pixels": len(passed), "caption_pixels": caption_passed}


def _validate_completed_media(
    job: RenderJob, manifest: dict[str, Any],
) -> dict[str, Any]:
    """自己申告値に頼らず完成媒体を検査する。"""

    if not _audio_has_packets(job.output):
        raise RuntimeError(f"完成動画に音声データがありません: {job.output}")
    result = {"audio_packets": True, "model_label": EXPECTED_MODEL_LABEL}
    result.update(_validate_graph_for_every_game(job, manifest))
    result.update(_decode_entire_media(job.output))
    return result


def _validate_partitions(jobs: Sequence[RenderJob]) -> dict[str, Any]:
    """2セットの57試合とフレーム区間が欠落・重複しないことを確認する。"""

    expected = [(1, 20), (21, 40), (41, EXPECTED_REVIEW_GAME_COUNT)]
    for spec in SET_SPECS:
        selected = [job for job in jobs if job.set_spec == spec]
        if [(job.start_game, job.end_game) for job in selected] != expected:
            raise RuntimeError(f"セット{spec.number}の試合分割が20/20/17ではありません")
        manifests = [json.loads(job.output.with_suffix(".manifest.json").read_text()) for job in selected]
        spans = [
            (int(value["stats"]["source_start_frame"]), int(value["stats"]["source_end_frame_exclusive"]))
            for value in manifests
        ]
        if any(left[1] != right[0] for left, right in zip(spans, spans[1:])):
            raise RuntimeError(f"セット{spec.number}の分割境界に欠落または重複があります: {spans}")
    return {"sets": 2, "games_per_set": 57, "split": [20, 20, 17], "frame_overlap": 0}


def _finish(jobs: Sequence[RenderJob], runs: dict[int, Path]) -> None:
    """6本全体の検収記録を上書きなしで確定する。"""

    outputs = [_validate_render(job) for job in jobs]
    partition_validation = _validate_partitions(jobs)
    value = {
        "format": "zenchi-two-set-redesign-review-delivery/v1",
        "completed_at_utc": _now_text(),
        "source_video": str(SOURCE_VIDEO),
        "source_group_id": SOURCE_GROUP_ID,
        "selection_root": str(SELECTION_ROOT),
        "limitations": [
            "48本による仮方式。100本以上で本決定する。",
            "対象のチャレンジャー級は48本感度分析で母数不足の保留層。",
            "映像は学習外だが、選手そのものの完全未見試験ではない。",
        ],
        "event_runs": {str(key): str(value) for key, value in runs.items()},
        "split_policy": "各セットを試合順1-20、21-40、41-最終へ分割",
        "output_count": len(outputs), "outputs": outputs,
        "partition_validation": partition_validation,
    }
    _write_json_exclusive(VERIFY_ROOT / "DELIVERY.json", value)
    _record_status("complete")


def _parse_args() -> argparse.Namespace:
    """再生成せず完成媒体だけを再検査できる引数を読む。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """待機から最終検証まで自律実行する。"""

    args = _parse_args()
    if not PYTHON.is_file() or not SOURCE_VIDEO.is_file():
        raise RuntimeError("実行環境または元動画が見つかりません")
    runs = _wait_for_runs()
    prediction_roots = _build_predictions(runs)
    jobs = _render_jobs(runs, prediction_roots)
    if not args.validate_only:
        _record_status("building_predictions")
        _record_status("rendering_six_videos")
        _render(jobs)
    _record_status("validating_six_videos")
    _finish(jobs, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
