"""短区間を現行認識で収集し、確定盤面の出来事原本v1を新規作成する。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch

from src.event_run_v1 import ArtifactHash, RuntimeEnvironmentSpec, SourceVideoSpec
from src.event_snapshot_export_v1 import (
    SnapshotExportRequest,
    SnapshotExportResult,
    discover_runtime_artifacts,
    export_snapshot_run,
    file_sha256,
)
from src.production_config import collect_flags


PROCESSED_RANGE_PATTERN = re.compile(
    r"\[lean\] processed_frame_range: start=(\d+) "
    r"end_exclusive=(\d+) requested_end_exclusive=(\d+)"
)
EVENT_OBSERVATION_SIDECAR_SUFFIX = "_event_observations_v1.json"
EVENT_ACCOUNTING_SIDECAR_SUFFIX = "_event_accounting_v1.json"
EVENT_PHYSICAL_SIDECAR_SUFFIX = "_event_physical_v1.json"


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """元映像の固定メタデータ。"""

    width: int
    height: int
    frame_count: int
    fps: float
    time_base_numerator: int
    time_base_denominator: int


def probe_video(path: Path) -> VideoMetadata:
    """OpenCVが実際に読む映像メタデータと時間基準を得る。"""

    capture = cv2.VideoCapture(str(path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0:
        raise ValueError("元映像の寸法・フレーム数・fpsを取得できません")
    time_base = Fraction(1.0 / fps).limit_denominator(1_000_000)
    return VideoMetadata(width, height, frame_count, fps, time_base.numerator, time_base.denominator)


def collection_tokens(
    start_sec: float,
    max_sec: float,
    score_region_calibration: str | None = None,
) -> tuple[str, ...]:
    """単一情報源の現行収集設定へ短区間条件を加える。"""

    tokens = collect_flags().split()
    for flag in (
        "--with-next",
        "--enable-phantom-board-guard",
        "--precise-seek",
        "--normalize-fps-30",
        "--enable-event-accounting-sidecar",
        "--enable-event-physical-sidecar",
    ):
        _append_flag(tokens, flag)
    _append_option(tokens, "--start-sec", _number_text(start_sec))
    _append_option(tokens, "--max-sec", _number_text(max_sec))
    _append_option(tokens, "--sample-interval", "0")
    if score_region_calibration is not None:
        _append_option(
            tokens, "--score-region-calibration", score_region_calibration,
        )
    return tuple(tokens)


def build_source_spec(
    project_root: Path,
    video: Path,
    metadata: VideoMetadata,
    source_sha256: str,
    start_frame: int,
    end_frame_exclusive: int,
) -> SourceVideoSpec:
    """実際にデコードできた処理範囲を固定する。"""

    relative_video = video.resolve().relative_to(project_root.resolve()).as_posix()
    return SourceVideoSpec(
        source_video_id=video.stem,
        source_video_path=relative_video,
        source_video_sha256=source_sha256,
        width=metadata.width,
        height=metadata.height,
        frame_count=metadata.frame_count,
        time_base_numerator=metadata.time_base_numerator,
        time_base_denominator=metadata.time_base_denominator,
        processing_start_frame=start_frame,
        processing_end_frame_exclusive=end_frame_exclusive,
    )


def run_collection(
    project_root: Path,
    video: Path,
    npz_path: Path,
    log_path: Path,
    tokens: tuple[str, ...],
) -> tuple[int, int, int]:
    """収集を新規出力へ実行し、ログを残す。"""

    anomaly_path = npz_path.with_name(npz_path.stem + "_boundary_anomalies.json")
    observation_path = npz_path.with_name(npz_path.stem + EVENT_OBSERVATION_SIDECAR_SUFFIX)
    accounting_path = npz_path.with_name(npz_path.stem + EVENT_ACCOUNTING_SIDECAR_SUFFIX)
    physical_path = npz_path.with_name(npz_path.stem + EVENT_PHYSICAL_SIDECAR_SUFFIX)
    if any(path.exists() for path in (
        npz_path, log_path, anomaly_path, observation_path, accounting_path, physical_path,
    )):
        raise FileExistsError("同名の収集NPZ、ログ、観測記録が既に存在します")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.collect_boards_lean",
        "--video",
        str(video),
        "--out-npz",
        str(npz_path),
        *tokens,
    ]
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=project_root, stdout=log, stderr=subprocess.STDOUT)
    if (result.returncode != 0 or not npz_path.is_file()
            or not observation_path.is_file() or not accounting_path.is_file()
            or not physical_path.is_file()):
        raise RuntimeError(f"短区間収集に失敗しました。ログ: {log_path}")
    return _read_processed_range(log_path)


def execute(args: argparse.Namespace) -> dict[str, object]:
    """一試行の収集から完了後再検査まで実行する。"""

    total_started = time.monotonic()
    project_root = Path(__file__).resolve().parent.parent
    video = args.video.resolve()
    metadata = probe_video(video)
    per_video_config = (
        args.score_region_calibration.resolve()
        if args.score_region_calibration is not None else None
    )
    config_relative = _relative_optional(project_root, per_video_config)
    tokens = collection_tokens(args.start_sec, args.max_sec, config_relative)
    requested_start, requested_end = _requested_range(metadata, args.start_sec, args.max_sec)
    source_sha256 = file_sha256(video)
    artifacts = discover_runtime_artifacts(project_root, per_video_config)
    runtime = runtime_environment_spec()
    collection_started = time.monotonic()
    processed = run_collection(project_root, video, args.work_npz, args.log, tokens)
    actual_start, actual_end, logged_requested_end = processed
    collection_seconds = time.monotonic() - collection_started
    _validate_processed_range(
        requested_start, requested_end, actual_start, actual_end, logged_requested_end
    )
    _verify_inputs_unchanged(project_root, video, source_sha256, artifacts, per_video_config)
    source = build_source_spec(
        project_root, video, metadata, source_sha256, actual_start, actual_end
    )
    request = SnapshotExportRequest(
        args.work_npz,
        args.output_root,
        args.attempt_id,
        source,
        tokens,
        artifacts,
        runtime,
        config_relative,
        _observation_sidecar_path(args.work_npz),
        _accounting_sidecar_path(args.work_npz),
        _physical_sidecar_path(args.work_npz),
    )
    export_started = time.monotonic()
    result = export_snapshot_run(request)
    export_seconds = time.monotonic() - export_started
    return _result_payload(
        result,
        requested_end,
        actual_end,
        collection_seconds,
        export_seconds,
        time.monotonic() - total_started,
    )


def _result_payload(
    result: SnapshotExportResult,
    requested_end: int,
    actual_end: int,
    collection_seconds: float,
    export_seconds: float,
    total_seconds: float,
) -> dict[str, object]:
    return {
        "run_dir": str(result.run_dir.resolve()),
        "build_id": result.build_id,
        "event_count": result.event_count,
        "semantic_sha256": result.semantic_sha256,
        "input_npz_sha256": result.input_npz_sha256,
        "input_row_count": result.input_row_count,
        "observed_row_count": result.observed_row_count,
        "excluded_provenance_counts": dict(result.excluded_provenance_counts),
        "observation_sidecar_sha256": result.observation_sidecar_sha256,
        "boundary_event_count": result.boundary_event_count,
        "winner_observed_event_count": result.winner_observed_event_count,
        "official_game_assignment_event_count": (
            result.official_game_assignment_event_count
        ),
        "accounting_sidecar_sha256": result.accounting_sidecar_sha256,
        "accounting_inspected_side_count": result.accounting_inspected_side_count,
        "accounting_nonzero_row_count": result.accounting_nonzero_row_count,
        "accounting_event_type_counts": dict(result.accounting_event_type_counts),
        "physical_sidecar_sha256": result.physical_sidecar_sha256,
        "physical_inspected_side_count": result.physical_inspected_side_count,
        "physical_row_count": result.physical_row_count,
        "physical_event_type_counts": dict(result.physical_event_type_counts),
        "excluded_winner_result_counts": dict(result.excluded_winner_result_counts),
        "requested_end_frame_exclusive": requested_end,
        "processed_end_frame_exclusive": actual_end,
        "collection_seconds": round(collection_seconds, 6),
        "export_seconds": round(export_seconds, 6),
        "total_seconds": round(total_seconds, 6),
    }


def _number_text(value: float) -> str:
    return repr(value)


def _observation_sidecar_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + EVENT_OBSERVATION_SIDECAR_SUFFIX)


def _accounting_sidecar_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + EVENT_ACCOUNTING_SIDECAR_SUFFIX)


def _physical_sidecar_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + EVENT_PHYSICAL_SIDECAR_SUFFIX)


def _requested_range(
    metadata: VideoMetadata, start_sec: float, max_sec: float
) -> tuple[int, int]:
    start = int(start_sec * metadata.fps) if start_sec > 0 else 0
    requested = int(max_sec * metadata.fps)
    return start, min(metadata.frame_count, start + requested)


def _read_processed_range(log_path: Path) -> tuple[int, int, int]:
    matches = PROCESSED_RANGE_PATTERN.findall(log_path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise RuntimeError("収集ログから実処理フレーム範囲を一意に取得できません")
    return tuple(int(value) for value in matches[0])


def _validate_processed_range(
    requested_start: int,
    requested_end: int,
    actual_start: int,
    actual_end: int,
    logged_requested_end: int,
) -> None:
    if actual_start != requested_start or logged_requested_end != requested_end:
        raise RuntimeError("収集器と実行器の要求フレーム範囲が一致しません")
    if actual_end <= actual_start or actual_end > requested_end:
        raise RuntimeError("収集器の実処理フレーム範囲が不正です")


def runtime_environment_spec() -> RuntimeEnvironmentSpec:
    """生成結果へ影響し得るライブラリ版と計算装置を固定する。"""

    cuda_version = str(torch.version.cuda) if torch.version.cuda is not None else "none"
    device = f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    return RuntimeEnvironmentSpec(
        platform.python_implementation(),
        platform.python_version(),
        platform.platform(),
        platform.machine() or "unknown",
        str(np.__version__),
        str(cv2.__version__),
        str(torch.__version__),
        cuda_version,
        device,
        _native_puyo_core_sha256(),
    )


def _native_puyo_core_sha256() -> str:
    try:
        import puyo_core  # type: ignore[import-not-found]
    except ImportError:
        return "absent"
    module_path = getattr(puyo_core, "__file__", None)
    if not module_path or not Path(module_path).is_file():
        raise RuntimeError("puyo_coreの実体ファイルを特定できません")
    return file_sha256(Path(module_path))


def _verify_inputs_unchanged(
    project_root: Path,
    video: Path,
    source_sha256: str,
    artifacts: tuple[ArtifactHash, ...],
    per_video_config: Path | None,
) -> None:
    if file_sha256(video) != source_sha256:
        raise RuntimeError("収集中に元映像の内容が変わりました")
    after = discover_runtime_artifacts(project_root, per_video_config)
    if after != artifacts:
        raise RuntimeError("収集中に対象コード・認識資産が変わりました")


def _relative_optional(project_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _append_flag(tokens: list[str], flag: str) -> None:
    if flag not in tokens:
        tokens.append(flag)


def _append_option(tokens: list[str], name: str, value: str) -> None:
    if name in tokens:
        index = tokens.index(name)
        if index + 1 >= len(tokens) or tokens[index + 1] != value:
            raise ValueError(f"収集設定の{name}が短区間指定と競合しています")
        return
    tokens.extend((name, value))


def parse_args() -> argparse.Namespace:
    """CLI引数を読む。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--work-npz", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--max-sec", type=float, required=True)
    parser.add_argument("--score-region-calibration", type=Path)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.start_sec < 0 or args.max_sec <= 0:
        parser.error("start-secは0以上、max-secは0より大きくしてください")
    return args


def main() -> int:
    """CLIエントリポイント。"""

    args = parse_args()
    result = execute(args)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        with args.result_json.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
