"""確定盤面NPZを自己完結した出来事原本の実行成果物へ確定する。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP
from pathlib import Path
from typing import Any

from src.event_accounting_adapter_v1 import (
    ACCOUNTING_ADAPTER_VERSION,
    EventAccountingSidecar,
    build_event_accounting_events,
    parse_event_accounting_sidecar_bytes,
)
from src.event_observation_adapter_v1 import (
    EventObservationSidecar,
    OBSERVATION_ADAPTER_VERSION,
    OFFICIAL_GAME_ASSIGNMENT_VERSION,
    build_event_observation_events,
    build_official_game_assignment_events,
    merge_and_resequence_batches,
    parse_event_observation_sidecar_bytes,
)
from src.event_physical_adapter_v1 import (
    PHYSICAL_ADAPTER_VERSION,
    EventPhysicalSidecar,
    build_event_physical_events,
    parse_event_physical_sidecar_bytes,
)
from src.event_run_v1 import (
    ArtifactHash,
    FinalizedRun,
    RunFinalizeSpec,
    RuntimeEnvironmentSpec,
    SourceVideoSpec,
    derive_build_id,
    finalize_event_run,
    validate_completed_run,
)
from src.event_snapshot_adapter_v1 import (
    ADAPTER_VERSION,
    VideoTimeBase,
    build_stable_board_batches,
    load_stable_snapshot_selection,
)
from src.event_source_v1 import EventPartWriter


CONFIG_FORMAT_VERSION = "event-snapshot-recognition-config/4"
RUN_SCHEMA_DIRECTORY = "schema=v1"
EVENT_PART_NAME = "part-00000.jsonl"
CONFIG_NAME = "recognition-config.json"
HASH_CHUNK_BYTES = 1024 * 1024
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
REQUIRED_RUNTIME_RELATIVE_PATHS = (
    "scripts/collect_boards_lean.py",
    "src/production_config.py",
    "src/recognition_pipeline.py",
    "models/cnn_phase_b_large_v2.pt",
)


class SnapshotExportError(ValueError):
    """安全に実行成果物を確定できない場合の例外。"""


@dataclass(frozen=True, slots=True)
class SnapshotExportRequest:
    """一つの確定盤面収集結果を確定する入力。"""

    npz_path: Path
    output_root: Path
    attempt_id: str
    source: SourceVideoSpec
    collection_tokens: tuple[str, ...]
    code_artifacts: tuple[ArtifactHash, ...]
    runtime: RuntimeEnvironmentSpec | None = None
    per_video_config_relative_path: str | None = None
    event_observation_sidecar_path: Path | None = None
    event_accounting_sidecar_path: Path | None = None
    event_physical_sidecar_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SnapshotExportResult:
    """確定した実行成果物の場所と意味要約。"""

    run_dir: Path
    build_id: str
    event_count: int
    semantic_sha256: str
    input_npz_sha256: str
    input_row_count: int
    observed_row_count: int
    excluded_provenance_counts: tuple[tuple[str, int], ...]
    observation_sidecar_sha256: str | None = None
    boundary_event_count: int = 0
    winner_observed_event_count: int = 0
    official_game_assignment_event_count: int = 0
    excluded_winner_result_counts: tuple[tuple[str, int], ...] = ()
    accounting_sidecar_sha256: str | None = None
    accounting_inspected_side_count: int = 0
    accounting_nonzero_row_count: int = 0
    accounting_event_type_counts: tuple[tuple[str, int], ...] = ()
    physical_sidecar_sha256: str | None = None
    physical_inspected_side_count: int = 0
    physical_row_count: int = 0
    physical_event_type_counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _ObservationInput:
    """同一バイト列から固定した観測サイドカーと内容要約。"""

    payload: bytes
    sha256: str
    observation: EventObservationSidecar


@dataclass(frozen=True, slots=True)
class _AccountingInput:
    """同一バイト列から固定した会計サイドカーと内容要約。"""

    payload: bytes
    sha256: str
    accounting: EventAccountingSidecar


@dataclass(frozen=True, slots=True)
class _PhysicalInput:
    """同一バイト列から固定した物理観測サイドカーと内容要約。"""

    payload: bytes
    sha256: str
    physical: EventPhysicalSidecar


def export_snapshot_run(request: SnapshotExportRequest) -> SnapshotExportResult:
    """NPZを変換し、説明・検査・完了印まで新規作成する。"""

    _validate_request(request)
    selection = load_stable_snapshot_selection(request.npz_path)
    _validate_snapshot_range(selection.snapshots, request.source)
    observation_input = _load_observation_input(request.event_observation_sidecar_path)
    accounting_input = _load_accounting_input(request.event_accounting_sidecar_path)
    physical_input = _load_physical_input(request.event_physical_sidecar_path)
    _validate_observation_range(observation_input, request.source)
    _validate_accounting_range(accounting_input, request.source)
    _validate_physical_range(physical_input, request.source)
    config_bytes = _recognition_config_bytes(
        request, observation_input, accounting_input, physical_input,
    )
    config = ArtifactHash(CONFIG_NAME, bytes_sha256(config_bytes))
    build_id = derive_build_id(
        request.source, config, request.code_artifacts, request.runtime
    )
    run_dir = _run_directory(request, build_id)
    _reserve_run_directory(run_dir)
    _write_exclusive(run_dir / CONFIG_NAME, config_bytes)
    finalized = _write_and_finalize(
        run_dir, request, selection.snapshots, build_id, config,
        observation_input, accounting_input, physical_input,
    )
    _verify_export(run_dir, finalized, config)
    return SnapshotExportResult(
        run_dir,
        build_id,
        finalized.event_count,
        finalized.semantic_sha256,
        file_sha256(request.npz_path),
        selection.input_row_count,
        selection.observed_row_count,
        selection.excluded_provenance_counts,
        observation_input.sha256 if observation_input is not None else None,
        len(observation_input.observation.boundaries) if observation_input else 0,
        _winner_observed_count(observation_input),
        _official_assignment_count(observation_input, request.source),
        _excluded_winner_counts(observation_input),
        accounting_input.sha256 if accounting_input is not None else None,
        _accounting_inspected_count(accounting_input),
        _accounting_row_count(accounting_input),
        _accounting_event_counts(accounting_input, request, build_id),
        physical_input.sha256 if physical_input is not None else None,
        _physical_inspected_count(physical_input),
        _physical_row_count(physical_input),
        _physical_event_counts(physical_input, request, build_id),
    )


def recognition_config_bytes(request: SnapshotExportRequest) -> bytes:
    """収集時に実際に渡した設定を正規化JSONへ固定する。"""

    observation_input = _load_observation_input(request.event_observation_sidecar_path)
    accounting_input = _load_accounting_input(request.event_accounting_sidecar_path)
    physical_input = _load_physical_input(request.event_physical_sidecar_path)
    return _recognition_config_bytes(
        request, observation_input, accounting_input, physical_input,
    )


def _recognition_config_bytes(
    request: SnapshotExportRequest,
    observation_input: _ObservationInput | None,
    accounting_input: _AccountingInput | None,
    physical_input: _PhysicalInput | None,
) -> bytes:
    value = {
        "format_version": CONFIG_FORMAT_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "event_observation_adapter_version": OBSERVATION_ADAPTER_VERSION,
        "official_game_assignment_version": OFFICIAL_GAME_ASSIGNMENT_VERSION,
        "event_accounting_adapter_version": ACCOUNTING_ADAPTER_VERSION,
        "event_physical_adapter_version": PHYSICAL_ADAPTER_VERSION,
        "source_video_id": request.source.source_video_id,
        "processing_start_frame": request.source.processing_start_frame,
        "processing_end_frame_exclusive": request.source.processing_end_frame_exclusive,
        "time_base_numerator": request.source.time_base_numerator,
        "time_base_denominator": request.source.time_base_denominator,
        "collection_tokens": list(request.collection_tokens),
        "per_video_config_relative_path": (
            request.per_video_config_relative_path or "none"
        ),
        "snapshot_event_type": "stable_board_observed",
        "snapshot_filter": {"board_provenance": "observed"},
        "event_observation_input": _observation_config_value(observation_input),
        "event_accounting_input": _accounting_config_value(accounting_input),
        "event_physical_input": _physical_config_value(physical_input),
    }
    return canonical_json_bytes(value)


def discover_runtime_artifacts(
    project_root: Path, per_video_config_path: Path | None = None
) -> tuple[ArtifactHash, ...]:
    """短区間試行で固定するコード・モデル・テンプレートを列挙する。"""

    _validate_required_runtime_paths(project_root)
    candidates = list((project_root / "src").rglob("*.py"))
    candidates.append(project_root / "scripts" / "collect_boards_lean.py")
    candidates.extend(_recognition_asset_paths(project_root))
    if per_video_config_path is not None:
        if not per_video_config_path.is_file():
            raise SnapshotExportError("明示された動画別認識設定がありません")
        candidates.append(per_video_config_path)
    existing = sorted({path.resolve() for path in candidates if path.is_file()})
    return tuple(_artifact_hash(project_root, path) for path in existing)


def file_sha256(path: Path) -> str:
    """ファイル内容のSHA-256を計算する。"""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    """バイト列のSHA-256を計算する。"""

    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    """設定用の正規化JSONバイト列を作る。"""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _validate_request(request: SnapshotExportRequest) -> None:
    if not request.npz_path.is_file():
        raise SnapshotExportError("入力NPZがありません")
    _validate_safe_id(request.attempt_id, "実行試行ID")
    _validate_safe_id(request.source.source_video_id, "元映像ID")
    if not request.collection_tokens:
        raise SnapshotExportError("収集設定は空にできません")
    if not request.code_artifacts:
        raise SnapshotExportError("対象コード・認識資産は空にできません")
    if request.runtime is None:
        raise SnapshotExportError("実行環境の版と計算装置を省略できません")
    if request.per_video_config_relative_path is not None:
        _validate_relative_text(request.per_video_config_relative_path)
    if (
        request.event_observation_sidecar_path is not None
        and not request.event_observation_sidecar_path.is_file()
    ):
        raise SnapshotExportError("境界・勝敗観測サイドカーがありません")
    if (
        request.event_accounting_sidecar_path is not None
        and not request.event_accounting_sidecar_path.is_file()
    ):
        raise SnapshotExportError("会計観測サイドカーがありません")
    if (
        request.event_physical_sidecar_path is not None
        and not request.event_physical_sidecar_path.is_file()
    ):
        raise SnapshotExportError("物理観測サイドカーがありません")


def _validate_safe_id(value: str, name: str) -> None:
    if not isinstance(value, str) or SAFE_ID_PATTERN.fullmatch(value) is None:
        raise SnapshotExportError(f"{name}にパス区切りまたは許可外文字を使えません")


def _validate_relative_text(value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise SnapshotExportError("動画別認識設定は正規化済み相対パスで指定します")


def _validate_snapshot_range(snapshots: tuple[Any, ...], source: SourceVideoSpec) -> None:
    if not snapshots:
        raise SnapshotExportError("実観測の確定盤面がありません")
    identities = {snapshot.source_video_id for snapshot in snapshots}
    if identities != {source.source_video_id}:
        raise SnapshotExportError("NPZの元映像IDが説明と一致しません")
    start = source.processing_start_frame
    end = source.processing_end_frame_exclusive
    if any(snapshot.frame_idx < start or snapshot.frame_idx >= end for snapshot in snapshots):
        raise SnapshotExportError("NPZに処理範囲外の盤面があります")


def _load_observation_input(path: Path | None) -> _ObservationInput | None:
    if path is None:
        return None
    try:
        payload = path.read_bytes()
        observation = parse_event_observation_sidecar_bytes(payload)
    except (OSError, ValueError) as error:
        raise SnapshotExportError(f"境界・勝敗観測サイドカーが不正です: {error}") from error
    return _ObservationInput(payload, bytes_sha256(payload), observation)


def _load_accounting_input(path: Path | None) -> _AccountingInput | None:
    if path is None:
        return None
    try:
        payload = path.read_bytes()
        accounting = parse_event_accounting_sidecar_bytes(payload)
    except (OSError, ValueError) as error:
        raise SnapshotExportError(f"会計観測サイドカーが不正です: {error}") from error
    return _AccountingInput(payload, bytes_sha256(payload), accounting)


def _load_physical_input(path: Path | None) -> _PhysicalInput | None:
    if path is None:
        return None
    try:
        payload = path.read_bytes()
        physical = parse_event_physical_sidecar_bytes(payload)
    except (OSError, ValueError) as error:
        raise SnapshotExportError(f"物理観測サイドカーが不正です: {error}") from error
    return _PhysicalInput(payload, bytes_sha256(payload), physical)


def _validate_observation_range(
    observation_input: _ObservationInput | None,
    source: SourceVideoSpec,
) -> None:
    if observation_input is None:
        return
    time_base = VideoTimeBase(source.time_base_numerator, source.time_base_denominator)
    observation = observation_input.observation
    start = _seconds_to_frame(observation.processing_start_sec, time_base)
    end = _seconds_to_frame(observation.processing_end_sec, time_base)
    if start != source.processing_start_frame:
        raise SnapshotExportError("観測サイドカーの処理開始位置が元映像説明と一致しません")
    end_gap = source.processing_end_frame_exclusive - end
    if end_gap < 0 or end_gap > 2:
        raise SnapshotExportError("観測サイドカーの処理終了位置が元映像説明と一致しません")


def _validate_accounting_range(
    accounting_input: _AccountingInput | None,
    source: SourceVideoSpec,
) -> None:
    if accounting_input is None:
        return
    accounting = accounting_input.accounting
    if accounting.processing_start_frame != source.processing_start_frame:
        raise SnapshotExportError("会計サイドカーの処理開始位置が元映像説明と一致しません")
    if accounting.processing_end_frame_exclusive != source.processing_end_frame_exclusive:
        raise SnapshotExportError("会計サイドカーの処理終了位置が元映像説明と一致しません")


def _validate_physical_range(
    physical_input: _PhysicalInput | None,
    source: SourceVideoSpec,
) -> None:
    if physical_input is None:
        return
    physical = physical_input.physical
    if physical.processing_start_frame != source.processing_start_frame:
        raise SnapshotExportError("物理観測の処理開始位置が元映像説明と一致しません")
    if physical.processing_end_frame_exclusive != source.processing_end_frame_exclusive:
        raise SnapshotExportError("物理観測の処理終了位置が元映像説明と一致しません")


def _seconds_to_frame(seconds: Any, time_base: VideoTimeBase) -> int:
    numerator = seconds * time_base.denominator
    return int(
        (numerator / time_base.numerator).to_integral_value(rounding=ROUND_HALF_UP)
    )


def _observation_config_value(
    observation_input: _ObservationInput | None,
) -> dict[str, str] | str:
    if observation_input is None:
        return "none"
    return {
        "format": "embedded-utf8-json",
        "sha256": observation_input.sha256,
        "content_utf8": observation_input.payload.decode("utf-8"),
    }


def _accounting_config_value(
    accounting_input: _AccountingInput | None,
) -> dict[str, str] | str:
    if accounting_input is None:
        return "none"
    return {
        "format": "embedded-utf8-json",
        "sha256": accounting_input.sha256,
        "content_utf8": accounting_input.payload.decode("utf-8"),
    }


def _physical_config_value(
    physical_input: _PhysicalInput | None,
) -> dict[str, str] | str:
    if physical_input is None:
        return "none"
    return {
        "format": "embedded-utf8-json",
        "sha256": physical_input.sha256,
        "content_utf8": physical_input.payload.decode("utf-8"),
    }


def _winner_observed_count(observation_input: _ObservationInput | None) -> int:
    if observation_input is None:
        return 0
    return sum(
        item.result == "winner_observed"
        for item in observation_input.observation.winner_results
    )


def _official_assignment_count(
    observation_input: _ObservationInput | None,
    source: SourceVideoSpec,
) -> int:
    if observation_input is None or source.processing_start_frame != 0:
        return 0
    return _winner_observed_count(observation_input)


def _excluded_winner_counts(
    observation_input: _ObservationInput | None,
) -> tuple[tuple[str, int], ...]:
    if observation_input is None:
        return ()
    counts: dict[str, int] = {}
    for item in observation_input.observation.winner_results:
        if item.result != "winner_observed":
            counts[item.result] = counts.get(item.result, 0) + 1
    return tuple(sorted(counts.items()))


def _accounting_inspected_count(accounting_input: _AccountingInput | None) -> int:
    return 0 if accounting_input is None else accounting_input.accounting.inspected_side_count


def _accounting_row_count(accounting_input: _AccountingInput | None) -> int:
    return 0 if accounting_input is None else len(accounting_input.accounting.rows)


def _accounting_event_counts(
    accounting_input: _AccountingInput | None,
    request: SnapshotExportRequest,
    build_id: str,
) -> tuple[tuple[str, int], ...]:
    if accounting_input is None:
        return ()
    time_base = VideoTimeBase(
        request.source.time_base_numerator, request.source.time_base_denominator,
    )
    events = build_event_accounting_events(
        accounting_input.accounting, source_video_id=request.source.source_video_id,
        build_id=build_id, attempt_id=request.attempt_id, time_base=time_base,
    )
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event["event_type"])
        counts[event_type] = counts.get(event_type, 0) + 1
    return tuple(sorted(counts.items()))


def _physical_inspected_count(physical_input: _PhysicalInput | None) -> int:
    return 0 if physical_input is None else physical_input.physical.inspected_side_count


def _physical_row_count(physical_input: _PhysicalInput | None) -> int:
    return 0 if physical_input is None else len(physical_input.physical.rows)


def _physical_event_counts(
    physical_input: _PhysicalInput | None,
    request: SnapshotExportRequest,
    build_id: str,
) -> tuple[tuple[str, int], ...]:
    if physical_input is None:
        return ()
    time_base = VideoTimeBase(
        request.source.time_base_numerator, request.source.time_base_denominator,
    )
    events = build_event_physical_events(
        physical_input.physical, source_video_id=request.source.source_video_id,
        build_id=build_id, attempt_id=request.attempt_id, time_base=time_base,
    )
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event["event_type"])
        counts[event_type] = counts.get(event_type, 0) + 1
    return tuple(sorted(counts.items()))


def _run_directory(request: SnapshotExportRequest, build_id: str) -> Path:
    return (
        request.output_root
        / RUN_SCHEMA_DIRECTORY
        / f"video={request.source.source_video_id}"
        / f"build={build_id}"
        / f"attempt={request.attempt_id}"
    )


def _reserve_run_directory(run_dir: Path) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SnapshotExportError("同じ実行試行ディレクトリが既に存在します") from exc


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        written = handle.write(payload)
        if written != len(payload):
            raise SnapshotExportError("設定ファイルを完全に書き込めませんでした")
        handle.flush()
        os.fsync(handle.fileno())


def _write_and_finalize(
    run_dir: Path,
    request: SnapshotExportRequest,
    snapshots: tuple[Any, ...],
    build_id: str,
    config: ArtifactHash,
    observation_input: _ObservationInput | None,
    accounting_input: _AccountingInput | None,
    physical_input: _PhysicalInput | None,
) -> FinalizedRun:
    batches = _build_export_batches(
        request, snapshots, build_id, observation_input, accounting_input, physical_input,
    )
    part = run_dir / "events" / EVENT_PART_NAME
    with EventPartWriter(part) as writer:
        for batch in batches:
            writer.append_batch(batch)
    spec = _finalize_spec(request, build_id, config)
    return finalize_event_run(run_dir, [part], spec)


def _build_export_batches(
    request: SnapshotExportRequest,
    snapshots: tuple[Any, ...],
    build_id: str,
    observation_input: _ObservationInput | None,
    accounting_input: _AccountingInput | None,
    physical_input: _PhysicalInput | None,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    time_base = VideoTimeBase(
        request.source.time_base_numerator,
        request.source.time_base_denominator,
    )
    batches = build_stable_board_batches(
        snapshots,
        build_id=build_id,
        attempt_id=request.attempt_id,
        time_base=time_base,
        occurrence_start_frame=request.source.processing_start_frame,
    )
    batches = _merge_accounting_batches(
        batches, request, build_id, time_base, accounting_input,
    )
    batches = _merge_physical_batches(
        batches, request, build_id, time_base, physical_input,
    )
    return _merge_observation_batches(
        batches, request, build_id, time_base, observation_input,
    )


def _merge_accounting_batches(
    batches: tuple[tuple[dict[str, Any], ...], ...], request: SnapshotExportRequest,
    build_id: str, time_base: VideoTimeBase,
    accounting_input: _AccountingInput | None,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    if accounting_input is None:
        return batches
    events = build_event_accounting_events(
        accounting_input.accounting, source_video_id=request.source.source_video_id,
        build_id=build_id, attempt_id=request.attempt_id, time_base=time_base,
    )
    return merge_and_resequence_batches(
        batches, events, build_id=build_id, attempt_id=request.attempt_id,
    )


def _merge_physical_batches(
    batches: tuple[tuple[dict[str, Any], ...], ...], request: SnapshotExportRequest,
    build_id: str, time_base: VideoTimeBase,
    physical_input: _PhysicalInput | None,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    if physical_input is None:
        return batches
    events = build_event_physical_events(
        physical_input.physical, source_video_id=request.source.source_video_id,
        build_id=build_id, attempt_id=request.attempt_id, time_base=time_base,
    )
    return merge_and_resequence_batches(
        batches, events, build_id=build_id, attempt_id=request.attempt_id,
    )


def _merge_observation_batches(
    batches: tuple[tuple[dict[str, Any], ...], ...], request: SnapshotExportRequest,
    build_id: str, time_base: VideoTimeBase,
    observation_input: _ObservationInput | None,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    if observation_input is None:
        return batches
    events = build_event_observation_events(
        observation_input.observation, source_video_id=request.source.source_video_id,
        build_id=build_id, attempt_id=request.attempt_id, time_base=time_base,
        processing_start_frame=request.source.processing_start_frame,
        processing_end_frame_exclusive=request.source.processing_end_frame_exclusive,
    )
    batches = merge_and_resequence_batches(
        batches, events, build_id=build_id, attempt_id=request.attempt_id,
    )
    assignments = build_official_game_assignment_events(
        batches, build_id=build_id, attempt_id=request.attempt_id,
        time_base=time_base,
        processing_start_frame=request.source.processing_start_frame,
        processing_end_frame_exclusive=request.source.processing_end_frame_exclusive,
    )
    return merge_and_resequence_batches(
        batches, assignments, build_id=build_id, attempt_id=request.attempt_id,
    )


def _finalize_spec(
    request: SnapshotExportRequest, build_id: str, config: ArtifactHash
) -> RunFinalizeSpec:
    return RunFinalizeSpec(
        build_id=build_id,
        attempt_id=request.attempt_id,
        source=request.source,
        recognition_config=config,
        code_artifacts=request.code_artifacts,
        sync_every_batches=1,
        max_unsynced_batches=0,
        max_unsynced_events=0,
        heavy_evidence_policy="hash-only",
        runtime=request.runtime,
    )


def _verify_export(run_dir: Path, result: FinalizedRun, config: ArtifactHash) -> None:
    validation = validate_completed_run(run_dir)
    if not validation.valid:
        raise SnapshotExportError(f"完了後の再検査に失敗しました: {validation.error}")
    if validation.event_count != result.event_count:
        raise SnapshotExportError("完了後の出来事件数が一致しません")
    if validation.semantic_sha256 != result.semantic_sha256:
        raise SnapshotExportError("完了後の意味内容要約値が一致しません")
    if file_sha256(run_dir / CONFIG_NAME) != config.sha256:
        raise SnapshotExportError("収集設定の内容要約値が一致しません")


def _recognition_asset_paths(project_root: Path) -> list[Path]:
    models_root = project_root / "models"
    return list(models_root.rglob("*")) if models_root.is_dir() else []


def _validate_required_runtime_paths(project_root: Path) -> None:
    missing = [
        relative
        for relative in REQUIRED_RUNTIME_RELATIVE_PATHS
        if not (project_root / relative).is_file()
    ]
    if missing:
        raise SnapshotExportError(f"必須の対象コード・認識資産がありません: {missing}")


def _artifact_hash(project_root: Path, path: Path) -> ArtifactHash:
    try:
        relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise SnapshotExportError("対象コード・認識資産がプロジェクト外にあります") from exc
    return ArtifactHash(relative, file_sha256(path))


__all__ = [
    "SnapshotExportError",
    "SnapshotExportRequest",
    "SnapshotExportResult",
    "canonical_json_bytes",
    "discover_runtime_artifacts",
    "export_snapshot_run",
    "file_sha256",
    "recognition_config_bytes",
]
