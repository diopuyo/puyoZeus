"""出来事原本v1の実行説明・検査結果・完了印を確定する。"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

from src.event_source_v1 import (
    SCHEMA_VERSION,
    SEMANTIC_HASH_VERSION,
    SemanticEventHasher,
    iter_committed_batches,
    validate_part,
)


MANIFEST_FORMAT_VERSION = "event-run-manifest/1"
VALIDATION_FORMAT_VERSION = "event-run-validation/1"
COMPLETE_FORMAT_VERSION = "event-run-complete/1"
MANIFEST_NAME = "manifest.json"
VALIDATION_NAME = "validation.json"
COMPLETE_NAME = "COMPLETE"
SHA256_HEX_LENGTH = 64
FIRST_EVENT_SEQ = 0
HASH_NAME = "sha256"
FILE_HASH_CHUNK_BYTES = 1024 * 1024


class RunFinalizeError(ValueError):
    """実行成果物を安全に確定できない場合の例外。"""


@dataclass(frozen=True, slots=True)
class SourceVideoSpec:
    """処理対象となる元映像と範囲。終端フレームは含まない。"""

    source_video_id: str
    source_video_path: str
    source_video_sha256: str
    width: int
    height: int
    frame_count: int
    time_base_numerator: int
    time_base_denominator: int
    processing_start_frame: int
    processing_end_frame_exclusive: int


@dataclass(frozen=True, slots=True)
class ArtifactHash:
    """設定またはコード成果物の内容要約。"""

    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentSpec:
    """認識結果へ影響し得る実行環境の版と計算装置。"""

    python_implementation: str
    python_version: str
    platform: str
    machine: str
    numpy_version: str
    opencv_version: str
    torch_version: str
    torch_cuda_version: str
    compute_device: str
    native_puyo_core_sha256: str


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    """中断部品の隔離・正常接頭辞・再開位置。"""

    source_part: str
    quarantine_copy: str
    recovered_part: str | None
    prior_adopted_part: str | None
    continuation_part: str | None
    valid_prefix_bytes: int
    resumed_from_seq: int


@dataclass(frozen=True, slots=True)
class RunFinalizeSpec:
    """一つの生成試行を確定するための不変情報。"""

    build_id: str
    attempt_id: str
    source: SourceVideoSpec
    recognition_config: ArtifactHash
    code_artifacts: tuple[ArtifactHash, ...]
    sync_every_batches: int
    max_unsynced_batches: int
    max_unsynced_events: int
    heavy_evidence_policy: str
    runtime: RuntimeEnvironmentSpec | None = None
    recovery_records: tuple[RecoveryRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class PartSummary:
    """検査済みJSON Lines部品の説明。"""

    relative_path: str
    byte_count: int
    batch_count: int
    event_count: int
    first_seq: int
    last_seq: int
    physical_sha256: str
    semantic_sha256: str


@dataclass(frozen=True, slots=True)
class FinalizedRun:
    """確定済み実行の成果物位置と要約値。"""

    manifest_path: Path
    validation_path: Path
    complete_path: Path
    event_count: int
    semantic_sha256: str


@dataclass(frozen=True, slots=True)
class CompletedRunValidation:
    """完了印を含む実行成果物の再検査結果。"""

    valid: bool
    event_count: int
    semantic_sha256: str
    error: str | None = None


@dataclass(slots=True)
class _RunSequenceState:
    expected_seq: int = FIRST_EVENT_SEQ
    last_available_frame: int = -1
    last_available_ms: int = -1
    batch_ids: set[str] = field(default_factory=set)
    semantic_hasher: SemanticEventHasher = field(default_factory=SemanticEventHasher)

    def add_batch(self, batch: Any, spec: RunFinalizeSpec) -> None:
        first = batch.events[0]
        _validate_event_identity(first, spec)
        frame = int(first["timing"]["available_frame"])
        milliseconds = int(first["timing"]["available_ms"])
        if frame <= self.last_available_frame or milliseconds < self.last_available_ms:
            raise RunFinalizeError("部品をまたいで公開位置が逆行または重複しています")
        if batch.batch_id in self.batch_ids:
            raise RunFinalizeError("部品をまたいで一括更新IDが重複しています")
        for event in batch.events:
            _validate_event_identity(event, spec)
            self.semantic_hasher.add(event)
        self.batch_ids.add(batch.batch_id)
        self.expected_seq = int(batch.events[-1]["seq"]) + 1
        self.last_available_frame = frame
        self.last_available_ms = milliseconds


def finalize_event_run(
    run_dir: Path,
    part_paths: Sequence[Path],
    spec: RunFinalizeSpec,
) -> FinalizedRun:
    """全部品を検査し、説明・検査・完了印をこの順で確定する。"""

    _validate_finalize_spec(spec)
    _reject_incomplete_existing_complete(run_dir)
    summaries, state = _summarize_parts(run_dir, part_paths, spec)
    _validate_recovery_files(run_dir, spec, summaries)
    validation = _build_validation(summaries, state)
    validation_bytes = _json_bytes(validation)
    validation_sha256 = _sha256(validation_bytes)
    manifest = _build_manifest(spec, summaries, state, validation_sha256)
    manifest_bytes = _json_bytes(manifest)
    complete_bytes = _build_complete_bytes(manifest_bytes, validation_bytes)
    validation_path = _ensure_atomic_content(run_dir / VALIDATION_NAME, validation_bytes)
    manifest_path = _ensure_atomic_content(run_dir / MANIFEST_NAME, manifest_bytes)
    complete_path = _ensure_atomic_content(run_dir / COMPLETE_NAME, complete_bytes)
    return FinalizedRun(
        manifest_path,
        validation_path,
        complete_path,
        state.semantic_hasher.event_count,
        state.semantic_hasher.hexdigest(),
    )


def derive_build_id(
    source: SourceVideoSpec,
    recognition_config: ArtifactHash,
    code_artifacts: Sequence[ArtifactHash],
    runtime: RuntimeEnvironmentSpec | None = None,
) -> str:
    """仕様版・映像内容・設定・コード・処理範囲から生成内容IDを作る。"""

    _validate_source(source)
    _validate_artifact(recognition_config)
    if not code_artifacts:
        raise RunFinalizeError("対象コード一覧は空にできません")
    for artifact in code_artifacts:
        _validate_artifact(artifact)
    if runtime is not None:
        _validate_runtime(runtime)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_video_sha256": source.source_video_sha256,
        "time_base_numerator": source.time_base_numerator,
        "time_base_denominator": source.time_base_denominator,
        "processing_start_frame": source.processing_start_frame,
        "processing_end_frame_exclusive": source.processing_end_frame_exclusive,
        "recognition_config_sha256": recognition_config.sha256,
        "code_artifacts": [
            asdict(item) for item in sorted(code_artifacts, key=lambda value: value.relative_path)
        ],
        "runtime": asdict(runtime) if runtime is not None else {},
    }
    return f"build-{_sha256(_json_bytes(identity))}"


def validate_completed_run(run_dir: Path) -> CompletedRunValidation:
    """完了印、説明、全部品を再計算し、採用可能かを返す。"""

    try:
        event_count, semantic_sha256 = _validate_completed_run(run_dir)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return CompletedRunValidation(False, 0, "", str(exc))
    return CompletedRunValidation(True, event_count, semantic_sha256)


def _validate_finalize_spec(spec: RunFinalizeSpec) -> None:
    _require_text(spec.build_id, "build_id")
    _require_text(spec.attempt_id, "attempt_id")
    _validate_source(spec.source)
    _validate_artifact(spec.recognition_config)
    if not spec.code_artifacts:
        raise RunFinalizeError("対象コード一覧は空にできません")
    for artifact in spec.code_artifacts:
        _validate_artifact(artifact)
    if spec.runtime is not None:
        _validate_runtime(spec.runtime)
    artifact_paths = [artifact.relative_path for artifact in spec.code_artifacts]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise RunFinalizeError("対象コード一覧のパスが重複しています")
    if spec.sync_every_batches <= 0 or spec.max_unsynced_batches < 0:
        raise RunFinalizeError("同期設定が不正です")
    if spec.max_unsynced_events < 0:
        raise RunFinalizeError("失われ得る最大行数が不正です")
    _require_text(spec.heavy_evidence_policy, "heavy_evidence_policy")
    for record in spec.recovery_records:
        _validate_recovery_record(record)


def _validate_recovery_record(record: RecoveryRecord) -> None:
    _validate_relative_path(record.source_part, "source_part")
    _validate_relative_path(record.quarantine_copy, "quarantine_copy")
    if record.recovered_part is not None:
        _validate_relative_path(record.recovered_part, "recovered_part")
    if record.prior_adopted_part is not None:
        _validate_relative_path(record.prior_adopted_part, "prior_adopted_part")
    if record.continuation_part is not None:
        _validate_relative_path(record.continuation_part, "continuation_part")
    if not _is_nonnegative_int(record.valid_prefix_bytes):
        raise RunFinalizeError("valid_prefix_bytes が不正です")
    if not _is_nonnegative_int(record.resumed_from_seq):
        raise RunFinalizeError("resumed_from_seq が不正です")


def _validate_source(source: SourceVideoSpec) -> None:
    _require_text(source.source_video_id, "source_video_id")
    _require_text(source.source_video_path, "source_video_path")
    _require_sha256(source.source_video_sha256, "source_video_sha256")
    positive = (source.width, source.height, source.frame_count, source.time_base_denominator)
    if any(not _is_positive_int(value) for value in positive):
        raise RunFinalizeError("映像寸法・フレーム数・時間基準が不正です")
    if not _is_positive_int(source.time_base_numerator):
        raise RunFinalizeError("時間基準の分子は1以上でなければなりません")
    start = source.processing_start_frame
    end = source.processing_end_frame_exclusive
    if not _is_nonnegative_int(start) or not _is_nonnegative_int(end):
        raise RunFinalizeError("処理フレーム範囲が不正です")
    if end <= start or end > source.frame_count:
        raise RunFinalizeError("処理フレーム範囲が不正です")


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_artifact(artifact: ArtifactHash) -> None:
    _validate_relative_path(artifact.relative_path, "relative_path")
    _require_sha256(artifact.sha256, f"{artifact.relative_path}.sha256")


def _validate_runtime(runtime: RuntimeEnvironmentSpec) -> None:
    for name, value in asdict(runtime).items():
        _require_text(value, f"runtime.{name}")


def _validate_relative_path(value: str, name: str) -> None:
    _require_text(value, name)
    path = Path(value)
    windows_path = PureWindowsPath(value)
    if value == "." or "\\" in value:
        raise RunFinalizeError(f"{name} は正規化済み相対パスでなければなりません")
    if path.is_absolute() or windows_path.is_absolute() or ".." in path.parts:
        raise RunFinalizeError(f"{name} は親参照のない相対パスでなければなりません")


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunFinalizeError(f"{name} は空でない文字列でなければなりません")
    return value


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH:
        raise RunFinalizeError(f"{name} はSHA-256形式ではありません")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RunFinalizeError(f"{name} はSHA-256形式ではありません") from exc
    if value != value.lower():
        raise RunFinalizeError(f"{name} は小文字SHA-256形式でなければなりません")


def _reject_incomplete_existing_complete(run_dir: Path) -> None:
    complete = run_dir / COMPLETE_NAME
    if not complete.exists():
        return
    if not (run_dir / MANIFEST_NAME).is_file() or not (run_dir / VALIDATION_NAME).is_file():
        raise RunFinalizeError("完了印だけが存在する不完全な実行ディレクトリです")


def _summarize_parts(
    run_dir: Path, part_paths: Sequence[Path], spec: RunFinalizeSpec
) -> tuple[list[PartSummary], _RunSequenceState]:
    if not part_paths:
        raise RunFinalizeError("出来事部品がありません")
    paths = sorted(part_paths, key=lambda path: _relative_part_path(run_dir, path))
    resolved_paths = [path.resolve() for path in paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise RunFinalizeError("同じ出来事部品が重複指定されています")
    state = _RunSequenceState()
    summaries: list[PartSummary] = []
    for path in paths:
        summaries.append(_summarize_part(run_dir, path, spec, state))
    return summaries, state


def _summarize_part(
    run_dir: Path, path: Path, spec: RunFinalizeSpec, state: _RunSequenceState
) -> PartSummary:
    relative_path = _relative_part_path(run_dir, path)
    validation = validate_part(path, expected_first_seq=state.expected_seq)
    if not validation.valid:
        raise RunFinalizeError(f"出来事部品が不正です: {relative_path}: {validation.error}")
    for batch in iter_committed_batches(path, expected_first_seq=state.expected_seq):
        state.add_batch(batch, spec)
    if validation.first_seq is None or validation.last_seq is None:
        raise RunFinalizeError("空の出来事部品は確定できません")
    return PartSummary(
        relative_path,
        path.stat().st_size,
        validation.batch_count,
        validation.event_count,
        validation.first_seq,
        validation.last_seq,
        validation.physical_sha256,
        validation.semantic_sha256,
    )


def _relative_part_path(run_dir: Path, path: Path) -> str:
    resolved_root = run_dir.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RunFinalizeError("出来事部品は実行ディレクトリ内になければなりません") from exc
    if not relative.parts or relative.parts[0] != "events":
        raise RunFinalizeError("出来事部品はeventsディレクトリ内になければなりません")
    return relative.as_posix()


def _validate_event_identity(event: Mapping[str, Any], spec: RunFinalizeSpec) -> None:
    expected = (spec.source.source_video_id, spec.build_id, spec.attempt_id)
    actual = (event["source_video_id"], event["build_id"], event["attempt_id"])
    if actual != expected:
        raise RunFinalizeError("出来事の元映像・生成内容・実行試行が説明と一致しません")


def _build_validation(
    summaries: Sequence[PartSummary], state: _RunSequenceState
) -> dict[str, Any]:
    event_count = sum(summary.event_count for summary in summaries)
    return {
        "format_version": VALIDATION_FORMAT_VERSION,
        "valid": True,
        "checks": [
            {"check_id": "all_parts_valid", "result": "pass", "part_count": len(summaries)},
            {"check_id": "event_seq_contiguous", "result": "pass", "event_count": event_count},
            {"check_id": "run_identity_consistent", "result": "pass", "event_count": event_count},
            {"check_id": "publication_ordered", "result": "pass", "batch_count": len(state.batch_ids)},
        ],
        "part_count": len(summaries),
        "batch_count": len(state.batch_ids),
        "event_count": event_count,
        "semantic_hash_version": SEMANTIC_HASH_VERSION,
        "semantic_sha256": state.semantic_hasher.hexdigest(),
    }


def _build_manifest(
    spec: RunFinalizeSpec,
    summaries: Sequence[PartSummary],
    state: _RunSequenceState,
    validation_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "build_id": spec.build_id,
        "attempt_id": spec.attempt_id,
        "source": asdict(spec.source),
        "recognition_config": asdict(spec.recognition_config),
        "runtime": asdict(spec.runtime) if spec.runtime is not None else {},
        "code_artifact_count": len(spec.code_artifacts),
        "code_artifacts": [asdict(item) for item in sorted(spec.code_artifacts, key=lambda x: x.relative_path)],
        "durability": _durability_dict(spec),
        "parts": [asdict(summary) for summary in summaries],
        "adopted_event_count": state.semantic_hasher.event_count,
        "semantic_hash_version": SEMANTIC_HASH_VERSION,
        "semantic_sha256": state.semantic_hasher.hexdigest(),
        "heavy_evidence_policy": spec.heavy_evidence_policy,
        "recovery_records": [asdict(record) for record in spec.recovery_records],
        "validation": {"relative_path": VALIDATION_NAME, "sha256": validation_sha256},
    }


def _durability_dict(spec: RunFinalizeSpec) -> dict[str, int]:
    return {
        "sync_every_batches": spec.sync_every_batches,
        "max_unsynced_batches": spec.max_unsynced_batches,
        "max_unsynced_events": spec.max_unsynced_events,
    }


def _build_complete_bytes(manifest_bytes: bytes, validation_bytes: bytes) -> bytes:
    return _json_bytes(
        {
            "format_version": COMPLETE_FORMAT_VERSION,
            "manifest_sha256": _sha256(manifest_bytes),
            "validation_sha256": _sha256(validation_bytes),
        }
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _sha256(payload: bytes) -> str:
    return hashlib.new(HASH_NAME, payload).hexdigest()


def _validate_completed_run(run_dir: Path) -> tuple[int, str]:
    complete, complete_bytes = _read_canonical_object(run_dir / COMPLETE_NAME)
    manifest, manifest_bytes = _read_canonical_object(run_dir / MANIFEST_NAME)
    validation, validation_bytes = _read_canonical_object(run_dir / VALIDATION_NAME)
    _validate_completion_hashes(complete, manifest_bytes, validation_bytes)
    spec = _spec_from_manifest(manifest)
    _validate_finalize_spec(spec)
    paths = _part_paths_from_manifest(run_dir, manifest)
    summaries, state = _summarize_parts(run_dir, paths, spec)
    _validate_recovery_files(run_dir, spec, summaries)
    expected_validation = _build_validation(summaries, state)
    if validation != expected_validation:
        raise RunFinalizeError("validation.json を再計算できません")
    expected_manifest = _build_manifest(spec, summaries, state, _sha256(validation_bytes))
    if manifest != expected_manifest:
        raise RunFinalizeError("manifest.json を再計算できません")
    return state.semantic_hasher.event_count, state.semantic_hasher.hexdigest()


def _read_canonical_object(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RunFinalizeError(f"{path.name} はJSONオブジェクトではありません")
    if payload != _json_bytes(value):
        raise RunFinalizeError(f"{path.name} が正規化形式ではありません")
    return value, payload


def _validate_completion_hashes(
    complete: Mapping[str, Any], manifest_bytes: bytes, validation_bytes: bytes
) -> None:
    expected = json.loads(_build_complete_bytes(manifest_bytes, validation_bytes))
    if complete != expected:
        raise RunFinalizeError("完了印をmanifestとvalidationから再計算できません")


def _spec_from_manifest(manifest: Mapping[str, Any]) -> RunFinalizeSpec:
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise RunFinalizeError("manifest形式版が一致しません")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RunFinalizeError("出来事仕様版が一致しません")
    durability = manifest["durability"]
    runtime_value = manifest.get("runtime", {})
    runtime = RuntimeEnvironmentSpec(**runtime_value) if runtime_value else None
    code_artifacts = tuple(ArtifactHash(**item) for item in manifest["code_artifacts"])
    if manifest.get("code_artifact_count") != len(code_artifacts):
        raise RunFinalizeError("対象コード・認識資産の件数が一致しません")
    return RunFinalizeSpec(
        build_id=manifest["build_id"],
        attempt_id=manifest["attempt_id"],
        source=SourceVideoSpec(**manifest["source"]),
        recognition_config=ArtifactHash(**manifest["recognition_config"]),
        code_artifacts=code_artifacts,
        sync_every_batches=durability["sync_every_batches"],
        max_unsynced_batches=durability["max_unsynced_batches"],
        max_unsynced_events=durability["max_unsynced_events"],
        heavy_evidence_policy=manifest["heavy_evidence_policy"],
        runtime=runtime,
        recovery_records=tuple(RecoveryRecord(**item) for item in manifest["recovery_records"]),
    )


def _part_paths_from_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> list[Path]:
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise RunFinalizeError("manifestに出来事部品がありません")
    paths: list[Path] = []
    for item in parts:
        if not isinstance(item, Mapping):
            raise RunFinalizeError("manifestの部品説明が不正です")
        relative_path = item.get("relative_path")
        _validate_relative_path(relative_path, "part.relative_path")
        paths.append(run_dir / relative_path)
    return paths


def _validate_recovery_files(
    run_dir: Path, spec: RunFinalizeSpec, summaries: Sequence[PartSummary]
) -> None:
    summary_map = {summary.relative_path: summary for summary in summaries}
    ordered_paths = [summary.relative_path for summary in summaries]
    for record in spec.recovery_records:
        source = _resolve_run_relative(run_dir, record.source_part)
        quarantine = _resolve_run_relative(run_dir, record.quarantine_copy)
        if not source.is_file() or not quarantine.is_file():
            raise RunFinalizeError("中断元または隔離コピーがありません")
        if source.stat().st_size != quarantine.stat().st_size:
            raise RunFinalizeError("中断元と隔離コピーの大きさが一致しません")
        if _file_sha256(source) != _file_sha256(quarantine):
            raise RunFinalizeError("中断元と隔離コピーの内容が一致しません")
        source_validation = validate_part(source)
        if source_validation.valid:
            raise RunFinalizeError("中断元が正常部品であり復旧記録と矛盾します")
        if source_validation.valid_prefix_bytes != record.valid_prefix_bytes:
            raise RunFinalizeError("中断元の正常接頭辞長が復旧記録と一致しません")
        _validate_recovery_links(run_dir, record, summary_map, ordered_paths, source)


def _validate_recovery_links(
    run_dir: Path,
    record: RecoveryRecord,
    summaries: Mapping[str, PartSummary],
    ordered_paths: Sequence[str],
    source: Path,
) -> None:
    if record.recovered_part is None:
        _validate_total_loss_links(record, summaries, ordered_paths)
        return
    recovered = _resolve_run_relative(run_dir, record.recovered_part)
    summary = summaries.get(record.recovered_part)
    if summary is None or not recovered.is_file():
        raise RunFinalizeError("復旧部品が採用部品一覧にありません")
    if recovered.stat().st_size != record.valid_prefix_bytes:
        raise RunFinalizeError("復旧部品の大きさが説明と一致しません")
    if not _prefix_matches(source, recovered, record.valid_prefix_bytes):
        raise RunFinalizeError("復旧部品が中断元の正常接頭辞と一致しません")
    if summary.last_seq + 1 != record.resumed_from_seq:
        raise RunFinalizeError("復旧後の再開通番が一致しません")
    _validate_adopted_neighbors(record, ordered_paths, record.recovered_part)


def _validate_total_loss_links(
    record: RecoveryRecord,
    summaries: Mapping[str, PartSummary],
    ordered_paths: Sequence[str],
) -> None:
    if record.valid_prefix_bytes != 0:
        raise RunFinalizeError("復旧部品なしなのに正常接頭辞が存在します")
    if record.prior_adopted_part is None:
        expected_seq = FIRST_EVENT_SEQ
        expected_continuation = ordered_paths[0] if ordered_paths else None
    else:
        prior = summaries.get(record.prior_adopted_part)
        if prior is None:
            raise RunFinalizeError("復旧直前の採用部品が存在しません")
        expected_seq = prior.last_seq + 1
        expected_continuation = _next_path(ordered_paths, record.prior_adopted_part)
    if record.resumed_from_seq != expected_seq:
        raise RunFinalizeError("全損部品後の再開通番が一致しません")
    if record.continuation_part != expected_continuation:
        raise RunFinalizeError("全損部品後の再開先が採用部品順と一致しません")


def _validate_adopted_neighbors(
    record: RecoveryRecord, ordered_paths: Sequence[str], recovered_part: str
) -> None:
    index = ordered_paths.index(recovered_part)
    expected_prior = ordered_paths[index - 1] if index > 0 else None
    expected_continuation = ordered_paths[index + 1] if index + 1 < len(ordered_paths) else None
    if record.prior_adopted_part != expected_prior:
        raise RunFinalizeError("復旧部品の直前部品が採用順と一致しません")
    if record.continuation_part != expected_continuation:
        raise RunFinalizeError("復旧後の再開先が採用順と一致しません")


def _next_path(ordered_paths: Sequence[str], current: str) -> str | None:
    index = ordered_paths.index(current)
    return ordered_paths[index + 1] if index + 1 < len(ordered_paths) else None


def _prefix_matches(source: Path, recovered: Path, size: int) -> bool:
    with source.open("rb") as source_handle, recovered.open("rb") as recovered_handle:
        remaining = size
        while remaining:
            chunk_size = min(remaining, FILE_HASH_CHUNK_BYTES)
            if source_handle.read(chunk_size) != recovered_handle.read(chunk_size):
                return False
            remaining -= chunk_size
    return True


def _resolve_run_relative(run_dir: Path, relative_path: str) -> Path:
    root = run_dir.resolve()
    path = (run_dir / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RunFinalizeError("実行説明の相対パスが実行範囲外です") from exc
    return path


def _file_sha256(path: Path) -> str:
    hasher = hashlib.new(HASH_NAME)
    with path.open("rb") as handle:
        while chunk := handle.read(FILE_HASH_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def _ensure_atomic_content(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return path
        raise RunFinalizeError(f"既存成果物と内容が一致しません: {path.name}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_synced(temporary, payload)
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        if not path.is_file() or path.read_bytes() != payload:
            raise RunFinalizeError(f"成果物の同時作成が競合しました: {path.name}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_synced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        written = handle.write(payload)
        if written != len(payload):
            raise RunFinalizeError("成果物を完全に書き込めませんでした")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ArtifactHash",
    "CompletedRunValidation",
    "FinalizedRun",
    "RecoveryRecord",
    "RunFinalizeError",
    "RunFinalizeSpec",
    "RuntimeEnvironmentSpec",
    "SourceVideoSpec",
    "finalize_event_run",
    "derive_build_id",
    "validate_completed_run",
]
