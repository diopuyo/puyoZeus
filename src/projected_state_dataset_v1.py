"""ProjectedStateObservationV1を上書き不能なJSONLへ保存・検証する。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from src.projected_state_observation_v1 import (
    PROJECTED_STATE_SCHEMA_VERSION,
    DualBoardState,
    IntegerQuantity,
    LandingBranch,
    ProjectedBoardState,
    ProjectedStateObservationV1,
    ProjectedStateReason,
    RootFamilyReference,
)


DATASET_FORMAT = "projected-state-dataset-v1"
OBSERVATIONS_NAME = "observations.jsonl"
MANIFEST_NAME = "manifest.json"
COMPLETE_NAME = "COMPLETE"
UNIQUE_INDEX_NAME = "uniqueness.sqlite3"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ProjectedStateDatasetError(ValueError):
    """datasetが完全性または正式契約を満たさない。"""


@dataclass(frozen=True, slots=True)
class ProjectedStateDatasetReceipt:
    """確定datasetの読取に必要な最小証跡。"""

    directory: Path
    record_count: int
    source_count: int
    observations_sha256: str
    manifest_sha256: str


def write_projected_state_dataset(
    directory: Path,
    observations: Iterable[ProjectedStateObservationV1],
    *,
    asset_sha256: Mapping[str, str],
    runtime_environment: Mapping[str, Any],
    before_complete: Callable[[], None] | None = None,
) -> ProjectedStateDatasetReceipt:
    """観測を新規directoryへ逐次保存し、最後にCOMPLETEを発行する。"""

    assets = _validated_assets(asset_sha256)
    runtime = _validated_json_mapping(runtime_environment, "実行環境")
    if before_complete is not None and not callable(before_complete):
        raise ProjectedStateDatasetError("確定直前検査はcallableでなければなりません")
    _reserve_directory(directory)
    data_path = directory / OBSERVATIONS_NAME
    index_path = directory / UNIQUE_INDEX_NAME
    count, sources, gates, digest = _write_observations(
        data_path, index_path, observations,
    )
    if count == 0:
        raise ProjectedStateDatasetError("0件datasetは確定できません")
    manifest = _manifest(
        count, sources, gates, digest, _sha_file(index_path), assets, runtime,
    )
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    _write_exclusive(directory / MANIFEST_NAME, manifest_bytes)
    persisted_manifest = _read_unpublished_manifest(directory, manifest_sha)
    payload_fingerprints = _payload_fingerprints(directory)
    _scan_and_validate(directory, persisted_manifest)
    if _payload_fingerprints(directory) != payload_fingerprints:
        raise ProjectedStateDatasetError("全量検査中にdataset payloadが変化しました")
    if before_complete is not None:
        before_complete()
        persisted_manifest = _read_unpublished_manifest(directory, manifest_sha)
        if _payload_fingerprints(directory) != payload_fingerprints:
            raise ProjectedStateDatasetError("確定直前にdataset payloadが変化しました")
    complete = _complete_document(manifest_sha, persisted_manifest)
    _write_exclusive(directory / COMPLETE_NAME, _canonical_json_bytes(complete))
    return _receipt(directory, manifest)


def _payload_fingerprints(directory: Path) -> dict[str, tuple[int, ...]]:
    """全量検査後のpayload同一性を確定直前にO(1)で再確認する。"""

    result: dict[str, tuple[int, ...]] = {}
    for name in (OBSERVATIONS_NAME, UNIQUE_INDEX_NAME):
        try:
            stat = (directory / name).stat()
        except OSError as exc:
            raise ProjectedStateDatasetError(f"必須fileを読めません: {name}") from exc
        result[name] = (
            stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
        )
    return result


def validate_projected_state_dataset(directory: Path) -> ProjectedStateDatasetReceipt:
    """COMPLETEから全ファイルと全DTOを読み直して検証する。"""

    complete = _read_canonical_json(directory / COMPLETE_NAME)
    manifest_bytes = _read_bytes(directory / MANIFEST_NAME)
    manifest = _decode_json_object(manifest_bytes, "manifest")
    if manifest_bytes != _canonical_json_bytes(manifest):
        raise ProjectedStateDatasetError("manifestがcanonical JSONではありません")
    _validate_complete(complete, manifest_bytes, manifest)
    _scan_and_validate(directory, manifest)
    return _receipt(directory, manifest)


def iter_projected_state_dataset(
    directory: Path,
    *,
    _manifest: Mapping[str, Any] | None = None,
) -> Iterator[ProjectedStateObservationV1]:
    """hashとcanonical表現を検査しながらDTOを逐次返す。"""

    manifest = dict(_manifest) if _manifest is not None else _validated_manifest(directory)
    if _manifest is None:
        _scan_and_validate(directory, manifest)
    yield from _iter_verified_records(directory, manifest)


def observation_from_dict(value: Mapping[str, Any]) -> ProjectedStateObservationV1:
    """canonical JSON objectを検証付きDTOへ復元する。"""

    try:
        return _observation_from_dict_unchecked(value)
    except ProjectedStateDatasetError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ProjectedStateDatasetError("観測DTOを復元できません") from exc


def _observation_from_dict_unchecked(
    value: Mapping[str, Any],
) -> ProjectedStateObservationV1:
    data = dict(value)
    if data.pop("contract_version", None) != PROJECTED_STATE_SCHEMA_VERSION:
        raise ProjectedStateDatasetError("観測contract versionが一致しません")
    data["reason_codes"] = tuple(ProjectedStateReason(v) for v in data["reason_codes"])
    for name in (
        "prefire_incoming_amount", "signed_projected_balance",
        "first_drop_amount", "leftover_after_first_drop",
    ):
        data[name] = IntegerQuantity(**data[name])
    data["current_state"] = _dual_board(data["current_state"])
    data["post_chain_state"] = _dual_board(data["post_chain_state"])
    data["p1_root_family"] = _root_family(data.get("p1_root_family"))
    data["p2_root_family"] = _root_family(data.get("p2_root_family"))
    data["landing_branches"] = tuple(
        _landing_branch(item) for item in data["landing_branches"]
    )
    return ProjectedStateObservationV1(**data)


def _write_observations(
    path: Path, index_path: Path,
    observations: Iterable[ProjectedStateObservationV1],
) -> tuple[int, set[str], Counter[str], str]:
    hasher = hashlib.sha256()
    sources: set[str] = set()
    gates: Counter[str] = Counter()
    count = 0
    connection = _create_unique_index(index_path)
    try:
        with path.open("xb") as handle:
            for value in observations:
                _require_observation(value)
                _insert_unique_index(connection, count + 1, value)
                payload = value.canonical_json_bytes()
                handle.write(payload)
                hasher.update(payload)
                sources.add(value.source_video_id)
                gates[value.gate_status] += 1
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        connection.commit()
    finally:
        connection.close()
    return count, sources, gates, hasher.hexdigest()


def _create_unique_index(path: Path) -> sqlite3.Connection:
    try:
        path.open("xb").close()
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE records ("
            "seq_no INTEGER PRIMARY KEY, observation_id TEXT NOT NULL UNIQUE, "
            "input_digest TEXT NOT NULL UNIQUE)"
        )
        return connection
    except (OSError, sqlite3.Error) as exc:
        raise ProjectedStateDatasetError("unique indexを作成できません") from exc


def _insert_unique_index(
    connection: sqlite3.Connection, seq_no: int,
    value: ProjectedStateObservationV1,
) -> None:
    try:
        connection.execute(
            "INSERT INTO records(seq_no, observation_id, input_digest) VALUES (?, ?, ?)",
            (seq_no, value.observation_id, value.input_digest),
        )
    except sqlite3.IntegrityError as exc:
        raise ProjectedStateDatasetError(
            "observation IDまたはinput digestが重複しています"
        ) from exc


def _require_observation(value: object) -> None:
    if not isinstance(value, ProjectedStateObservationV1):
        raise ProjectedStateDatasetError("観測列に正式DTO以外があります")


def _scan_and_validate(directory: Path, manifest: Mapping[str, Any]) -> None:
    _validate_manifest_contract(manifest)
    index_path = directory / UNIQUE_INDEX_NAME
    if _safe_sha_file(index_path) != manifest["unique_index_sha256"]:
        raise ProjectedStateDatasetError("unique index SHA-256が一致しません")
    connection = _open_unique_index(index_path)
    try:
        summary, digest = _scan_observation_file(
            directory / OBSERVATIONS_NAME, connection,
        )
        if digest != manifest["observations_sha256"]:
            raise ProjectedStateDatasetError("observations SHA-256が一致しません")
        _validate_observed_summary(manifest, summary)
        _validate_index_count(connection, summary[0])
    finally:
        connection.close()


def _scan_observation_file(
    path: Path, connection: sqlite3.Connection,
) -> tuple[tuple[int, set[str], Counter[str]], str]:
    hasher = hashlib.sha256()
    count = 0
    sources: set[str] = set()
    gates: Counter[str] = Counter()
    try:
        with path.open("rb") as handle:
            for count, line in enumerate(handle, start=1):
                hasher.update(line)
                value = _decode_observation_line(line, count)
                _verify_index_record(connection, count, value)
                sources.add(value.source_video_id)
                gates[value.gate_status] += 1
    except OSError as exc:
        raise ProjectedStateDatasetError("observations fileを読めません") from exc
    return (count, sources, gates), hasher.hexdigest()


def _iter_verified_records(
    directory: Path, manifest: Mapping[str, Any],
) -> Iterator[ProjectedStateObservationV1]:
    _validate_manifest_contract(manifest)
    index_path = directory / UNIQUE_INDEX_NAME
    if _safe_sha_file(index_path) != manifest["unique_index_sha256"]:
        raise ProjectedStateDatasetError("unique index SHA-256が一致しません")
    connection = _open_unique_index(index_path)
    try:
        yield from _iter_verified_open(directory, manifest, connection)
    finally:
        connection.close()


def _iter_verified_open(
    directory: Path, manifest: Mapping[str, Any], connection: sqlite3.Connection,
) -> Iterator[ProjectedStateObservationV1]:
    hasher = hashlib.sha256()
    summary: tuple[int, set[str], Counter[str]] = (0, set(), Counter())
    try:
        with (directory / OBSERVATIONS_NAME).open("rb") as handle:
            for seq_no, line in enumerate(handle, start=1):
                hasher.update(line)
                value = _decode_observation_line(line, seq_no)
                _verify_index_record(connection, seq_no, value)
                summary = _extend_summary(summary, value)
                yield value
    except OSError as exc:
        raise ProjectedStateDatasetError("observations fileを読めません") from exc
    if hasher.hexdigest() != manifest["observations_sha256"]:
        raise ProjectedStateDatasetError("observations SHA-256が一致しません")
    _validate_observed_summary(manifest, summary)
    _validate_index_count(connection, summary[0])


def _extend_summary(
    summary: tuple[int, set[str], Counter[str]], value: ProjectedStateObservationV1,
) -> tuple[int, set[str], Counter[str]]:
    count, sources, gates = summary
    sources.add(value.source_video_id)
    gates[value.gate_status] += 1
    return count + 1, sources, gates


def _open_unique_index(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        columns = connection.execute("PRAGMA table_info(records)").fetchall()
        expected = [
            (0, "seq_no", "INTEGER", 0, None, 1),
            (1, "observation_id", "TEXT", 1, None, 0),
            (2, "input_digest", "TEXT", 1, None, 0),
        ]
        if columns != expected:
            raise ProjectedStateDatasetError("unique index schemaが一致しません")
        return connection
    except (OSError, sqlite3.Error) as exc:
        raise ProjectedStateDatasetError("unique indexを読めません") from exc


def _verify_index_record(
    connection: sqlite3.Connection, seq_no: int,
    value: ProjectedStateObservationV1,
) -> None:
    row = connection.execute(
        "SELECT observation_id, input_digest FROM records WHERE seq_no = ?", (seq_no,),
    ).fetchone()
    if row != (value.observation_id, value.input_digest):
        raise ProjectedStateDatasetError("unique indexと観測行が一致しません")


def _validate_index_count(connection: sqlite3.Connection, expected: int) -> None:
    row = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT observation_id), "
        "COUNT(DISTINCT input_digest) FROM records"
    ).fetchone()
    if row is None or row != (expected, expected, expected):
        raise ProjectedStateDatasetError("unique indexと観測件数が一致しません")


def _validate_observed_summary(
    manifest: Mapping[str, Any],
    summary: tuple[int, set[str], Counter[str]],
) -> None:
    count, sources, gates = summary
    expected = (
        manifest.get("record_count"), manifest.get("source_count"),
        manifest.get("source_video_ids"), manifest.get("gate_status_counts"),
    )
    actual = (count, len(sources), sorted(sources), dict(sorted(gates.items())))
    if expected != actual or count <= 0:
        raise ProjectedStateDatasetError("manifestと観測内容の集計が一致しません")
    _validated_assets(manifest.get("asset_sha256", {}))
    _validated_json_mapping(manifest.get("runtime_environment", {}), "実行環境")


def _manifest(
    count: int, sources: set[str], gates: Counter[str], observations_sha: str,
    index_sha: str,
    assets: Mapping[str, str], runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": DATASET_FORMAT,
        "schema_version": PROJECTED_STATE_SCHEMA_VERSION,
        "record_count": count,
        "source_count": len(sources),
        "source_video_ids": sorted(sources),
        "gate_status_counts": dict(sorted(gates.items())),
        "observations_file": OBSERVATIONS_NAME,
        "observations_sha256": observations_sha,
        "unique_index_file": UNIQUE_INDEX_NAME,
        "unique_index_sha256": index_sha,
        "asset_sha256": dict(assets),
        "runtime_environment": dict(runtime),
    }


def _complete_document(manifest_sha: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": f"{DATASET_FORMAT}-complete/1",
        "manifest_sha256": manifest_sha,
        "observations_sha256": manifest["observations_sha256"],
        "unique_index_sha256": manifest["unique_index_sha256"],
        "record_count": manifest["record_count"],
    }


def _read_unpublished_manifest(
    directory: Path, expected_sha256: str,
) -> dict[str, Any]:
    payload = _read_bytes(directory / MANIFEST_NAME)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ProjectedStateDatasetError("保存後manifest SHA-256が一致しません")
    manifest = _decode_json_object(payload, "manifest")
    if payload != _canonical_json_bytes(manifest):
        raise ProjectedStateDatasetError("保存後manifestがcanonical JSONではありません")
    _validate_manifest_contract(manifest)
    return manifest


def _validated_manifest(directory: Path) -> dict[str, Any]:
    complete = _read_canonical_json(directory / COMPLETE_NAME)
    manifest_bytes = _read_bytes(directory / MANIFEST_NAME)
    manifest = _decode_json_object(manifest_bytes, "manifest")
    if manifest_bytes != _canonical_json_bytes(manifest):
        raise ProjectedStateDatasetError("manifestがcanonical JSONではありません")
    _validate_complete(complete, manifest_bytes, manifest)
    return manifest


def _validate_complete(
    complete: Mapping[str, Any], manifest_bytes: bytes, manifest: Mapping[str, Any],
) -> None:
    expected_fields = {
        "format", "manifest_sha256", "observations_sha256",
        "unique_index_sha256", "record_count",
    }
    if set(complete) != expected_fields:
        raise ProjectedStateDatasetError("COMPLETE field集合が一致しません")
    if complete.get("format") != f"{DATASET_FORMAT}-complete/1":
        raise ProjectedStateDatasetError("COMPLETE formatが一致しません")
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if complete.get("manifest_sha256") != actual_manifest_sha:
        raise ProjectedStateDatasetError("manifest SHA-256が一致しません")
    _validate_manifest_contract(manifest)
    for field in ("record_count", "observations_sha256", "unique_index_sha256"):
        if complete.get(field) != manifest.get(field):
            raise ProjectedStateDatasetError(f"COMPLETEの{field}が一致しません")


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    expected_fields = {
        "format", "schema_version", "record_count", "source_count",
        "source_video_ids", "gate_status_counts", "observations_file",
        "observations_sha256", "unique_index_file", "unique_index_sha256",
        "asset_sha256", "runtime_environment",
    }
    if set(manifest) != expected_fields:
        raise ProjectedStateDatasetError("manifest field集合が一致しません")
    if manifest.get("format") != DATASET_FORMAT:
        raise ProjectedStateDatasetError("dataset formatが一致しません")
    if manifest.get("schema_version") != PROJECTED_STATE_SCHEMA_VERSION:
        raise ProjectedStateDatasetError("dataset schema versionが一致しません")
    _validate_manifest_files(manifest)
    _validate_manifest_counts(manifest)


def _validate_manifest_files(manifest: Mapping[str, Any]) -> None:
    if manifest.get("observations_file") != OBSERVATIONS_NAME:
        raise ProjectedStateDatasetError("observations file名が一致しません")
    if manifest.get("unique_index_file") != UNIQUE_INDEX_NAME:
        raise ProjectedStateDatasetError("unique index file名が一致しません")
    for field in ("observations_sha256", "unique_index_sha256"):
        digest = manifest.get(field)
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ProjectedStateDatasetError(f"manifestの{field}が不正です")


def _validate_manifest_counts(manifest: Mapping[str, Any]) -> None:
    count = manifest.get("record_count")
    source_count = manifest.get("source_count")
    sources = manifest.get("source_video_ids")
    gates = manifest.get("gate_status_counts")
    if type(count) is not int or count <= 0:
        raise ProjectedStateDatasetError("manifest record countが不正です")
    if type(source_count) is not int or source_count <= 0:
        raise ProjectedStateDatasetError("manifest source countが不正です")
    if not isinstance(sources, list) or sources != sorted(set(sources)):
        raise ProjectedStateDatasetError("manifest source一覧が不正です")
    if source_count != len(sources) or any(not isinstance(v, str) or not v for v in sources):
        raise ProjectedStateDatasetError("manifest source集計が不正です")
    if not isinstance(gates, dict):
        raise ProjectedStateDatasetError("manifest gate集計が不正です")
    if any(k not in {"guaranteed", "not_applicable", "untrusted"} for k in gates):
        raise ProjectedStateDatasetError("manifest gate名が不正です")
    if any(type(v) is not int or v < 0 for v in gates.values()):
        raise ProjectedStateDatasetError("manifest gate件数が不正です")
    if sum(gates.values()) != count:
        raise ProjectedStateDatasetError("manifest gate集計が不正です")


def _decode_observation_line(line: bytes, line_number: int) -> ProjectedStateObservationV1:
    if not line.endswith(b"\n") or line in {b"", b"\n"}:
        raise ProjectedStateDatasetError(f"観測行{line_number}がcanonical行ではありません")
    decoded = _decode_json_object(line, f"観測行{line_number}")
    value = observation_from_dict(decoded)
    if value.canonical_json_bytes() != line:
        raise ProjectedStateDatasetError(f"観測行{line_number}の表現がcanonicalではありません")
    return value


def _projected_board(value: Mapping[str, Any]) -> ProjectedBoardState:
    grid = value.get("grid")
    mask = value.get("unknown_mask")
    return ProjectedBoardState(
        None if grid is None else tuple(tuple(int(cell) for cell in row) for row in grid),
        None if mask is None else tuple(tuple(bool(cell) for cell in row) for row in mask),
        value["provenance"], value.get("source_event_id"), value["present"],
    )


def _dual_board(value: Mapping[str, Any]) -> DualBoardState:
    return DualBoardState(_projected_board(value["p1"]), _projected_board(value["p2"]))


def _root_family(value: Mapping[str, Any] | None) -> RootFamilyReference | None:
    if value is None:
        return None
    return RootFamilyReference(
        value["side"], value["game_idx"], value["root_anchor"],
        tuple(value["prefire_anchor_event_ids"]),
    )


def _landing_branch(value: Mapping[str, Any]) -> LandingBranch:
    return LandingBranch(tuple(value["remainder_columns"]), _dual_board(value["boards"]))


def _validated_assets(value: Mapping[str, str]) -> dict[str, str]:
    if not value:
        raise ProjectedStateDatasetError("asset SHA-256は空にできません")
    result: dict[str, str] = {}
    for path, digest in value.items():
        if not isinstance(path, str) or not path or path in result:
            raise ProjectedStateDatasetError("asset pathが不正または重複しています")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ProjectedStateDatasetError("asset SHA-256が不正です")
        result[path] = digest
    return dict(sorted(result.items()))


def _validated_json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not value:
        raise ProjectedStateDatasetError(f"{label}は空にできません")
    try:
        payload = _canonical_json_bytes(dict(value))
        decoded = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ProjectedStateDatasetError(f"{label}がcanonical JSONではありません") from exc
    return dict(decoded)


def _reserve_directory(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ProjectedStateDatasetError("出力directoryは既に存在します") from exc


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        if handle.write(payload) != len(payload):
            raise ProjectedStateDatasetError("dataset fileを完全に保存できません")
        handle.flush()
        os.fsync(handle.fileno())


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ProjectedStateDatasetError(f"必須fileを読めません: {path.name}") from exc


def _read_canonical_json(path: Path) -> dict[str, Any]:
    payload = _read_bytes(path)
    value = _decode_json_object(payload, path.name)
    if payload != _canonical_json_bytes(value):
        raise ProjectedStateDatasetError(f"{path.name}がcanonical JSONではありません")
    return value


def _decode_json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectedStateDatasetError(f"{label}がJSONではありません") from exc
    if not isinstance(value, dict):
        raise ProjectedStateDatasetError(f"{label}はJSON object必須です")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_sha_file(path: Path) -> str:
    try:
        return _sha_file(path)
    except OSError as exc:
        raise ProjectedStateDatasetError(f"必須fileを読めません: {path.name}") from exc


def _receipt(
    directory: Path, manifest: Mapping[str, Any],
) -> ProjectedStateDatasetReceipt:
    return ProjectedStateDatasetReceipt(
        directory=directory,
        record_count=int(manifest["record_count"]),
        source_count=int(manifest["source_count"]),
        observations_sha256=str(manifest["observations_sha256"]),
        manifest_sha256=_sha_file(directory / MANIFEST_NAME),
    )
