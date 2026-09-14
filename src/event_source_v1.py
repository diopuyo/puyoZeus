"""出来事原本 v1 の最小書込み・読取り・完全性検査。

本モジュールは既存認識経路へ接続しない。追記専用 JSON Lines の物理形式と、
実行試行に依存しない意味内容要約値だけを先に固定する。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO


SCHEMA_VERSION = "event-source/1.0"
LOG_FORMAT_VERSION = "event-source-log/1"
SEMANTIC_HASH_VERSION = "semantic-hash/v1"
HASH_NAME = "sha256"
LENGTH_PREFIX_BYTES = 8
DEFAULT_SYNC_EVERY_BATCHES = 1
MAX_BATCH_EVENTS = 10_000
COPY_CHUNK_BYTES = 1024 * 1024

RECORD_BATCH_BEGIN = "batch_begin"
RECORD_EVENT = "event"
RECORD_BATCH_COMMIT = "batch_commit"

ALLOWED_SIDES = frozenset({"p1", "p2", "both", "system", "unknown"})
ALLOWED_ASSERTION_STATES = frozenset({"provisional", "confirmed", "unknown", "conflict"})
ALLOWED_VALUE_FORMS = frozenset({"exact", "range", "candidates", "unknown", "conflict"})
NON_SEMANTIC_KEYS = frozenset(
    {
        "attempt_id",
        "storage_path",
        "absolute_path",
        "file_updated_at",
        "run_started_at",
        "run_finished_at",
        "processing_time_ms",
    }
)
# 重い証拠の保存先などはネスト位置が一定でないため、意味対象外キーを全階層で除く。
# 同名の意味項目を将来追加せず、必要なら別名と仕様版を与える。
UNORDERED_LIST_KEYS = frozenset(
    {
        "evidence",
        "checks",
        "missing_information",
        "heavy_evidence_refs",
        "verified_cause_event_ids",
        "inferred_cause_event_ids",
        "garbage_lot_ids",
        "target_event_ids",
        "reason_codes",
    }
)
REQUIRED_EVENT_KEYS = frozenset(
    {
        "record_kind",
        "schema_version",
        "source_video_id",
        "build_id",
        "attempt_id",
        "seq",
        "event_id",
        "availability_batch_id",
        "batch_index",
        "batch_size",
        "event_type",
        "side",
        "timing",
        "assertion",
        "evidence",
        "checks",
        "missing_information",
        "relations",
        "heavy_evidence_refs",
        "payload",
    }
)
REQUIRED_TIMING_KEYS = frozenset(
    {
        "occurred_earliest_frame",
        "occurred_earliest_ms",
        "occurred_latest_frame",
        "occurred_latest_ms",
        "available_frame",
        "available_ms",
    }
)


class EventSourceError(ValueError):
    """出来事原本の形式・完全性違反。"""


class PartReadError(EventSourceError):
    """部品読取り中の最初の不整合。"""

    def __init__(self, message: str, line_no: int, valid_prefix_bytes: int) -> None:
        super().__init__(message)
        self.line_no = line_no
        self.valid_prefix_bytes = valid_prefix_bytes


@dataclass(frozen=True, slots=True)
class CommittedBatch:
    """開始・出来事・確定が一致した一括更新。"""

    batch_id: str
    events: tuple[dict[str, Any], ...]
    start_offset: int
    end_offset: int
    semantic_sha256: str


@dataclass(frozen=True, slots=True)
class PartValidation:
    """JSON Lines 部品の検査結果。"""

    valid: bool
    batch_count: int
    event_count: int
    first_seq: int | None
    last_seq: int | None
    valid_prefix_bytes: int
    physical_sha256: str
    semantic_sha256: str
    error: str | None = None
    error_line: int | None = None


class SemanticEventHasher:
    """部品境界に依存せず、採用出来事列の意味内容を逐次要約する。"""

    def __init__(self) -> None:
        self._hasher = hashlib.new(HASH_NAME)
        self.event_count = 0

    def add(self, event: Mapping[str, Any]) -> None:
        _update_length_prefixed(self._hasher, semantic_event_bytes(event))
        self.event_count += 1

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()

    def copy(self) -> SemanticEventHasher:
        """現在のprefixを共有せず複製する。"""
        copied = SemanticEventHasher()
        copied._hasher = self._hasher.copy()
        copied.event_count = self.event_count
        return copied


def _require_nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventSourceError(f"{name} は空でない文字列でなければならない")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EventSourceError(f"{name} は0以上の整数でなければならない")
    return value


def _normalize_semantic(value: Any, parent_key: str = "") -> Any:
    if value is None or isinstance(value, float):
        raise EventSourceError("意味項目では null と浮動小数を使用できない")
    if isinstance(value, Mapping):
        return _normalize_mapping(value)
    if isinstance(value, list):
        return _normalize_list(value, parent_key)
    if isinstance(value, (str, int, bool)):
        return value
    raise EventSourceError(f"JSONで扱えない型です: {type(value).__name__}")


def _normalize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if any(not isinstance(key, str) for key in value):
        raise EventSourceError("JSONオブジェクトのキーは文字列でなければならない")
    for key in sorted(value):
        if key in NON_SEMANTIC_KEYS:
            continue
        normalized[key] = _normalize_semantic(value[key], key)
    return normalized


def _normalize_list(value: list[Any], parent_key: str) -> list[Any]:
    normalized = [_normalize_semantic(item) for item in value]
    if parent_key not in UNORDERED_LIST_KEYS:
        return normalized
    return sorted(normalized, key=_canonical_json_bytes)


def _canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def _physical_json_line(value: Mapping[str, Any]) -> bytes:
    _normalize_semantic(value)
    return _canonical_json_bytes(value) + b"\n"


def semantic_event_bytes(event: Mapping[str, Any]) -> bytes:
    """実行試行固有情報を除いた正規化済み出来事バイト列。"""

    validate_event(event)
    normalized = _normalize_semantic(event)
    return _canonical_json_bytes(normalized)


def _update_length_prefixed(hasher: Any, payload: bytes) -> None:
    hasher.update(len(payload).to_bytes(LENGTH_PREFIX_BYTES, "big"))
    hasher.update(payload)


def _hash_payloads(payloads: Sequence[bytes]) -> str:
    hasher = hashlib.new(HASH_NAME)
    for payload in payloads:
        _update_length_prefixed(hasher, payload)
    return hasher.hexdigest()


def semantic_events_sha256(events: Sequence[Mapping[str, Any]]) -> str:
    """通番順の出来事列から意味内容要約値を作る。"""

    ordered = sorted(events, key=lambda item: int(item["seq"]))
    payloads = [semantic_event_bytes(event) for event in ordered]
    return _hash_payloads(payloads)


def _event_ids_sha256(events: Sequence[Mapping[str, Any]]) -> str:
    payloads = [str(event["event_id"]).encode("utf-8") for event in events]
    return _hash_payloads(payloads)


def _validate_timing(timing: Any) -> None:
    if not isinstance(timing, Mapping):
        raise EventSourceError("timing はオブジェクトでなければならない")
    missing = REQUIRED_TIMING_KEYS - set(timing)
    if missing:
        raise EventSourceError(f"timing の必須項目がありません: {sorted(missing)}")
    values = {key: _require_nonnegative_int(timing[key], key) for key in REQUIRED_TIMING_KEYS}
    _validate_timing_order(values)
    _validate_first_visible(timing)


def _validate_timing_order(values: Mapping[str, int]) -> None:
    if values["occurred_earliest_frame"] > values["occurred_latest_frame"]:
        raise EventSourceError("物理発生フレーム区間が逆転しています")
    if values["occurred_earliest_ms"] > values["occurred_latest_ms"]:
        raise EventSourceError("物理発生時刻区間が逆転しています")
    if values["available_frame"] < values["occurred_latest_frame"]:
        raise EventSourceError("利用可能フレームを物理発生区間より前にできない")
    if values["available_ms"] < values["occurred_latest_ms"]:
        raise EventSourceError("利用可能時刻を物理発生区間より前にできない")


def _validate_first_visible(timing: Mapping[str, Any]) -> None:
    has_frame = "first_visible_frame" in timing
    has_ms = "first_visible_ms" in timing
    if has_frame != has_ms:
        raise EventSourceError("最初の観測位置はフレームとミリ秒を対で持つ")
    if has_frame:
        _require_nonnegative_int(timing["first_visible_frame"], "first_visible_frame")
        _require_nonnegative_int(timing["first_visible_ms"], "first_visible_ms")


def validate_event(event: Mapping[str, Any]) -> None:
    """単一出来事の共通不変条件を検査する。"""

    missing = REQUIRED_EVENT_KEYS - set(event)
    if missing:
        raise EventSourceError(f"出来事の必須項目がありません: {sorted(missing)}")
    if event["record_kind"] != RECORD_EVENT or event["schema_version"] != SCHEMA_VERSION:
        raise EventSourceError("出来事行の種類または仕様版が不正です")
    _validate_event_identity(event)
    _validate_timing(event["timing"])
    _validate_event_containers(event)
    _normalize_semantic(event)


def _validate_event_identity(event: Mapping[str, Any]) -> None:
    build_id = _require_nonempty_text(event["build_id"], "build_id")
    _require_nonempty_text(event["source_video_id"], "source_video_id")
    _require_nonempty_text(event["attempt_id"], "attempt_id")
    _require_nonempty_text(event["availability_batch_id"], "availability_batch_id")
    seq = _require_nonnegative_int(event["seq"], "seq")
    _require_nonnegative_int(event["batch_index"], "batch_index")
    _require_nonnegative_int(event["batch_size"], "batch_size")
    if event["event_id"] != f"{build_id}:{seq}":
        raise EventSourceError("event_id を build_id と seq から再計算できません")
    if event["side"] not in ALLOWED_SIDES:
        raise EventSourceError("side が許可値ではありません")
    _require_nonempty_text(event["event_type"], "event_type")


def _validate_event_containers(event: Mapping[str, Any]) -> None:
    assertion = event["assertion"]
    if not isinstance(assertion, Mapping):
        raise EventSourceError("assertion はオブジェクトでなければならない")
    if assertion.get("state") not in ALLOWED_ASSERTION_STATES:
        raise EventSourceError("assertion.state が許可値ではありません")
    if assertion.get("value_form") not in ALLOWED_VALUE_FORMS:
        raise EventSourceError("assertion.value_form が許可値ではありません")
    _require_mapping_list(event["evidence"], "evidence")
    _require_mapping_list(event["checks"], "checks")
    missing = _require_list(event["missing_information"], "missing_information")
    if any(not isinstance(item, str) or not item for item in missing):
        raise EventSourceError("missing_information は空でない文字列の配列でなければならない")
    _require_mapping(event["relations"], "relations")
    _require_mapping_list(event["heavy_evidence_refs"], "heavy_evidence_refs")
    _require_mapping(event["payload"], "payload")


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise EventSourceError(f"{name} は配列でなければならない")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EventSourceError(f"{name} はオブジェクトでなければならない")
    return value


def _require_mapping_list(value: Any, name: str) -> list[Any]:
    items = _require_list(value, name)
    if any(not isinstance(item, Mapping) for item in items):
        raise EventSourceError(f"{name} の各要素はオブジェクトでなければならない")
    return items


def validate_event_batch(
    events: Sequence[Mapping[str, Any]], expected_first_seq: int | None = None
) -> None:
    """一つの利用可能更新として反映する出来事群を検査する。"""

    if not events or len(events) > MAX_BATCH_EVENTS:
        raise EventSourceError("一括更新の出来事件数が許可範囲外です")
    for event in events:
        validate_event(event)
    first = events[0]
    _validate_batch_shared_fields(events, first)
    _validate_batch_positions(events, expected_first_seq)


def _validate_batch_shared_fields(
    events: Sequence[Mapping[str, Any]], first: Mapping[str, Any]
) -> None:
    timing = first["timing"]
    shared = (
        first["source_video_id"],
        first["build_id"],
        first["attempt_id"],
        first["availability_batch_id"],
        first["batch_size"],
        timing["available_frame"],
        timing["available_ms"],
    )
    if any(_batch_shared_tuple(event) != shared for event in events):
        raise EventSourceError("一括更新内の共有項目が一致しません")


def _batch_shared_tuple(event: Mapping[str, Any]) -> tuple[Any, ...]:
    timing = event["timing"]
    return (
        event["source_video_id"],
        event["build_id"],
        event["attempt_id"],
        event["availability_batch_id"],
        event["batch_size"],
        timing["available_frame"],
        timing["available_ms"],
    )


def _validate_batch_positions(
    events: Sequence[Mapping[str, Any]], expected_first_seq: int | None
) -> None:
    first_seq = int(events[0]["seq"])
    if expected_first_seq is not None and first_seq != expected_first_seq:
        raise EventSourceError("一括更新の先頭通番が期待値と一致しません")
    for index, event in enumerate(events):
        if event["batch_index"] != index or event["batch_size"] != len(events):
            raise EventSourceError("一括更新内の位置または件数が不正です")
        if event["seq"] != first_seq + index:
            raise EventSourceError("一括更新内の通番が連続していません")


def _build_begin(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = events[0]
    return {
        "record_kind": RECORD_BATCH_BEGIN,
        "log_format_version": LOG_FORMAT_VERSION,
        "build_id": first["build_id"],
        "attempt_id": first["attempt_id"],
        "availability_batch_id": first["availability_batch_id"],
        "event_count": len(events),
        "first_seq": first["seq"],
        "last_seq": events[-1]["seq"],
        "event_ids_sha256": _event_ids_sha256(events),
        "available_frame": first["timing"]["available_frame"],
        "available_ms": first["timing"]["available_ms"],
    }


def _build_commit(
    events: Sequence[Mapping[str, Any]], physical_payload: bytes
) -> dict[str, Any]:
    first = events[0]
    return {
        "record_kind": RECORD_BATCH_COMMIT,
        "log_format_version": LOG_FORMAT_VERSION,
        "build_id": first["build_id"],
        "attempt_id": first["attempt_id"],
        "availability_batch_id": first["availability_batch_id"],
        "event_count": len(events),
        "event_ids_sha256": _event_ids_sha256(events),
        "semantic_hash_version": SEMANTIC_HASH_VERSION,
        "semantic_batch_sha256": semantic_events_sha256(events),
        "physical_payload_sha256": hashlib.new(HASH_NAME, physical_payload).hexdigest(),
    }


class EventPartWriter:
    """既存ファイルを上書きしない JSON Lines 部品書込み器。"""

    def __init__(
        self,
        path: Path,
        *,
        expected_first_seq: int = 0,
        sync_every_batches: int = DEFAULT_SYNC_EVERY_BATCHES,
    ) -> None:
        if sync_every_batches <= 0:
            raise ValueError("sync_every_batches は1以上でなければならない")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("xb")
        self._next_seq = expected_first_seq
        self._last_available_frame = -1
        self._last_available_ms = -1
        self._batch_count = 0
        self._sync_every_batches = sync_every_batches
        self._part_identity: tuple[str, str, str] | None = None
        self._seen_batch_ids: set[str] = set()
        self._closed = False
        self._failed = False

    def append_batch(self, events: Sequence[Mapping[str, Any]]) -> None:
        """検査済み一括更新を開始・出来事・確定の順で追記する。"""

        self._ensure_open()
        validate_event_batch(events, self._next_seq)
        self._validate_part_sequence(events)
        payload = self._encode_batch(events)
        self._write_payload(payload)
        self._record_batch(events)
        self._batch_count += 1
        self._next_seq = int(events[-1]["seq"]) + 1
        if self._batch_count % self._sync_every_batches == 0:
            self.sync()

    def _validate_part_sequence(self, events: Sequence[Mapping[str, Any]]) -> None:
        first = events[0]
        identity = _event_part_identity(first)
        if self._part_identity is not None and identity != self._part_identity:
            raise EventSourceError("部品内で元映像・生成内容・実行試行を変更できません")
        batch_id = str(first["availability_batch_id"])
        if batch_id in self._seen_batch_ids:
            raise EventSourceError("一括更新IDが部品内で重複しています")
        frame, milliseconds = _event_available_position(first)
        if frame <= self._last_available_frame or milliseconds < self._last_available_ms:
            raise EventSourceError("公開位置を過去または同一フレームへ戻せません")

    def _record_batch(self, events: Sequence[Mapping[str, Any]]) -> None:
        first = events[0]
        self._part_identity = _event_part_identity(first)
        self._seen_batch_ids.add(str(first["availability_batch_id"]))
        self._last_available_frame, self._last_available_ms = _event_available_position(first)

    def _write_payload(self, payload: bytes) -> None:
        try:
            written = self._handle.write(payload)
        except OSError:
            self._failed = True
            raise
        if written != len(payload):
            self._failed = True
            raise EventSourceError("一括更新を完全に書き込めませんでした")

    def _encode_batch(self, events: Sequence[Mapping[str, Any]]) -> bytes:
        begin_line = _physical_json_line(_build_begin(events))
        event_lines = [_physical_json_line(event) for event in events]
        physical_payload = begin_line + b"".join(event_lines)
        commit_line = _physical_json_line(_build_commit(events, physical_payload))
        return physical_payload + commit_line

    def _ensure_open(self) -> None:
        if self._closed:
            raise EventSourceError("閉じた書込み器は使用できません")
        if self._failed:
            raise EventSourceError("書込み失敗後の部品は継続利用できません")

    def sync(self) -> None:
        """PythonバッファとOSバッファを永続化する。"""

        self._ensure_open()
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._closed:
            return
        try:
            if not self._failed:
                self.sync()
        finally:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> EventPartWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _read_json_line(handle: BinaryIO, line_no: int) -> tuple[dict[str, Any], bytes]:
    raw = handle.readline()
    if not raw:
        raise EventSourceError("一括更新の途中でファイル終端に到達しました")
    if not raw.endswith(b"\n"):
        raise EventSourceError("末尾行が改行で完結していません")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventSourceError(f"{line_no}行目が有効なJSONではありません") from exc
    if not isinstance(value, dict):
        raise EventSourceError(f"{line_no}行目はJSONオブジェクトではありません")
    return value, raw


def _event_part_identity(event: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(event["source_video_id"]),
        str(event["build_id"]),
        str(event["attempt_id"]),
    )


def _event_available_position(event: Mapping[str, Any]) -> tuple[int, int]:
    timing = event["timing"]
    return int(timing["available_frame"]), int(timing["available_ms"])


def _validate_control_pair(
    begin: Mapping[str, Any], commit: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> None:
    expected_ids = _event_ids_sha256(events)
    shared_keys = ("build_id", "attempt_id", "availability_batch_id", "event_count")
    if any(begin.get(key) != commit.get(key) for key in shared_keys):
        raise EventSourceError("開始行と確定行の共有項目が一致しません")
    if commit.get("log_format_version") != LOG_FORMAT_VERSION:
        raise EventSourceError("確定行のログ物理形式版が一致しません")
    if commit.get("semantic_hash_version") != SEMANTIC_HASH_VERSION:
        raise EventSourceError("確定行の意味要約規則版が一致しません")
    if begin.get("event_ids_sha256") != expected_ids:
        raise EventSourceError("開始行の出来事ID要約値が一致しません")
    if commit.get("event_ids_sha256") != expected_ids:
        raise EventSourceError("確定行の出来事ID要約値が一致しません")
    if commit.get("semantic_batch_sha256") != semantic_events_sha256(events):
        raise EventSourceError("一括更新の意味内容要約値が一致しません")


def _validate_begin(begin: Mapping[str, Any]) -> int:
    if begin.get("record_kind") != RECORD_BATCH_BEGIN:
        raise EventSourceError("一括更新開始行がありません")
    if begin.get("log_format_version") != LOG_FORMAT_VERSION:
        raise EventSourceError("ログ物理形式版が一致しません")
    count = _require_nonnegative_int(begin.get("event_count"), "event_count")
    if count <= 0 or count > MAX_BATCH_EVENTS:
        raise EventSourceError("一括更新の出来事件数が許可範囲外です")
    return count


def _read_batch(
    handle: BinaryIO, line_no: int, expected_seq: int | None
) -> tuple[CommittedBatch, int]:
    start_offset = handle.tell()
    begin, begin_raw = _read_json_line(handle, line_no)
    count = _validate_begin(begin)
    events, event_raw, next_line = _read_event_lines(handle, line_no + 1, count)
    commit, _ = _read_json_line(handle, next_line)
    if commit.get("record_kind") != RECORD_BATCH_COMMIT:
        raise EventSourceError("一括更新確定行がありません")
    validate_event_batch(events, expected_seq)
    _validate_begin_against_events(begin, events)
    _validate_control_pair(begin, commit, events)
    physical_payload = begin_raw + b"".join(event_raw)
    if commit.get("physical_payload_sha256") != hashlib.new(HASH_NAME, physical_payload).hexdigest():
        raise EventSourceError("一括更新の物理要約値が一致しません")
    batch = CommittedBatch(
        str(begin["availability_batch_id"]), tuple(events), start_offset,
        handle.tell(), semantic_events_sha256(events),
    )
    return batch, next_line + 1


def _read_event_lines(
    handle: BinaryIO, line_no: int, count: int
) -> tuple[list[dict[str, Any]], list[bytes], int]:
    events: list[dict[str, Any]] = []
    raw_lines: list[bytes] = []
    current_line = line_no
    for _ in range(count):
        event, raw = _read_json_line(handle, current_line)
        events.append(event)
        raw_lines.append(raw)
        current_line += 1
    return events, raw_lines, current_line


def _validate_begin_against_events(
    begin: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> None:
    first = events[0]
    checks = {
        "build_id": first["build_id"],
        "attempt_id": first["attempt_id"],
        "availability_batch_id": first["availability_batch_id"],
        "first_seq": first["seq"],
        "last_seq": events[-1]["seq"],
        "available_frame": first["timing"]["available_frame"],
        "available_ms": first["timing"]["available_ms"],
    }
    if any(begin.get(key) != value for key, value in checks.items()):
        raise EventSourceError("開始行と出来事行の内容が一致しません")


def iter_committed_batches(
    path: Path, *, expected_first_seq: int | None = None
) -> Iterator[CommittedBatch]:
    """完全に確定した一括更新だけを順番に返す。"""

    valid_prefix = 0
    line_no = 1
    state = _PartSequenceState(expected_seq=expected_first_seq)
    with path.open("rb") as handle:
        while _has_more_bytes(handle):
            try:
                batch, line_no = _read_batch(handle, line_no, state.expected_seq)
                state.add(batch)
            except EventSourceError as exc:
                raise PartReadError(str(exc), line_no, valid_prefix) from exc
            valid_prefix = batch.end_offset
            yield batch


def _has_more_bytes(handle: BinaryIO) -> bool:
    position = handle.tell()
    found = bool(handle.read(1))
    handle.seek(position)
    return found


@dataclass(slots=True)
class _PartSequenceState:
    expected_seq: int | None = None
    identity: tuple[str, str, str] | None = None
    last_available_frame: int = -1
    last_available_ms: int = -1
    batch_ids: set[str] = field(default_factory=set)

    def add(self, batch: CommittedBatch) -> None:
        first = batch.events[0]
        identity = _event_part_identity(first)
        if self.identity is not None and identity != self.identity:
            raise EventSourceError("部品内で元映像・生成内容・実行試行が変化しています")
        frame, milliseconds = _event_available_position(first)
        if frame <= self.last_available_frame or milliseconds < self.last_available_ms:
            raise EventSourceError("一括更新間の公開位置が逆行または重複しています")
        if batch.batch_id in self.batch_ids:
            raise EventSourceError("一括更新IDが部品内で重複しています")
        self.identity = identity
        self.last_available_frame = frame
        self.last_available_ms = milliseconds
        self.batch_ids.add(batch.batch_id)
        self.expected_seq = int(batch.events[-1]["seq"]) + 1


def _physical_file_sha256(path: Path) -> str:
    hasher = hashlib.new(HASH_NAME)
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_part(path: Path, *, expected_first_seq: int | None = None) -> PartValidation:
    """部品全体を走査し、最初の不整合と正常確定済み接頭辞を返す。"""

    state = _ValidationState()
    try:
        for batch in iter_committed_batches(path, expected_first_seq=expected_first_seq):
            state.add(batch)
    except PartReadError as exc:
        return state.result(path, error=str(exc), error_line=exc.line_no,
                            valid_prefix_bytes=exc.valid_prefix_bytes)
    if state.batch_count == 0:
        return state.result(
            path,
            error="確定済み一括更新が一件もありません",
            error_line=1,
            valid_prefix_bytes=0,
        )
    return state.result(path, valid_prefix_bytes=path.stat().st_size)


@dataclass(slots=True)
class _ValidationState:
    batch_count: int = 0
    event_count: int = 0
    first_seq: int | None = None
    last_seq: int | None = None
    semantic_hasher: SemanticEventHasher | None = None

    def __post_init__(self) -> None:
        self.semantic_hasher = SemanticEventHasher()

    def add(self, batch: CommittedBatch) -> None:
        if self.first_seq is None:
            self.first_seq = int(batch.events[0]["seq"])
        self.last_seq = int(batch.events[-1]["seq"])
        self.batch_count += 1
        self.event_count += len(batch.events)
        for event in batch.events:
            self.semantic_hasher.add(event)

    def result(
        self, path: Path, *, valid_prefix_bytes: int,
        error: str | None = None, error_line: int | None = None,
    ) -> PartValidation:
        return PartValidation(
            error is None, self.batch_count, self.event_count, self.first_seq,
            self.last_seq, valid_prefix_bytes, _physical_file_sha256(path),
            self.semantic_hasher.hexdigest(), error, error_line,
        )


def copy_recoverable_prefix(
    source: Path, quarantine_copy: Path, recovered_part: Path
) -> PartValidation:
    """破損元を保持し、正常確定済み接頭辞だけを別名へ複製する。"""

    validation = validate_part(source)
    if validation.valid:
        raise EventSourceError("正常な部品に復旧処理は不要です")
    _ensure_full_copy(source, quarantine_copy)
    if validation.valid_prefix_bytes == 0:
        return validation
    _ensure_prefix_copy(source, recovered_part, validation.valid_prefix_bytes)
    recovered = validate_part(recovered_part)
    if not recovered.valid:
        raise EventSourceError("複製した正常接頭辞の再検査に失敗しました")
    return validation


def _ensure_full_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if _files_equal(source, destination):
            return
        raise EventSourceError("既存の隔離先が破損元と一致しません")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


def _ensure_prefix_copy(source: Path, destination: Path, size: int) -> None:
    if destination.exists():
        if _prefix_copy_matches(source, destination, size):
            return
        raise EventSourceError("既存の復旧先が正常接頭辞と一致しません")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        remaining = size
        while remaining:
            chunk = src.read(min(remaining, COPY_CHUNK_BYTES))
            if not chunk:
                raise EventSourceError("正常接頭辞の読取り中に終端へ到達しました")
            dst.write(chunk)
            remaining -= len(chunk)
        dst.flush()
        os.fsync(dst.fileno())


def _files_equal(first: Path, second: Path) -> bool:
    return first.stat().st_size == second.stat().st_size and (
        _physical_file_sha256(first) == _physical_file_sha256(second)
    )


def _prefix_copy_matches(source: Path, destination: Path, size: int) -> bool:
    if destination.stat().st_size != size:
        return False
    with source.open("rb") as src, destination.open("rb") as dst:
        remaining = size
        while remaining:
            chunk_size = min(remaining, COPY_CHUNK_BYTES)
            if src.read(chunk_size) != dst.read(chunk_size):
                return False
            remaining -= chunk_size
    return True


__all__ = [
    "CommittedBatch",
    "EventPartWriter",
    "EventSourceError",
    "PartReadError",
    "PartValidation",
    "SemanticEventHasher",
    "copy_recoverable_prefix",
    "iter_committed_batches",
    "semantic_event_bytes",
    "semantic_events_sha256",
    "validate_event",
    "validate_event_batch",
    "validate_part",
]
