"""出来事原本 v1 の追記・決定性・中断復旧試験。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.event_source_v1 import (
    EventPartWriter,
    EventSourceError,
    PartReadError,
    copy_recoverable_prefix,
    iter_committed_batches,
    semantic_event_bytes,
    semantic_events_sha256,
    validate_event,
    validate_event_batch,
    validate_part,
)


def _event(
    seq: int = 0,
    *,
    attempt_id: str = "attempt-001",
    batch_id: str = "build-test:batch-010",
    batch_index: int = 0,
    batch_size: int = 1,
    available_frame: int = 10,
) -> dict[str, Any]:
    available_ms = available_frame * 100
    return {
        "record_kind": "event",
        "schema_version": "event-source/1.0",
        "source_video_id": "v038",
        "build_id": "build-test",
        "attempt_id": attempt_id,
        "seq": seq,
        "event_id": f"build-test:{seq}",
        "availability_batch_id": batch_id,
        "batch_index": batch_index,
        "batch_size": batch_size,
        "event_type": "chain_started",
        "side": "p1",
        "timing": {
            "occurred_earliest_frame": available_frame - 1,
            "occurred_earliest_ms": available_ms - 100,
            "occurred_latest_frame": available_frame - 1,
            "occurred_latest_ms": available_ms - 100,
            "available_frame": available_frame,
            "available_ms": available_ms,
        },
        "assertion": {"state": "confirmed", "value_form": "exact"},
        "evidence": [],
        "checks": [],
        "missing_information": [],
        "relations": {"revision": {"action": "none", "target_event_ids": []}},
        "heavy_evidence_refs": [],
        "payload": {"chain_step": 1},
    }


def _batch(
    first_seq: int,
    size: int,
    *,
    frame: int,
    attempt_id: str = "attempt-001",
) -> list[dict[str, Any]]:
    batch_id = f"build-test:batch-{frame:03d}"
    return [
        _event(
            first_seq + index,
            attempt_id=attempt_id,
            batch_id=batch_id,
            batch_index=index,
            batch_size=size,
            available_frame=frame,
        )
        for index in range(size)
    ]


def _write_part(path: Path, attempt_id: str = "attempt-001") -> None:
    with EventPartWriter(path) as writer:
        writer.append_batch(_batch(0, 2, frame=10, attempt_id=attempt_id))
        writer.append_batch(_batch(2, 1, frame=12, attempt_id=attempt_id))


def _replace_json_line(path: Path, line_index: int, **updates: Any) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[line_index])
    value.update(updates)
    lines[line_index] = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    path.write_bytes(b"".join(lines))


def test_valid_event_and_batch_are_accepted() -> None:
    events = _batch(4, 2, frame=20)
    validate_event(events[0])
    validate_event_batch(events, expected_first_seq=4)


def test_semantic_hash_ignores_attempt_id() -> None:
    first = _event(attempt_id="attempt-001")
    second = _event(attempt_id="attempt-999")
    assert semantic_event_bytes(first) == semantic_event_bytes(second)
    assert semantic_events_sha256([first]) == semantic_events_sha256([second])


def test_semantic_hash_normalizes_unordered_lists() -> None:
    first = _event()
    first["evidence"] = [{"evidence_type": "b"}, {"evidence_type": "a"}]
    first["missing_information"] = ["z", "a"]
    second = copy.deepcopy(first)
    second["evidence"].reverse()
    second["missing_information"].reverse()
    assert semantic_event_bytes(first) == semantic_event_bytes(second)


def test_semantic_hash_preserves_ordered_candidate_list() -> None:
    first = _event()
    first["payload"]["candidate_values"] = [1, 2, 3]
    second = copy.deepcopy(first)
    second["payload"]["candidate_values"] = [3, 2, 1]
    assert semantic_event_bytes(first) != semantic_event_bytes(second)


def test_nested_storage_path_is_nonsemantic_but_content_hash_is_semantic() -> None:
    first = _event()
    first["heavy_evidence_refs"] = [
        {"evidence_type": "frame", "storage_path": "C:/first.png", "content_sha256": "a" * 64}
    ]
    second = copy.deepcopy(first)
    second["heavy_evidence_refs"][0]["storage_path"] = "/tmp/second.png"
    assert semantic_event_bytes(first) == semantic_event_bytes(second)
    second["heavy_evidence_refs"][0]["content_sha256"] = "b" * 64
    assert semantic_event_bytes(first) != semantic_event_bytes(second)


def test_unknown_extension_key_changes_semantic_hash() -> None:
    first = _event()
    second = copy.deepcopy(first)
    second["future_extension"] = {"revision": 1}
    assert semantic_event_bytes(first) != semantic_event_bytes(second)


@pytest.mark.parametrize("invalid", [None, 0.5, float("nan")])
def test_null_and_floats_are_rejected_from_semantic_fields(invalid: Any) -> None:
    event = _event()
    event["payload"]["invalid"] = invalid
    with pytest.raises(EventSourceError):
        validate_event(event)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("evidence", {}),
        ("evidence", ["not-object"]),
        ("relations", []),
        ("payload", []),
        ("missing_information", [""]),
    ],
)
def test_required_container_shapes_are_enforced(field: str, invalid: Any) -> None:
    event = _event()
    event[field] = invalid
    with pytest.raises(EventSourceError):
        validate_event(event)


def test_identity_and_timing_invariants_are_enforced() -> None:
    wrong_id = _event()
    wrong_id["event_id"] = "wrong"
    with pytest.raises(EventSourceError, match="event_id"):
        validate_event(wrong_id)
    reversed_time = _event()
    reversed_time["timing"]["occurred_latest_frame"] = 11
    with pytest.raises(EventSourceError, match="利用可能フレーム"):
        validate_event(reversed_time)


def test_first_visible_position_requires_frame_and_time_pair() -> None:
    event = _event()
    event["timing"]["first_visible_frame"] = 9
    with pytest.raises(EventSourceError, match="対"):
        validate_event(event)


def test_batch_position_and_shared_fields_are_enforced() -> None:
    wrong_position = _batch(0, 2, frame=10)
    wrong_position[1]["batch_index"] = 0
    with pytest.raises(EventSourceError, match="位置"):
        validate_event_batch(wrong_position)
    mixed_attempt = _batch(0, 2, frame=10)
    mixed_attempt[1]["attempt_id"] = "attempt-002"
    with pytest.raises(EventSourceError, match="共有項目"):
        validate_event_batch(mixed_attempt)


def test_batch_count_boundaries_are_enforced() -> None:
    with pytest.raises(EventSourceError, match="件数"):
        validate_event_batch([])
    with pytest.raises(EventSourceError, match="件数"):
        validate_event_batch([_event()] * 10_001)


def test_writer_and_reader_round_trip_two_batches(tmp_path: Path) -> None:
    path = tmp_path / "part-00000.jsonl"
    _write_part(path)
    batches = list(iter_committed_batches(path, expected_first_seq=0))
    result = validate_part(path, expected_first_seq=0)
    assert [len(batch.events) for batch in batches] == [2, 1]
    assert result.valid
    assert (result.batch_count, result.event_count) == (2, 3)
    assert (result.first_seq, result.last_seq) == (0, 2)
    assert result.valid_prefix_bytes == path.stat().st_size


def test_independent_attempts_have_same_semantics_but_different_bytes(tmp_path: Path) -> None:
    first = tmp_path / "attempt-001.jsonl"
    second = tmp_path / "attempt-002.jsonl"
    _write_part(first, "attempt-001")
    _write_part(second, "attempt-002")
    first_result = validate_part(first)
    second_result = validate_part(second)
    assert first_result.semantic_sha256 == second_result.semantic_sha256
    assert first_result.physical_sha256 != second_result.physical_sha256
    assert first.read_bytes() != second.read_bytes()


def test_writer_never_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "existing.jsonl"
    original = b"user-owned\n"
    path.write_bytes(original)
    with pytest.raises(FileExistsError):
        EventPartWriter(path)
    assert path.read_bytes() == original


def test_writer_rejects_nonincreasing_publication_frame(tmp_path: Path) -> None:
    path = tmp_path / "part.jsonl"
    with EventPartWriter(path) as writer:
        writer.append_batch(_batch(0, 1, frame=10))
        same_frame = _batch(1, 1, frame=10)
        same_frame[0]["availability_batch_id"] = "build-test:batch-010-second"
        with pytest.raises(EventSourceError, match="公開位置"):
            writer.append_batch(same_frame)
    assert validate_part(path).event_count == 1


def test_writer_rejects_identity_change_and_duplicate_batch_id(tmp_path: Path) -> None:
    path = tmp_path / "part.jsonl"
    with EventPartWriter(path) as writer:
        writer.append_batch(_batch(0, 1, frame=10))
        changed_attempt = _batch(1, 1, frame=12, attempt_id="attempt-002")
        with pytest.raises(EventSourceError, match="元映像・生成内容・実行試行"):
            writer.append_batch(changed_attempt)
        duplicate_id = _batch(1, 1, frame=12)
        duplicate_id[0]["availability_batch_id"] = "build-test:batch-010"
        with pytest.raises(EventSourceError, match="一括更新ID"):
            writer.append_batch(duplicate_id)


def test_writer_rejects_unexpected_first_sequence(tmp_path: Path) -> None:
    path = tmp_path / "part.jsonl"
    with EventPartWriter(path, expected_first_seq=7) as writer:
        with pytest.raises(EventSourceError, match="先頭通番"):
            writer.append_batch(_batch(0, 1, frame=10))


def test_missing_commit_is_rejected_without_adopting_partial_batch(tmp_path: Path) -> None:
    path = tmp_path / "missing-commit.jsonl"
    _write_part(path)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))
    result = validate_part(path)
    assert not result.valid
    assert (result.batch_count, result.event_count) == (1, 2)
    assert result.valid_prefix_bytes == len(b"".join(lines[:4]))


def test_incomplete_last_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "truncated.jsonl"
    _write_part(path)
    path.write_bytes(path.read_bytes()[:-3])
    result = validate_part(path)
    assert not result.valid
    assert result.event_count == 2
    assert "改行" in (result.error or "")


@pytest.mark.parametrize(
    ("line_index", "updates"),
    [
        (0, {"event_ids_sha256": "0" * 64}),
        (3, {"semantic_batch_sha256": "0" * 64}),
        (3, {"physical_payload_sha256": "0" * 64}),
        (3, {"semantic_hash_version": "semantic-hash/v999"}),
    ],
)
def test_control_line_tampering_is_detected(
    tmp_path: Path, line_index: int, updates: dict[str, Any]
) -> None:
    path = tmp_path / f"tampered-{line_index}-{len(updates)}.jsonl"
    _write_part(path)
    _replace_json_line(path, line_index, **updates)
    assert not validate_part(path).valid


def test_event_line_tampering_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "tampered-event.jsonl"
    _write_part(path)
    _replace_json_line(path, 1, payload={"chain_step": 999})
    result = validate_part(path)
    assert not result.valid
    assert "意味内容要約値" in (result.error or "")


def test_event_line_physical_reformatting_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "reformatted-event.jsonl"
    _write_part(path)
    lines = path.read_bytes().splitlines(keepends=True)
    reformatted = lines[1].replace(b'"batch_index":0', b'"batch_index": 0')
    assert reformatted != lines[1]
    lines[1] = reformatted
    path.write_bytes(b"".join(lines))
    result = validate_part(path)
    assert not result.valid
    assert "物理要約値" in (result.error or "")


def test_iterator_reports_last_committed_prefix_on_corruption(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.jsonl"
    _write_part(path)
    lines = path.read_bytes().splitlines(keepends=True)
    expected_prefix = len(b"".join(lines[:4]))
    _replace_json_line(path, 4, record_kind="wrong")
    iterator = iter_committed_batches(path)
    assert len(next(iterator).events) == 2
    with pytest.raises(PartReadError) as captured:
        next(iterator)
    assert captured.value.valid_prefix_bytes == expected_prefix


def test_reader_rejects_cross_batch_identity_change(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    combined = tmp_path / "combined.jsonl"
    with EventPartWriter(first) as writer:
        writer.append_batch(_batch(0, 1, frame=10))
    with EventPartWriter(second, expected_first_seq=1) as writer:
        writer.append_batch(_batch(1, 1, frame=12, attempt_id="attempt-002"))
    combined.write_bytes(first.read_bytes() + second.read_bytes())
    result = validate_part(combined)
    assert not result.valid
    assert result.event_count == 1
    assert "実行試行" in (result.error or "")


def test_reader_rejects_cross_batch_publication_regression(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    combined = tmp_path / "combined.jsonl"
    with EventPartWriter(first) as writer:
        writer.append_batch(_batch(0, 1, frame=10))
    with EventPartWriter(second, expected_first_seq=1) as writer:
        writer.append_batch(_batch(1, 1, frame=9))
    combined.write_bytes(first.read_bytes() + second.read_bytes())
    result = validate_part(combined)
    assert not result.valid
    assert result.event_count == 1
    assert "公開位置" in (result.error or "")


def test_recovery_preserves_original_and_copies_only_committed_prefix(tmp_path: Path) -> None:
    source = tmp_path / "broken.jsonl"
    quarantine = tmp_path / "quarantine" / "broken.jsonl"
    recovered = tmp_path / "recovered" / "part-00000.jsonl"
    _write_part(source)
    original_lines = source.read_bytes().splitlines(keepends=True)
    source.write_bytes(b"".join(original_lines[:-1]))
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    broken = copy_recoverable_prefix(source, quarantine, recovered)
    assert not broken.valid
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert quarantine.read_bytes() == source.read_bytes()
    assert recovered.read_bytes() == b"".join(original_lines[:4])
    assert validate_part(recovered).valid


def test_recovery_never_overwrites_destination(tmp_path: Path) -> None:
    source = tmp_path / "broken.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    recovered = tmp_path / "recovered.jsonl"
    _write_part(source)
    source.write_bytes(source.read_bytes()[:-1])
    quarantine.write_bytes(b"keep")
    with pytest.raises(EventSourceError, match="一致しません"):
        copy_recoverable_prefix(source, quarantine, recovered)
    assert quarantine.read_bytes() == b"keep"


def test_recovery_can_be_retried_when_existing_outputs_match(tmp_path: Path) -> None:
    source = tmp_path / "broken.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    recovered = tmp_path / "recovered.jsonl"
    _write_part(source)
    source.write_bytes(source.read_bytes()[:-1])
    first = copy_recoverable_prefix(source, quarantine, recovered)
    second = copy_recoverable_prefix(source, quarantine, recovered)
    assert first == second
    assert validate_part(recovered).valid


def test_empty_part_is_invalid_and_recovery_creates_no_empty_part(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    recovered = tmp_path / "recovered.jsonl"
    source.write_bytes(b"")
    result = validate_part(source)
    assert not result.valid
    assert result.valid_prefix_bytes == 0
    copy_recoverable_prefix(source, quarantine, recovered)
    assert quarantine.exists()
    assert not recovered.exists()
