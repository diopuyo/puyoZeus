"""旧出来事試作を出来事原本v1へ一方向変換する。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping, Sequence

from src.event_data_contract import (
    EventRecord,
    EventSide,
    EventSource,
    EventStatus,
    EventType,
)
from src.event_source_v1 import RECORD_EVENT, SCHEMA_VERSION, validate_event_batch


EVENT_TYPE_MAP = {
    EventType.MATCH_START_CANDIDATE: "match_boundary_evidence",
    EventType.MATCH_START_CONFIRMED: "match_boundary_evidence",
    EventType.MATCH_END_CANDIDATE: "match_boundary_evidence",
    EventType.WINNER_CONFIRMED: "winner_observed",
    EventType.NEXT_SHIFT: "next_queue_observed",
    EventType.ATTACK_CANCELED: "garbage_cancelled",
    EventType.GARBAGE_COMMITTED_TO_FALL: "garbage_fall_reserved",
}
SIDE_MAP = {
    EventSide.P1: "p1",
    EventSide.P2: "p2",
    EventSide.SYSTEM: "system",
}
STATUS_MAP = {
    EventStatus.PROVISIONAL: ("provisional", "exact"),
    EventStatus.FINAL: ("confirmed", "exact"),
    EventStatus.REVOKED: ("confirmed", "exact"),
    EventStatus.CONFLICT: ("conflict", "conflict"),
    EventStatus.UNKNOWN: ("unknown", "unknown"),
}
VERIFIED_SOURCES = frozenset({EventSource.DIRECT_OBSERVATION, EventSource.RULE_DERIVED})
ESTIMATED_SOURCES = frozenset(
    {EventSource.MULTI_SIGNAL_ESTIMATE, EventSource.SINGLE_SIGNAL_ESTIMATE}
)


@dataclass(frozen=True, slots=True)
class LegacyMigrationContext:
    """一つの旧同時群をv1識別・時間基準へ結ぶ情報。"""

    source_video_id: str
    build_id: str
    attempt_id: str
    first_seq: int
    availability_batch_id: str
    time_base_numerator: int
    time_base_denominator: int
    processing_start_frame: int = 0

    def __post_init__(self) -> None:
        texts = (
            self.source_video_id,
            self.build_id,
            self.attempt_id,
            self.availability_batch_id,
        )
        if any(not value.strip() for value in texts):
            raise ValueError("移行識別情報は空にできません")
        integers = (
            self.first_seq,
            self.time_base_numerator,
            self.time_base_denominator,
            self.processing_start_frame,
        )
        if any(value < 0 for value in integers) or min(
            self.time_base_numerator, self.time_base_denominator
        ) == 0:
            raise ValueError("移行通番・時間基準・開始位置が不正です")


def migrate_legacy_group(
    records: Sequence[EventRecord],
    context: LegacyMigrationContext,
    *,
    known_event_id_map: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """旧同時群を意味を強めずにv1の一括更新へ変換する。"""

    if not records:
        raise ValueError("空の旧出来事群は移行できません")
    _validate_legacy_group(records)
    known = dict(known_event_id_map or {})
    _validate_known_mapping(records, context, known)
    ordered = _order_legacy_group(records, known)
    id_map = dict(known)
    for index, record in enumerate(ordered):
        id_map[record.event_id] = f"{context.build_id}:{context.first_seq + index}"
    migrated = tuple(
        _migrate_record(record, index, len(ordered), context, id_map)
        for index, record in enumerate(ordered)
    )
    validate_event_batch(migrated, expected_first_seq=context.first_seq)
    return migrated


def _validate_known_mapping(
    records: Sequence[EventRecord],
    context: LegacyMigrationContext,
    known: Mapping[str, str],
) -> None:
    record_ids = {record.event_id for record in records}
    if record_ids & set(known):
        raise ValueError("移行済みの旧出来事IDを再利用できません")
    values = [str(value) for value in known.values()]
    if len(values) != len(set(values)):
        raise ValueError("既知の旧出来事ID対応に移行先の重複があります")
    prefix = f"{context.build_id}:"
    if any(not value.startswith(prefix) for value in values):
        raise ValueError("既知の旧出来事ID対応が別の生成内容を参照しています")
    generated = {
        f"{context.build_id}:{context.first_seq + index}" for index in range(len(records))
    }
    if generated & set(values):
        raise ValueError("新しい移行通番が既知の移行先と重複しています")


def _validate_legacy_group(records: Sequence[EventRecord]) -> None:
    first = records[0]
    expected = (first.event_group_id, first.match_id, first.timing.available_at_sec)
    if any(
        (record.event_group_id, record.match_id, record.timing.available_at_sec) != expected
        for record in records
    ):
        raise ValueError("旧同時群の群・試合・利用可能時刻が一致しません")
    ids = [record.event_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("旧同時群の出来事IDが重複しています")


def _order_legacy_group(
    records: Sequence[EventRecord], known_event_id_map: Mapping[str, str]
) -> tuple[EventRecord, ...]:
    group_ids = {record.event_id for record in records}
    known_ids = set(known_event_id_map)
    unresolved = list(records)
    ordered: list[EventRecord] = []
    emitted = set(known_ids)
    while unresolved:
        ready = [
            record
            for record in unresolved
            if set(record.cause_event_ids) <= emitted
            and (record.revision_of is None or record.revision_of in known_ids)
        ]
        if not ready:
            unknown = _unknown_legacy_references(unresolved, group_ids | known_ids)
            if unknown:
                raise ValueError(f"旧参照IDの移行先がありません: {sorted(unknown)}")
            raise ValueError("旧同時群の原因関係に循環があります")
        for record in ready:
            unresolved.remove(record)
            ordered.append(record)
            emitted.add(record.event_id)
    return tuple(ordered)


def _unknown_legacy_references(
    records: Sequence[EventRecord], allowed: set[str]
) -> set[str]:
    referenced = {
        value
        for record in records
        for value in (*record.cause_event_ids, *(() if record.revision_of is None else (record.revision_of,)))
    }
    return referenced - allowed


def _migrate_record(
    record: EventRecord,
    batch_index: int,
    batch_size: int,
    context: LegacyMigrationContext,
    id_map: Mapping[str, str],
) -> dict[str, Any]:
    seq = context.first_seq + batch_index
    timing, timing_missing = _migrate_timing(record, context)
    relations, relation_missing = _migrate_relations(record, context, id_map)
    assertion_state, value_form, assertion_missing = _assertion(record)
    return {
        "record_kind": RECORD_EVENT,
        "schema_version": SCHEMA_VERSION,
        "source_video_id": context.source_video_id,
        "build_id": context.build_id,
        "attempt_id": context.attempt_id,
        "seq": seq,
        "event_id": f"{context.build_id}:{seq}",
        "availability_batch_id": context.availability_batch_id,
        "batch_index": batch_index,
        "batch_size": batch_size,
        "event_type": EVENT_TYPE_MAP.get(record.event_type, record.event_type.value),
        "side": SIDE_MAP[record.side],
        "timing": timing,
        "assertion": {"state": assertion_state, "value_form": value_form},
        "evidence": _migrate_evidence(record),
        "checks": [_migration_check()],
        "missing_information": sorted(
            set(timing_missing + relation_missing + assertion_missing)
        ),
        "relations": relations,
        "heavy_evidence_refs": [],
        "payload": _migrate_payload(record),
    }


def _migrate_timing(
    record: EventRecord, context: LegacyMigrationContext
) -> tuple[dict[str, int], list[str]]:
    timing = record.timing
    available_frame, available_ms = _position(timing.available_at_sec, context)
    missing: list[str] = []
    if timing.occurred_start_sec is None:
        earliest_frame = context.processing_start_frame
        earliest_ms = _frame_ms(earliest_frame, context)
        missing.append("legacy_occurrence_start_missing")
    else:
        earliest_frame, earliest_ms = _position(timing.occurred_start_sec, context)
    if timing.occurred_end_sec is None:
        latest_frame, latest_ms = available_frame, available_ms
        missing.append("legacy_occurrence_end_missing")
    else:
        latest_frame, latest_ms = _position(timing.occurred_end_sec, context)
    if earliest_frame > latest_frame or latest_frame > available_frame:
        raise ValueError("旧物理発生区間をv1の利用可能位置へ変換できません")
    visible_frame, visible_ms = _position(timing.observed_at_sec, context)
    return {
        "occurred_earliest_frame": earliest_frame,
        "occurred_earliest_ms": earliest_ms,
        "occurred_latest_frame": latest_frame,
        "occurred_latest_ms": latest_ms,
        "available_frame": available_frame,
        "available_ms": available_ms,
        "first_visible_frame": visible_frame,
        "first_visible_ms": visible_ms,
    }, missing


def _assertion(record: EventRecord) -> tuple[str, str, list[str]]:
    if record.source == EventSource.UNOBSERVABLE:
        return "unknown", "unknown", ["legacy_value_unobservable"]
    if record.source in ESTIMATED_SOURCES:
        return "provisional", "candidates", ["legacy_final_estimate_not_promoted"]
    state, value_form = STATUS_MAP[record.status]
    return state, value_form, []


def _position(seconds: float, context: LegacyMigrationContext) -> tuple[int, int]:
    decimal_seconds = Decimal(str(seconds))
    if not decimal_seconds.is_finite():
        raise ValueError("旧時刻は有限でなければなりません")
    frame = _rounded_int(
        decimal_seconds * context.time_base_denominator / context.time_base_numerator
    )
    return frame, _rounded_int(decimal_seconds * 1000)


def _frame_ms(frame: int, context: LegacyMigrationContext) -> int:
    value = Decimal(frame * context.time_base_numerator * 1000)
    return _rounded_int(value / context.time_base_denominator)


def _rounded_int(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _migrate_relations(
    record: EventRecord,
    context: LegacyMigrationContext,
    id_map: Mapping[str, str],
) -> tuple[dict[str, Any], list[str]]:
    causes = [_mapped_event_id(value, id_map, "原因") for value in record.cause_event_ids]
    verified = causes if record.source in VERIFIED_SOURCES else []
    inferred = [] if record.source in VERIFIED_SOURCES else causes
    action, targets, missing = _revision(record, id_map)
    relations: dict[str, Any] = {
        "simultaneous_group_id": f"{context.build_id}:legacy-group:{record.event_group_id}",
        "verified_cause_event_ids": verified,
        "inferred_cause_event_ids": inferred,
        "revision": {"action": action, "target_event_ids": targets},
    }
    _add_scoped_relation(relations, "chain_id", record.chain_id, context.build_id)
    _add_scoped_relation(relations, "attack_id", record.attack_id, context.build_id)
    return relations, missing


def _revision(
    record: EventRecord, id_map: Mapping[str, str]
) -> tuple[str, list[str], list[str]]:
    if record.revision_of is None:
        return "none", [], []
    if record.source not in VERIFIED_SOURCES:
        raise ValueError("推定または観測不能由来の旧訂正は自動移行できません")
    target = _mapped_event_id(record.revision_of, id_map, "訂正対象")
    if record.status == EventStatus.REVOKED:
        return "revokes", [target], []
    return "corrects", [target], ["legacy_revision_action_ambiguous"]


def _mapped_event_id(value: str, id_map: Mapping[str, str], label: str) -> str:
    if value not in id_map:
        raise ValueError(f"旧{label}IDの移行先がありません: {value}")
    return str(id_map[value])


def _add_scoped_relation(
    relations: dict[str, Any], name: str, value: str | None, build_id: str
) -> None:
    if value is not None:
        relations[name] = f"{build_id}:legacy-{name}:{value}"


def _migrate_evidence(record: EventRecord) -> list[dict[str, Any]]:
    evidence = [
        {
            "evidence_type": "legacy_contract_source",
            "method_id": "event_data_contract",
            "method_version": "legacy/v0",
            "raw_value": record.source.value,
        }
    ]
    evidence.extend(
        {
            "evidence_type": "legacy_text_reference",
            "method_id": "event_data_contract",
            "method_version": "legacy/v0",
            "raw_value": value,
        }
        for value in record.evidence
    )
    return evidence


def _migration_check() -> dict[str, Any]:
    return {
        "check_id": "legacy_contract_migration",
        "check_version": "legacy-to-event-source/v1",
        "result": "pass",
        "reason_codes": [],
    }


def _migrate_payload(record: EventRecord) -> dict[str, Any]:
    attributes = [
        _migrate_attribute(name, value) for name, value in record.attributes
    ]
    payload: dict[str, Any] = {
        "legacy_event_id": record.event_id,
        "legacy_match_id_unverified": record.match_id,
        "legacy_event_type": record.event_type.value,
        "legacy_status": record.status.value,
        "attributes": attributes,
    }
    if record.confidence is not None:
        payload["legacy_confidence_ppm"] = _rounded_int(
            Decimal(str(record.confidence)) * 1_000_000
        )
    if record.timing.finalized_at_sec is not None:
        payload["legacy_finalized_ms"] = _rounded_int(
            Decimal(str(record.timing.finalized_at_sec)) * 1000
        )
    if record.before_state_id is not None:
        payload["legacy_before_state_id"] = record.before_state_id
    if record.after_state_id is not None:
        payload["legacy_after_state_id"] = record.after_state_id
    if record.source == EventSource.UNOBSERVABLE:
        payload["legacy_value_unobservable"] = True
    return payload


def _migrate_attribute(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {"name": name, "value_state": "unknown"}
    if isinstance(value, float):
        decimal_value = Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError(f"旧属性の浮動小数が有限ではありません: {name}")
        return {"name": name, "value_decimal": format(decimal_value, "f")}
    return {"name": name, "value": value}


__all__ = ["LegacyMigrationContext", "migrate_legacy_group"]
