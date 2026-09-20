"""旧出来事試作の互換読取用契約。

新規原本は ``event_source_v1`` を使う。このモジュールは既存試験と旧メモリ上データを
一方向移行するために残し、新しい永続成果物へ直接書き出さない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


LEGACY_CONTRACT_DEPRECATED = True


class EventSource(StrEnum):
    """出来事を得た方法。"""

    DIRECT_OBSERVATION = "direct_observation"
    RULE_DERIVED = "rule_derived"
    MULTI_SIGNAL_ESTIMATE = "multi_signal_estimate"
    SINGLE_SIGNAL_ESTIMATE = "single_signal_estimate"
    UNOBSERVABLE = "unobservable"


class EventStatus(StrEnum):
    """出来事の確定状態。"""

    PROVISIONAL = "provisional"
    FINAL = "final"
    REVOKED = "revoked"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class EventSide(StrEnum):
    """出来事が属する側。"""

    P1 = "1P"
    P2 = "2P"
    SYSTEM = "system"


class EventType(StrEnum):
    """先行測定で保存する物理的な出来事。"""

    MATCH_START_CANDIDATE = "match_start_candidate"
    MATCH_START_CONFIRMED = "match_start_confirmed"
    MATCH_END_CANDIDATE = "match_end_candidate"
    WINNER_CONFIRMED = "winner_confirmed"
    CONTROL_GAINED = "control_gained"
    CONTROL_LOST = "control_lost"
    NEXT_SHIFT = "next_shift"
    PLACEMENT_COMPLETED = "placement_completed"
    CHAIN_STARTED = "chain_started"
    CHAIN_STEP_CONFIRMED = "chain_step_confirmed"
    CHAIN_PHYSICAL_END = "chain_physical_end"
    ATTACK_PROVISIONAL_UPDATED = "attack_provisional_updated"
    ATTACK_FINALIZED = "attack_finalized"
    ATTACK_CANCELED = "attack_canceled"
    GARBAGE_COMMITTED_TO_FALL = "garbage_committed_to_fall"
    GARBAGE_FALL_STARTED = "garbage_fall_started"
    GARBAGE_FALL_COMPLETED = "garbage_fall_completed"
    ALL_CLEAR_GAINED = "all_clear_gained"
    ALL_CLEAR_CONSUMED = "all_clear_consumed"
    DEATH_CANDIDATE = "death_candidate"
    DEATH_CONFIRMED = "death_confirmed"
    DATA_GAP_STARTED = "data_gap_started"
    DATA_GAP_ENDED = "data_gap_ended"


EventValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class EventTiming:
    """物理発生・観測・利用可能・確定の時刻を分離する。"""

    observed_at_sec: float
    available_at_sec: float
    occurred_start_sec: float | None = None
    occurred_end_sec: float | None = None
    finalized_at_sec: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.observed_at_sec,
            self.available_at_sec,
            self.occurred_start_sec,
            self.occurred_end_sec,
            self.finalized_at_sec,
        )
        if any(value is not None and value < 0.0 for value in values):
            raise ValueError("時刻は0以上でなければならない")
        if self.available_at_sec < self.observed_at_sec:
            raise ValueError("利用可能時刻を観測時刻より前にできない")
        if self._has_reversed_occurrence_interval():
            raise ValueError("物理発生区間の終了を開始より前にできない")
        if self._finalized_after_availability():
            raise ValueError("確定前の情報を利用可能にはできない")

    def _has_reversed_occurrence_interval(self) -> bool:
        return (
            self.occurred_start_sec is not None
            and self.occurred_end_sec is not None
            and self.occurred_end_sec < self.occurred_start_sec
        )

    def _finalized_after_availability(self) -> bool:
        return (
            self.finalized_at_sec is not None
            and self.finalized_at_sec > self.available_at_sec
        )


@dataclass(frozen=True, slots=True)
class EventRecord:
    """上書きしない単一出来事。"""

    event_id: str
    event_group_id: str
    match_id: str
    side: EventSide
    event_type: EventType
    timing: EventTiming
    source: EventSource
    status: EventStatus
    before_state_id: str | None = None
    after_state_id: str | None = None
    chain_id: str | None = None
    attack_id: str | None = None
    cause_event_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    confidence: float | None = None
    attributes: tuple[tuple[str, EventValue], ...] = ()
    revision_of: str | None = None

    def __post_init__(self) -> None:
        self._validate_required_ids()
        self._validate_confidence()
        self._validate_status_timing()
        self._validate_attributes()
        if self.event_id in self.cause_event_ids:
            raise ValueError("出来事自身を原因にはできない")
        if self.revision_of == self.event_id:
            raise ValueError("出来事自身を訂正対象にはできない")

    def _validate_required_ids(self) -> None:
        required = (self.event_id, self.event_group_id, self.match_id)
        if any(not value.strip() for value in required):
            raise ValueError("出来事・群・試合の識別子は空にできない")

    def _validate_confidence(self) -> None:
        if self.confidence is None:
            return
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("信頼度は0から1の範囲でなければならない")

    def _validate_status_timing(self) -> None:
        finalized = self.timing.finalized_at_sec
        if self.status == EventStatus.PROVISIONAL and finalized is not None:
            raise ValueError("暫定出来事に確定時刻は設定できない")
        if self.status != EventStatus.PROVISIONAL and finalized is None:
            raise ValueError("暫定以外の出来事には確定時刻が必要")
        if self.status == EventStatus.REVOKED and self.revision_of is None:
            raise ValueError("取消記録には訂正対象が必要")

    def _validate_attributes(self) -> None:
        names = [name for name, _ in self.attributes]
        if any(not name.strip() for name in names):
            raise ValueError("属性名は空にできない")
        if len(names) != len(set(names)):
            raise ValueError("同じ属性名を重複して保存できない")


@dataclass(slots=True)
class EventLedger:
    """同時出来事群を原子的に追記する台帳。"""

    _events: list[EventRecord] = field(default_factory=list, init=False)
    _by_id: dict[str, EventRecord] = field(default_factory=dict, init=False)
    _last_available_at_sec: float = field(default=-1.0, init=False)

    @property
    def events(self) -> tuple[EventRecord, ...]:
        return tuple(self._events)

    def append_group(self, events: tuple[EventRecord, ...]) -> None:
        """同時出来事群を検証して一括追記する。"""

        if not events:
            raise ValueError("空の出来事群は追加できない")
        self._validate_group_identity(events)
        self._validate_new_ids(events)
        self._validate_monotonic_availability(events[0])
        self._validate_references(events)
        for event in events:
            self._events.append(event)
            self._by_id[event.event_id] = event
        self._last_available_at_sec = events[0].timing.available_at_sec

    def prefix_until(self, available_at_sec: float) -> tuple[EventRecord, ...]:
        """指定時刻までに利用可能だった出来事だけを返す。"""

        return tuple(
            event
            for event in self._events
            if event.timing.available_at_sec <= available_at_sec
        )

    @staticmethod
    def _validate_group_identity(events: tuple[EventRecord, ...]) -> None:
        first = events[0]
        group_keys = {
            (event.event_group_id, event.match_id, event.timing.available_at_sec)
            for event in events
        }
        if group_keys != {
            (first.event_group_id, first.match_id, first.timing.available_at_sec)
        }:
            raise ValueError("同時出来事群は群・試合・利用可能時刻を共有する必要がある")

    def _validate_new_ids(self, events: tuple[EventRecord, ...]) -> None:
        new_ids = [event.event_id for event in events]
        if len(new_ids) != len(set(new_ids)):
            raise ValueError("同じ群に出来事識別子の重複がある")
        if any(event_id in self._by_id for event_id in new_ids):
            raise ValueError("既存出来事を上書きできない")

    def _validate_monotonic_availability(self, event: EventRecord) -> None:
        if event.timing.available_at_sec < self._last_available_at_sec:
            raise ValueError("利用可能時刻を過去へ戻して追記できない")

    def _validate_references(self, events: tuple[EventRecord, ...]) -> None:
        group_ids = {event.event_id for event in events}
        allowed_ids = set(self._by_id) | group_ids
        for event in events:
            if any(cause not in allowed_ids for cause in event.cause_event_ids):
                raise ValueError("存在しない原因出来事は参照できない")
            if event.revision_of is not None and event.revision_of not in self._by_id:
                raise ValueError("訂正対象は過去に記録済みでなければならない")


__all__ = [
    "EventLedger",
    "EventRecord",
    "EventSide",
    "EventSource",
    "EventStatus",
    "EventTiming",
    "EventType",
    "EventValue",
    "LEGACY_CONTRACT_DEPRECATED",
]
