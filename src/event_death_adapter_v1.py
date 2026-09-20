"""時間的死亡確認sidecarを厳格に読み、出来事原本へ渡せる形へ正規化する。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.event_snapshot_adapter_v1 import VideoTimeBase
from src.event_source_v1 import SCHEMA_VERSION


DEATH_SIDECAR_VERSION = "event-death-observation-sidecar/v1"
DEATH_ADAPTER_VERSION = "event-death-adapter/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SIDES = ("p1", "p2")
_STATES = frozenset({"clear", "pending", "released", "confirmed"})
_SOURCES = frozenset({None, "placement", "ojama"})
_TRANSITIONS = frozenset({
    None,
    "candidate_placement",
    "candidate_ojama",
    "released_placement",
    "released_ojama",
    "released_survival_placement",
    "released_survival_ojama",
    "confirmed_placement",
    "confirmed_ojama",
})
_BOUNDARY_OUTCOMES = frozenset({
    None,
    "no_candidate",
    "confirmed",
    "suppressed_ambiguous",
    "rejected_game_idx_mismatch",
    "rejected_survival_evidence",
})
_DECODE_STATUS_COMPLETE = "requested_range_complete"
_DECODE_STATUS_EARLY_END = "early_eos_or_decode_failure"
_DECODE_STATUS_LEGACY = "legacy_unspecified"


class EventDeathAdapterError(ValueError):
    """死亡sidecarを因果的に利用できない場合の例外。"""


@dataclass(frozen=True, slots=True)
class DeathSideState:
    """一観測フレームにおける片側の時間的死亡状態。"""

    state: str
    transition: str | None
    source: str | None
    pending_elapsed_ms: int | None
    boundary_outcome: str | None
    confirmation_delay_ms: int | None = None


@dataclass(frozen=True, slots=True)
class DeathBoundaryState:
    """左右を一回で解決した正式境界の記録。"""

    occurred: bool
    closing_game_idx: int | None
    opening_game_idx: int | None
    outcome_p1: str | None
    outcome_p2: str | None


@dataclass(frozen=True, slots=True)
class DeathObservationRow:
    """dense sidecarの一観測フレーム。"""

    frame_idx: int
    available_frame: int
    available_ms: int
    game_idx: int
    death_generation: int
    p1: DeathSideState
    p2: DeathSideState
    boundary: DeathBoundaryState


@dataclass(frozen=True, slots=True)
class EventDeathSidecar:
    """外部SHAへ束縛できる時間的死亡確認sidecar。"""

    observer_version: str
    source_video_id: str
    source_video_sha256: str
    processing_start_frame: int
    processing_end_frame_exclusive: int
    time_base_numerator: int
    time_base_denominator: int
    sample_interval_frames: int
    observation_rate_hz: float
    stationary_confirm_sec: float
    observed_frame_count: int
    inspected_side_count: int
    event_count: int
    death_generation: int
    pending_at_end_p1: bool
    pending_at_end_p2: bool
    counters: tuple[tuple[str, int], ...]
    rows: tuple[DeathObservationRow, ...]
    content_sha256: str
    # 旧v1 sidecarでは両方とも未記録なので、None/legacyで読取互換を保つ。
    requested_end_frame_exclusive: int | None = None
    decode_status: str = _DECODE_STATUS_LEGACY


class _DeathEventBuilder:
    """dense状態列から変化行だけを追記出来事へ変換する。"""

    def __init__(
        self, sidecar: EventDeathSidecar, source: str, build: str,
        attempt: str, time_base: VideoTimeBase,
    ) -> None:
        self.sidecar = sidecar
        self.source = source
        self.build = build
        self.attempt = attempt
        self.time_base = time_base
        self.events: list[dict[str, Any]] = []
        self.candidates: dict[str, tuple[str, int]] = {}

    def build_events(self) -> tuple[dict[str, Any], ...]:
        for row in self.sidecar.rows:
            # observerは旧gameの境界解決後に現frameを更新する。同一frameに
            # 新gameの遷移があっても旧candidateと混同しないよう順序を保つ。
            if row.boundary.occurred:
                self._append_boundary(row)
            self._append_transition(row, "p1", row.p1)
            self._append_transition(row, "p2", row.p2)
        return tuple(self.events)

    def _append_transition(
        self, row: DeathObservationRow, side: str, state: DeathSideState,
    ) -> None:
        transition = state.transition
        if transition is None:
            return
        if transition.startswith("candidate_"):
            token = self._candidate_token(side, row.death_generation)
            self.candidates[side] = (token, row.frame_idx)
            event = self._base_event(row, side, "death_candidate", row.frame_idx)
            event["assertion"] = {"state": "provisional", "value_form": "exact"}
            event["payload"].update({
                "provisional_observation_id": token,
                "candidate_source": state.source,
                "death_generation": row.death_generation,
            })
            self.events.append(event)
            return
        event_type = (
            "death_confirmed" if transition.startswith("confirmed_")
            else "death_candidate_released"
        )
        self._append_resolution(row, side, state, event_type, transition)

    def _append_resolution(
        self, row: DeathObservationRow, side: str, state: DeathSideState,
        event_type: str, transition: str,
    ) -> None:
        candidate = self.candidates.pop(side, None)
        earliest = row.frame_idx if candidate is None else candidate[1]
        event = self._base_event(row, side, event_type, earliest)
        payload = {
            "candidate_source": state.source,
            "death_generation": row.death_generation,
            "transition": transition,
        }
        if state.confirmation_delay_ms is not None:
            payload["confirmation_delay_ms"] = state.confirmation_delay_ms
        event["payload"].update(payload)
        if candidate is None:
            event["missing_information"].append("candidate_event_not_in_processing_range")
        else:
            event["relations"]["revision"] = {
                "action": "confirms" if event_type == "death_confirmed" else "revokes",
                "target_event_ids": [],
                "target_observation_ids": [candidate[0]],
            }
        self.events.append(event)

    def _append_boundary(self, row: DeathObservationRow) -> None:
        boundary = row.boundary
        event = self._base_event(row, "system", "death_boundary_resolved", row.frame_idx)
        event["payload"] = {
            "closing_game_idx": boundary.closing_game_idx,
            "opening_game_idx": boundary.opening_game_idx,
            "outcomes": {"p1": boundary.outcome_p1, "p2": boundary.outcome_p2},
            "death_generation": row.death_generation,
        }
        targets = [candidate[0] for candidate in self.candidates.values()]
        if targets:
            event["relations"]["revision"] = {
                "action": "resolves_at_boundary",
                "target_event_ids": [],
                "target_observation_ids": targets,
            }
        self.candidates.clear()
        self.events.append(event)

    def _base_event(
        self, row: DeathObservationRow, side: str | None,
        event_type: str, earliest_frame: int,
    ) -> dict[str, Any]:
        timing = {
            "occurred_earliest_frame": earliest_frame,
            "occurred_earliest_ms": self.time_base.frame_to_ms(earliest_frame),
            "occurred_latest_frame": row.frame_idx,
            "occurred_latest_ms": row.available_ms,
            "available_frame": row.available_frame,
            "available_ms": row.available_ms,
        }
        return {
            "record_kind": "event", "schema_version": SCHEMA_VERSION,
            "source_video_id": self.source, "build_id": self.build,
            "attempt_id": self.attempt, "seq": 0, "event_id": f"{self.build}:0",
            "availability_batch_id": f"{self.build}:unsequenced",
            "batch_index": 0, "batch_size": 1, "event_type": event_type, "side": side,
            "timing": timing, "assertion": {"state": "confirmed", "value_form": "exact"},
            "evidence": [{
                "evidence_type": "temporal_death_confirmation",
                "method_id": self.sidecar.observer_version,
                "method_version": DEATH_ADAPTER_VERSION,
                "from_frame": earliest_frame, "to_frame": row.frame_idx,
            }],
            "checks": [{
                "check_id": "death_sidecar_integrity",
                "check_version": DEATH_ADAPTER_VERSION,
                "result": "pass", "reason_codes": [],
            }],
            "missing_information": [],
            "relations": {"revision": {"action": "none", "target_event_ids": []}},
            "heavy_evidence_refs": [],
            "payload": {"game_idx": row.game_idx},
        }

    def _candidate_token(self, side: str, generation: int) -> str:
        return f"{self.build}:death-candidate-{side}-{generation:06d}"


def canonical_death_sidecar_bytes(value: Mapping[str, Any]) -> bytes:
    """死亡sidecarの唯一の直列化を返す。"""

    try:
        text = json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise EventDeathAdapterError("死亡sidecarを正規化できません") from error
    return text.encode("utf-8") + b"\n"


def parse_event_death_sidecar_bytes(
    payload: bytes, *, require_decode_contract: bool = False,
) -> EventDeathSidecar:
    """canonical JSONを検査する。strict指定時はlegacy decode契約を拒否する。"""

    value = _decode_value(payload)
    if canonical_death_sidecar_bytes(value) != payload:
        raise EventDeathAdapterError("死亡sidecarがcanonical bytesではありません")
    header = _parse_header(value)
    if require_decode_contract and header["requested_end"] is None:
        raise EventDeathAdapterError("正式利用には死亡sidecarの新decode契約が必要です")
    rows = _parse_rows(value.get("events"), header)
    _validate_totals(value, rows, header)
    pending = _mapping(value.get("pending_at_end_of_stream"), "終了時pending")
    counters = _parse_counters(value.get("counters"))
    return EventDeathSidecar(
        observer_version=_text(value.get("observer_version"), "observer_version"),
        source_video_id=header["source_video_id"],
        source_video_sha256=header["source_video_sha256"],
        processing_start_frame=header["start"],
        processing_end_frame_exclusive=header["end"],
        time_base_numerator=header["numerator"],
        time_base_denominator=header["denominator"],
        sample_interval_frames=header["interval"],
        observation_rate_hz=_positive_number(value.get("observation_rate_hz"), "観測rate"),
        stationary_confirm_sec=_positive_number(
            value.get("stationary_confirm_sec"), "死亡確認秒数",
        ),
        observed_frame_count=_integer(value.get("observed_frame_count"), "観測frame数"),
        inspected_side_count=_integer(value.get("inspected_side_count"), "観測side数"),
        event_count=_integer(value.get("event_count"), "event数"),
        death_generation=_integer(value.get("death_generation"), "最終generation"),
        pending_at_end_p1=_boolean(pending.get("p1"), "1P終了時pending"),
        pending_at_end_p2=_boolean(pending.get("p2"), "2P終了時pending"),
        counters=counters,
        rows=rows,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        requested_end_frame_exclusive=header["requested_end"],
        decode_status=header["decode_status"],
    )


def load_event_death_sidecar(
    path: Path, *, expected_sha256: str | None = None,
    require_decode_contract: bool = False,
) -> EventDeathSidecar:
    """実bytesを期待SHAへ束縛し、任意で新decode契約を必須化する。"""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EventDeathAdapterError(f"死亡sidecarを読めません: {path}") from error
    actual = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise EventDeathAdapterError("死亡sidecarの期待SHA-256が不正です")
        if actual != expected_sha256:
            raise EventDeathAdapterError("死亡sidecarの実bytesが期待SHA-256と一致しません")
    return parse_event_death_sidecar_bytes(
        payload, require_decode_contract=require_decode_contract,
    )


def build_event_death_events(
    sidecar: EventDeathSidecar, *, source_video_id: str,
    build_id: str, attempt_id: str, time_base: VideoTimeBase,
) -> tuple[dict[str, Any], ...]:
    """時間的死亡確認の変化だけを出来事原本行へ変換する。"""

    if source_video_id != sidecar.source_video_id:
        raise EventDeathAdapterError("死亡sidecarの元映像IDがexport要求と一致しません")
    if (
        sidecar.time_base_numerator != time_base.numerator
        or sidecar.time_base_denominator != time_base.denominator
    ):
        raise EventDeathAdapterError("死亡sidecarのtimebaseがexport要求と一致しません")
    builder = _DeathEventBuilder(
        sidecar, source_video_id, build_id, attempt_id, time_base,
    )
    return builder.build_events()


def _decode_value(payload: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EventDeathAdapterError("死亡sidecarがUTF-8 JSONではありません") from error
    return _mapping(value, "死亡sidecar")


def _parse_header(value: Mapping[str, Any]) -> dict[str, Any]:
    """固定headerを読み、sidecar完了とdecode完走を別契約で検査する。"""
    if value.get("schema_version") != DEATH_SIDECAR_VERSION:
        raise EventDeathAdapterError("死亡sidecarのschema versionが不正です")
    # completion/end_of_streamはsidecar自体が原子的にfinalize済みである契約。
    # 要求区間のdecode完走は下のdecode_statusで独立に検査する。
    if value.get("completion") != "complete" or value.get("end_of_stream") is not True:
        raise EventDeathAdapterError("死亡sidecarが完走状態ではありません")
    timebase = _mapping(value.get("timebase"), "timebase")
    header = {
        "source_video_id": _text(value.get("source_video_id"), "source video ID"),
        "source_video_sha256": _sha256(value.get("source_video_sha256")),
        "start": _integer(value.get("processing_start_frame"), "処理開始frame"),
        "end": _integer(value.get("processing_end_frame_exclusive"), "処理終了frame"),
        "numerator": _integer(timebase.get("numerator"), "timebase numerator"),
        "denominator": _integer(timebase.get("denominator"), "timebase denominator"),
        "interval": _integer(value.get("sample_interval_frames"), "観測間隔"),
    }
    if header["start"] < 0 or header["end"] <= header["start"]:
        raise EventDeathAdapterError("死亡sidecarの処理範囲が不正です")
    if header["numerator"] <= 0 or header["denominator"] <= 0 or header["interval"] <= 0:
        raise EventDeathAdapterError("死亡sidecarのtimebaseまたは観測間隔が不正です")
    requested_end, decode_status = _parse_decode_completion(value, header["end"])
    header["requested_end"] = requested_end
    header["decode_status"] = decode_status
    return header


def _parse_decode_completion(
    value: Mapping[str, Any], processing_end: int,
) -> tuple[int | None, str]:
    """新decode契約を検査し、旧sidecarはlegacyとして読み続ける。"""
    has_requested = "requested_end_frame_exclusive" in value
    has_status = "decode_status" in value
    if not has_requested and not has_status:
        return None, _DECODE_STATUS_LEGACY
    if has_requested != has_status:
        raise EventDeathAdapterError("死亡sidecarのdecode契約が片側だけです")
    requested = _integer(
        value.get("requested_end_frame_exclusive"), "要求終了frame",
    )
    status = value.get("decode_status")
    if requested < processing_end:
        raise EventDeathAdapterError("死亡sidecarの要求終了frameが処理終了より前です")
    expected = (
        _DECODE_STATUS_COMPLETE
        if requested == processing_end else _DECODE_STATUS_EARLY_END
    )
    if status != expected:
        raise EventDeathAdapterError("死亡sidecarのdecode状態と処理範囲が矛盾します")
    if status == _DECODE_STATUS_EARLY_END:
        raise EventDeathAdapterError("死亡sidecarは要求区間をdecode完走していません")
    return requested, str(status)


def _parse_rows(value: Any, header: Mapping[str, Any]) -> tuple[DeathObservationRow, ...]:
    if not isinstance(value, list) or not value:
        raise EventDeathAdapterError("死亡sidecarにdense観測行がありません")
    require_confirmation = header["requested_end"] is not None
    rows = tuple(
        _parse_row(item, require_confirmation_contract=require_confirmation)
        for item in value
    )
    _validate_row_order(rows, header)
    return rows


def _parse_row(
    value: Any, *, require_confirmation_contract: bool = False,
) -> DeathObservationRow:
    row = _mapping(value, "死亡観測行")
    sides = _mapping(row.get("sides"), "死亡観測の左右状態")
    boundary = _parse_boundary(row.get("boundary"))
    return DeathObservationRow(
        frame_idx=_integer(row.get("frame_idx"), "観測frame"),
        available_frame=_integer(row.get("available_frame"), "利用可能frame"),
        available_ms=_integer(row.get("available_ms"), "利用可能ms"),
        game_idx=_integer(row.get("game_idx"), "game index"),
        death_generation=_integer(row.get("death_generation"), "death generation"),
        p1=_parse_side(
            sides.get("p1"), "p1", require_confirmation_contract,
        ),
        p2=_parse_side(
            sides.get("p2"), "p2", require_confirmation_contract,
        ),
        boundary=boundary,
    )


def _parse_side(
    value: Any, side: str, require_confirmation_contract: bool = False,
) -> DeathSideState:
    item = _mapping(value, f"{side}死亡状態")
    state = item.get("state")
    transition = item.get("transition")
    source = item.get("source")
    outcome = item.get("boundary_outcome")
    if state not in _STATES or transition not in _TRANSITIONS:
        raise EventDeathAdapterError(f"{side}の死亡状態または遷移が不正です")
    if source not in _SOURCES or outcome not in _BOUNDARY_OUTCOMES:
        raise EventDeathAdapterError(f"{side}の死亡由来または境界結果が不正です")
    elapsed = item.get("pending_elapsed_ms")
    if elapsed is not None:
        elapsed = _integer(elapsed, f"{side} pending経過ms")
        if elapsed < 0:
            raise EventDeathAdapterError(f"{side} pending経過msが負です")
    has_confirmation = "confirmation_delay_ms" in item
    confirmation = item.get("confirmation_delay_ms", elapsed)
    if confirmation is not None:
        confirmation = _integer(confirmation, f"{side} 確定遅延ms")
        if confirmation < 0:
            raise EventDeathAdapterError(f"{side} 確定遅延msが負です")
    is_confirmed = isinstance(transition, str) and transition.startswith("confirmed_")
    if require_confirmation_contract and is_confirmed and not has_confirmation:
        raise EventDeathAdapterError(f"{side} confirmed行に確定遅延fieldがありません")
    if has_confirmation and is_confirmed and confirmation is None:
        raise EventDeathAdapterError(f"{side} confirmed行に確定遅延がありません")
    if has_confirmation and not is_confirmed and confirmation is not None:
        raise EventDeathAdapterError(f"{side} 非confirmed行に確定遅延があります")
    return DeathSideState(
        str(state), transition, source, elapsed, outcome, confirmation,
    )


def _parse_boundary(value: Any) -> DeathBoundaryState:
    item = _mapping(value, "死亡境界状態")
    occurred = _boolean(item.get("occurred"), "境界発生")
    outcomes = _mapping(item.get("outcomes"), "境界左右結果")
    closing = _optional_integer(item.get("closing_game_idx"), "終了game index")
    opening = _optional_integer(item.get("opening_game_idx"), "開始game index")
    p1 = _boundary_outcome(outcomes.get("p1"), "p1")
    p2 = _boundary_outcome(outcomes.get("p2"), "p2")
    if occurred and (closing is None or opening != closing + 1):
        raise EventDeathAdapterError("死亡境界のgame indexが連続していません")
    if not occurred and any(item is not None for item in (closing, opening, p1, p2)):
        raise EventDeathAdapterError("境界なし行に境界結果が入っています")
    return DeathBoundaryState(occurred, closing, opening, p1, p2)


def _boundary_outcome(value: Any, side: str) -> str | None:
    if value is None:
        return None
    item = _mapping(value, f"{side}境界結果")
    outcome = item.get("outcome")
    if outcome not in _BOUNDARY_OUTCOMES - {None}:
        raise EventDeathAdapterError(f"{side}境界結果が不正です")
    source = item.get("source")
    if source not in _SOURCES:
        raise EventDeathAdapterError(f"{side}境界由来が不正です")
    return str(outcome)


def _validate_row_order(
    rows: tuple[DeathObservationRow, ...], header: Mapping[str, Any],
) -> None:
    previous_frame = header["start"] - 1
    previous_ms = -1
    previous_generation = 0
    for row in rows:
        if row.frame_idx != row.available_frame:
            raise EventDeathAdapterError("死亡観測frameと利用可能frameが一致しません")
        if not header["start"] <= row.frame_idx < header["end"]:
            raise EventDeathAdapterError("死亡観測行が処理範囲外です")
        if row.frame_idx <= previous_frame or row.available_ms < previous_ms:
            raise EventDeathAdapterError("死亡観測行の公開位置が逆行または重複しました")
        expected_ms = (
            row.frame_idx * 1000 * header["numerator"] // header["denominator"]
        )
        if row.available_ms != expected_ms:
            raise EventDeathAdapterError("死亡観測行のtimebaseが一致しません")
        if row.game_idx < 0 or row.death_generation < previous_generation:
            raise EventDeathAdapterError("死亡観測行のgameまたはgenerationが逆行しました")
        if row.death_generation - previous_generation > 1:
            raise EventDeathAdapterError("death generationが一度に2以上進みました")
        previous_frame, previous_ms = row.frame_idx, row.available_ms
        previous_generation = row.death_generation


def _validate_totals(
    value: Mapping[str, Any], rows: tuple[DeathObservationRow, ...],
    header: Mapping[str, Any],
) -> None:
    observed = _integer(value.get("observed_frame_count"), "観測frame数")
    inspected = _integer(value.get("inspected_side_count"), "観測side数")
    event_count = _integer(value.get("event_count"), "event数")
    generation = _integer(value.get("death_generation"), "最終generation")
    if observed != len(rows) or event_count != len(rows) or inspected != 2 * observed:
        raise EventDeathAdapterError("死亡sidecarの母数がdense観測行と一致しません")
    if rows[-1].death_generation != generation:
        raise EventDeathAdapterError("死亡sidecarの最終generationが一致しません")
    if rows[0].frame_idx < header["start"] or rows[-1].frame_idx >= header["end"]:
        raise EventDeathAdapterError("死亡sidecarの先頭または末尾行が処理範囲外です")


def _parse_counters(value: Any) -> tuple[tuple[str, int], ...]:
    counters = _mapping(value, "死亡確認counter")
    parsed: list[tuple[str, int]] = []
    for key, raw in counters.items():
        if not isinstance(key, str) or not key:
            raise EventDeathAdapterError("死亡確認counter名が不正です")
        number = _integer(raw, f"死亡確認counter {key}")
        if number < 0:
            raise EventDeathAdapterError("死亡確認counterが負です")
        parsed.append((key, number))
    return tuple(sorted(parsed))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EventDeathAdapterError(f"{name}はJSON objectでなければなりません")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EventDeathAdapterError(f"{name}は空でない文字列でなければなりません")
    return value


def _sha256(value: Any) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise EventDeathAdapterError("元映像SHA-256が不正です")
    return value


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EventDeathAdapterError(f"{name}は整数でなければなりません")
    return value


def _optional_integer(value: Any, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise EventDeathAdapterError(f"{name}は真偽値でなければなりません")
    return value


def _positive_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise EventDeathAdapterError(f"{name}は0より大きい数値でなければなりません")
    return float(value)


__all__ = [
    "DEATH_ADAPTER_VERSION",
    "DEATH_SIDECAR_VERSION",
    "DeathBoundaryState",
    "DeathObservationRow",
    "DeathSideState",
    "EventDeathAdapterError",
    "EventDeathSidecar",
    "build_event_death_events",
    "canonical_death_sidecar_bytes",
    "load_event_death_sidecar",
    "parse_event_death_sidecar_bytes",
]
