"""死亡候補を未来参照なしで記録する canonical sidecar observer。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping, TypeAlias

from src.board import Board
from src.death_confirmation import (
    DeathConfirmStats,
    DeathConfirmTracker,
    resolve_boundary_confirmations,
)

SCHEMA_VERSION: str = "event-death-observation-sidecar/v1"
OBSERVER_VERSION: str = "event-death-observer/v1"
SIDES: tuple[str, str] = ("p1", "p2")
STATE_STABLE: str = "STABLE"
TARGET_OBSERVATION_RATE_HZ: float = 30.0
OBSERVATION_RATE_TOLERANCE_HZ: float = 0.1
SHA256_PATTERN: re.Pattern[str] = re.compile(r"[0-9a-f]{64}")
DECODE_STATUS_COMPLETE: str = "requested_range_complete"
DECODE_STATUS_EARLY_END: str = "early_eos_or_decode_failure"

SideInput: TypeAlias = tuple[object, Board | None, object]
BoundaryResult: TypeAlias = dict[str, object]


def canonical_sidecar_bytes(value: Mapping[str, object]) -> bytes:
    """JSON表現をhash可能な一意のUTF-8バイト列へ固定する。"""
    text = json.dumps(
        value, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def write_sidecar_exclusive(path: Path, value: Mapping[str, object]) -> None:
    """既存成果物を変更せず、新規sidecarだけを排他的に作成する。"""
    payload = canonical_sidecar_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)


def _state_name(value: object) -> str:
    """BoardState/文字列をDeathConfirmTrackerの公開契約へ正規化する。"""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    if isinstance(value, str) and value:
        return value.upper()
    raise ValueError("state は非空の BoardState または文字列である必要があります")


def _source_from_event(event: str | None) -> str | None:
    if event is None:
        return None
    if event.endswith("_placement"):
        return "placement"
    if event.endswith("_ojama"):
        return "ojama"
    raise ValueError(f"未知の死亡遷移です: {event}")


class EventDeathRecorder:
    """両sideを同一generationで観測するstatefulな収集wrapper。"""

    def __init__(
        self, source_video_id: str, source_video_sha256: str,
        timebase_numerator: int, timebase_denominator: int,
        sample_interval_frames: int, observation_rate_hz: float,
        stationary_confirm_sec: float = 1.5,
    ) -> None:
        self._validate_contract(
            source_video_id, source_video_sha256, timebase_numerator,
            timebase_denominator, sample_interval_frames, observation_rate_hz,
            stationary_confirm_sec,
        )
        self.source_video_id = source_video_id
        self.source_video_sha256 = source_video_sha256
        self.timebase = {"numerator": timebase_numerator, "denominator": timebase_denominator}
        self.sample_interval_frames = sample_interval_frames
        self.observation_rate_hz = observation_rate_hz
        self.stationary_confirm_sec = stationary_confirm_sec
        self._trackers = {side: DeathConfirmTracker(stationary_confirm_sec) for side in SIDES}
        self._stats = DeathConfirmStats()
        self._events: list[dict[str, object]] = []
        self._phase = {side: "clear" for side in SIDES}
        self._source: dict[str, str | None] = {side: None for side in SIDES}
        self._game_idx: int | None = None
        self._last_frame_idx: int | None = None
        self._last_t_sec: float | None = None
        self._generation = 0
        self._match_evidence_rejected_frame_count = 0

    @staticmethod
    def _validate_contract(
        source_id: str, source_sha: str, numerator: int, denominator: int,
        interval: int, rate_hz: float, confirm_sec: float,
    ) -> None:
        if not source_id:
            raise ValueError("source_video_id は空にできません")
        if SHA256_PATTERN.fullmatch(source_sha) is None:
            raise ValueError("source_video_sha256 は小文字64桁hexが必要です")
        if numerator <= 0 or denominator <= 0:
            raise ValueError("timebase は正の分子・分母が必要です")
        if interval <= 0:
            raise ValueError("sample_interval_frames は正数が必要です")
        if not math.isclose(
            rate_hz, TARGET_OBSERVATION_RATE_HZ,
            abs_tol=OBSERVATION_RATE_TOLERANCE_HZ,
        ):
            raise ValueError("observation_rate_hz は30fps相当が必要です")
        if not math.isfinite(confirm_sec) or confirm_sec <= 0:
            raise ValueError("stationary_confirm_sec は正の有限値が必要です")

    def observe(
        self, frame_idx: int, t_sec: float, game_idx: int,
        sides: Mapping[str, SideInput], is_match_active: bool = True,
        match_evidence: bool = True,
    ) -> None:
        """境界を旧gameへ原子適用した後、両sideの現frameを記録する。"""
        normalized = self._validate_observation(frame_idx, t_sec, game_idx, sides)
        boundary = self._resolve_boundary(game_idx)
        if not match_evidence:
            self._match_evidence_rejected_frame_count += 1
        rows: dict[str, dict[str, object]] = {}
        transitions: list[str | None] = []
        for side in SIDES:
            state, board, next_key = normalized[side]
            event, delay = self._trackers[side].update(
                state, self._death_cell(state, board), t_sec, next_key,
                is_match_active=is_match_active, game_idx=game_idx,
                match_evidence=match_evidence,
            )
            self._stats.record(event, delay)
            transitions.append(event)
            rows[side] = self._side_value(side, event, delay, t_sec, boundary)
        if boundary["occurred"] or any(event is not None for event in transitions):
            self._generation += 1
        self._events.append(self._event_value(frame_idx, game_idx, rows, boundary))
        self._last_frame_idx, self._last_t_sec = frame_idx, t_sec

    def _validate_observation(
        self, frame_idx: int, t_sec: float, game_idx: int,
        sides: Mapping[str, SideInput],
    ) -> dict[str, tuple[str, Board | None, object]]:
        if self._last_frame_idx is not None and frame_idx <= self._last_frame_idx:
            raise ValueError("frame_idx は単調増加が必要です")
        if not math.isfinite(t_sec) or (self._last_t_sec is not None and t_sec <= self._last_t_sec):
            raise ValueError("t_sec は有限かつ単調増加が必要です")
        if game_idx < 0:
            raise ValueError("game_idx は0以上が必要です")
        if set(sides) != set(SIDES):
            raise ValueError("sides はp1/p2のexact setが必要です")
        result: dict[str, tuple[str, Board | None, object]] = {}
        for side in SIDES:
            state, board, next_key = sides[side]
            normalized = _state_name(state)
            if normalized == STATE_STABLE and not isinstance(board, Board):
                raise ValueError(f"{side} STABLEにはconfirmed_boardが必要です")
            result[side] = normalized, board, next_key
        return result

    @staticmethod
    def _death_cell(state: str, board: Board | None) -> bool:
        """STABLE確定盤面以外へ静的死亡判定を適用しない。"""
        return bool(board.is_dead()) if state == STATE_STABLE and board is not None else False

    def _resolve_boundary(self, game_idx: int) -> BoundaryResult:
        old_idx = self._game_idx
        if old_idx is None:
            self._game_idx = game_idx
            return self._empty_boundary()
        if game_idx == old_idx:
            return self._empty_boundary()
        if game_idx != old_idx + 1:
            raise ValueError("game_idx は同値または1増分だけが許可されます")
        p1, p2 = resolve_boundary_confirmations(
            self._trackers["p1"], self._trackers["p2"],
            game_idx=old_idx, stats=self._stats,
        )
        outcomes = {"p1": self._outcome_value(p1), "p2": self._outcome_value(p2)}
        self._phase = {side: "clear" for side in SIDES}
        self._source = {side: None for side in SIDES}
        self._game_idx = game_idx
        return {
            "occurred": True, "closing_game_idx": old_idx,
            "opening_game_idx": game_idx, "outcomes": outcomes,
        }

    @staticmethod
    def _empty_boundary() -> BoundaryResult:
        return {
            "occurred": False, "closing_game_idx": None,
            "opening_game_idx": None, "outcomes": {},
        }

    @staticmethod
    def _outcome_value(result: tuple[str | None, str]) -> dict[str, object]:
        return {"source": result[0], "outcome": result[1]}

    def _side_value(
        self, side: str, event: str | None, confirmation_delay_sec: float | None,
        t_sec: float,
        boundary: BoundaryResult,
    ) -> dict[str, object]:
        tracker = self._trackers[side]
        source = _source_from_event(event)
        if event is not None:
            self._source[side] = source
            if event.startswith("candidate_"):
                self._phase[side] = "pending"
            elif event.startswith("released_"):
                self._phase[side] = "released"
            elif event.startswith("confirmed_"):
                self._phase[side] = "confirmed"
        elif tracker.has_pending_candidate():
            self._phase[side] = "pending"
        elif self._phase[side] == "released":
            # released は旧予測を遮断する一観測frameだけのbarrier。
            # 次のdense行からclearへ戻し、同generationの再計算結果を許可する。
            self._phase[side], self._source[side] = "clear", None
        elif (
            not tracker.resolved_is_dead_for_game(self._game_idx)
            and self._phase[side] == "confirmed"
        ):
            self._phase[side], self._source[side] = "clear", None
        elapsed = tracker.pending_elapsed_sec(t_sec)
        outcome_value = boundary["outcomes"].get(side) if boundary["occurred"] else None
        boundary_outcome = (
            outcome_value["outcome"] if isinstance(outcome_value, Mapping) else None
        )
        return {
            "state": self._phase[side], "transition": event,
            "source": self._source[side],
            "pending_elapsed_ms": None if elapsed is None else round(elapsed * 1000),
            "confirmation_delay_ms": (
                None if confirmation_delay_sec is None
                else round(confirmation_delay_sec * 1000)
            ),
            "boundary_outcome": boundary_outcome,
        }

    def _event_value(
        self, frame_idx: int, game_idx: int,
        sides: Mapping[str, object], boundary: BoundaryResult,
    ) -> dict[str, object]:
        available_ms = (
            frame_idx * 1000 * self.timebase["numerator"]
            // self.timebase["denominator"]
        )
        return {
            "frame_idx": frame_idx, "available_frame": frame_idx,
            "available_ms": available_ms, "game_idx": game_idx,
            "death_generation": self._generation,
            "sides": dict(sides), "boundary": boundary,
        }

    def sidecar_value(
        self, processing_start_frame: int,
        processing_end_frame_exclusive: int,
        requested_end_frame_exclusive: int | None = None,
    ) -> dict[str, object]:
        """EOSでpendingを解決せず、finalize済みcanonical Mappingを返す。

        ``completion``/``end_of_stream`` はsidecar観測列を原子的に閉じたこと、
        ``decode_status`` は要求区間を最後までdecodeできたかを表す。後者を
        分離し、途中の ``cap.read`` 失敗を正常完走に見せない。
        """
        requested_end = (
            processing_end_frame_exclusive
            if requested_end_frame_exclusive is None
            else requested_end_frame_exclusive
        )
        self._validate_range(
            processing_start_frame, processing_end_frame_exclusive, requested_end,
        )
        observed = len(self._events)
        pending = {
            side: self._trackers[side].has_pending_candidate() for side in SIDES
        }
        decode_status = (
            DECODE_STATUS_COMPLETE
            if processing_end_frame_exclusive == requested_end
            else DECODE_STATUS_EARLY_END
        )
        return {
            "schema_version": SCHEMA_VERSION, "observer_version": OBSERVER_VERSION,
            "source_video_id": self.source_video_id,
            "source_video_sha256": self.source_video_sha256,
            "processing_start_frame": processing_start_frame,
            "processing_end_frame_exclusive": processing_end_frame_exclusive,
            "requested_end_frame_exclusive": requested_end,
            "decode_status": decode_status,
            "timebase": dict(self.timebase),
            "sample_interval_frames": self.sample_interval_frames,
            "observation_rate_hz": self.observation_rate_hz,
            "stationary_confirm_sec": self.stationary_confirm_sec,
            "completion": "complete", "end_of_stream": True,
            "observed_frame_count": observed, "inspected_side_count": observed * 2,
            "event_count": observed, "death_generation": self._generation,
            "pending_at_end_of_stream": pending,
            "confirmation_delays_ms": [
                round(value * 1000) for value in self._stats.confirm_delays_sec
            ],
            "counters": self._counter_value(), "events": list(self._events),
        }

    def _validate_range(self, start: int, end: int, requested_end: int) -> None:
        if start < 0 or end <= start:
            raise ValueError("processing frame range が不正です")
        if requested_end < end:
            raise ValueError("requested end はprocessing end以後である必要があります")
        if not self._events:
            raise ValueError("死亡sidecarにはdense観測行が必要です")
        if self._events and self._events[0]["frame_idx"] < start:
            raise ValueError("eventがprocessing_start_frameより前です")
        if self._events and self._events[-1]["frame_idx"] >= end:
            raise ValueError("eventがprocessing_end_frame_exclusive以後です")

    def _counter_value(self) -> dict[str, object]:
        stats = self._stats
        return {
            "frames_total": stats.frames_total,
            "candidate_placement": stats.candidate_placement,
            "candidate_ojama": stats.candidate_ojama,
            "released_placement": stats.released_placement,
            "released_ojama": stats.released_ojama,
            "released_survival_placement": stats.released_survival_placement,
            "released_survival_ojama": stats.released_survival_ojama,
            "confirmed_placement": stats.confirmed_placement,
            "confirmed_ojama": stats.confirmed_ojama,
            "threshold_confirmed": stats.threshold_confirmed,
            "released_by_chain": stats.released_by_chain,
            "released_by_next_or_tsumo": stats.released_by_next_or_tsumo,
            "total_boundaries": stats.total_boundaries,
            "pending_at_boundary": stats.pending_at_boundary,
            "boundary_confirmed": stats.boundary_confirmed,
            "boundary_rejected_survival_evidence": stats.boundary_rejected_survival_evidence,
            "ambiguous_both_pending": stats.ambiguous_both_pending,
            "match_evidence_rejected_frame_count": (
                self._match_evidence_rejected_frame_count
            ),
        }
