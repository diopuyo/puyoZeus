"""連鎖開始・全消し・おじゃま着地盤面を非消費型サイドカーへ記録する。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.chain_detector import (
    CHAIN_MECHANISM_BASELINE,
    CHAIN_MECHANISM_FORMULA,
    CHAIN_MECHANISM_FORMULA_READ,
)
from src.chain_id_resolver import (
    ChainIdResolver,
    ChainObservation,
    ObservationKind,
)


PHYSICAL_SIDECAR_VERSION = "event-physical-sidecar/v1"
PHYSICAL_OBSERVER_VERSION = "event-physical-observer/v1"
BOARD_ROWS = 13
BOARD_COLUMNS = 6
ALLOWED_CELL_VALUES = frozenset({0, 1, 2, 3, 4, 5, 9, 10})
STABLE_STATE = "stable"
OJAMA_FALL_STATE = "ojama_fall"
TSUMO_FALL_STATE = "tsumo_fall"
SIDE_VALUES = ("p1", "p2")
_MECHANISM_KIND = {
    CHAIN_MECHANISM_FORMULA: ObservationKind.FORMULA_STEP,
    CHAIN_MECHANISM_FORMULA_READ: ObservationKind.FORMULA_STEP,
    CHAIN_MECHANISM_BASELINE: ObservationKind.CHAIN_SETTLED,
}


Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class StableBoardEvidence:
    """一つのsideで最後に直接観測できた安定盤面。"""

    frame_idx: int
    t_sec: float
    grid: Grid
    raw_grid: Grid | None = None


@dataclass(frozen=True, slots=True)
class LandingCandidate:
    """おじゃま落下状態へ入った時点の盤面と時刻。"""

    start_frame: int
    start_sec: float
    game_idx: int
    before: StableBoardEvidence | None
    before_is_immediate_previous_observation: bool
    opponent_chain_active_at_start: bool


@dataclass(frozen=True, slots=True)
class PhysicalObservationRow:
    """一つの物理信号または盤面比較。"""

    frame_idx: int
    t_sec: float
    game_idx: int
    side: str
    observation_type: str
    payload: dict[str, Any]


class EventPhysicalRecorder:
    """認識結果を読むだけで、既存の状態機械を変更せず物理信号を記録する。"""

    def __init__(self) -> None:
        self._rows: list[PhysicalObservationRow] = []
        self._last_chain_key: dict[str, tuple[Any, ...] | None] = {
            side: None for side in SIDE_VALUES
        }
        self._boundary_chain_trigger: dict[str, float | None] = {
            side: None for side in SIDE_VALUES
        }
        self._last_state: dict[str, str | None] = {side: None for side in SIDE_VALUES}
        self._has_ever_placed: dict[str, bool] = {side: False for side in SIDE_VALUES}
        self._last_stable: dict[str, StableBoardEvidence | None] = {
            side: None for side in SIDE_VALUES
        }
        self._landing: dict[str, LandingCandidate | None] = {
            side: None for side in SIDE_VALUES
        }
        self._last_game_idx: int | None = None
        self._observed_frame_count = 0
        self._inspected_side_count = 0
        self._last_observed_frame: int | None = None
        self._last_observed_sec: float | None = None
        self._resolver = ChainIdResolver()
        self._physical_chain_ids: dict[tuple[int, str, float], int] = {}
        self._resolver_chain_aliases: dict[int, int] = {}
        self._emitted_chain_ids: set[int] = set()
        self._all_clear_gained_ids: set[int] = set()
        self._all_clear_consumed_ids: set[int] = set()
        self._counters = _zero_counters()

    def observe(
        self, frame_idx: int, t_sec: float, game_idx: int,
        sides: Sequence[tuple[str, object, object | None, object | None]],
        *, raw_boards: Mapping[str, object | None] | None = None,
        match_evidence: bool = True,
    ) -> None:
        """現在フレームのside別状態・盤面・連鎖イベントを観測する。"""
        _validate_position(frame_idx, t_sec, game_idx)
        boundary = self._last_game_idx is not None and game_idx != self._last_game_idx
        if boundary:
            self._remember_boundary_chain_residuals(sides)
            self._close_boundary_landings(frame_idx, t_sec, game_idx)
        chain_active = {
            side: _chain_event_in_progress(chain_event, t_sec)
            for side, _state, _board, chain_event in sides
        }
        seen: set[str] = set()
        for side, state, board, chain_event in sides:
            _validate_side(side, seen)
            raw_board = None if raw_boards is None else raw_boards.get(side)
            self._observe_side(
                frame_idx, t_sec, game_idx, side, state, board, raw_board,
                chain_event, boundary, match_evidence,
                chain_active.get(_other_side(side), False),
            )
            seen.add(side)
        self._last_game_idx = game_idx
        self._observed_frame_count += 1
        self._inspected_side_count += len(seen)
        self._last_observed_frame = frame_idx
        self._last_observed_sec = t_sec

    def _observe_side(
        self, frame_idx: int, t_sec: float, game_idx: int, side: str,
        state: object, board: object | None, raw_board: object | None,
        chain_event: object | None, suppress_chain: bool, match_evidence: bool,
        opponent_chain_active: bool,
    ) -> None:
        state_value = _state_value(state)
        if match_evidence and self._is_first_placement(side, state_value):
            self._has_ever_placed[side] = True
        if not self._has_ever_placed[side]:
            self._count_preplacement(
                side, state_value, chain_event, match_evidence,
            )
            self._last_state[side] = state_value
            return
        self._counters["eligible_side_frame_count"] += 1
        # 非安定中の盤面は物理演出を含み、比較証拠には使わない。
        # 直列化もしないことで、観測追加が不要な失敗源になるのを避ける。
        grid = _grid_value(board) if state_value == STABLE_STATE else None
        raw_grid = _grid_value(raw_board) if state_value == STABLE_STATE else None
        self._observe_chain(
            frame_idx, t_sec, game_idx, side, chain_event, suppress_chain,
        )
        self._observe_landing(
            frame_idx, t_sec, game_idx, side, state_value, grid, raw_grid,
            opponent_chain_active,
        )
        if state_value == STABLE_STATE and grid is not None:
            self._last_stable[side] = StableBoardEvidence(
                frame_idx, t_sec, grid, raw_grid,
            )
        self._last_state[side] = state_value

    def _is_first_placement(self, side: str, state: str) -> bool:
        return state == STABLE_STATE and self._last_state[side] == TSUMO_FALL_STATE

    def _count_preplacement(
        self, side: str, state: str, chain_event: object | None,
        match_evidence: bool,
    ) -> None:
        self._counters["preplacement_side_frame_count"] += 1
        if not match_evidence:
            self._counters["match_evidence_rejected_side_frame_count"] += 1
        if chain_event is not None:
            self._counters["preplacement_chain_event_present_frame_count"] += 1
            self._clear_changed_boundary_trigger(side, chain_event)
        else:
            self._last_chain_key[side] = None
            self._boundary_chain_trigger[side] = None
        if state == OJAMA_FALL_STATE:
            self._counters["preplacement_ojama_fall_state_frame_count"] += 1

    def _observe_chain(
        self, frame_idx: int, t_sec: float, game_idx: int,
        side: str, event: object | None, suppress: bool = False,
    ) -> None:
        if event is None:
            self._last_chain_key[side] = None
            self._boundary_chain_trigger[side] = None
            return
        self._counters["chain_event_present_frame_count"] += 1
        key = _chain_event_key(event)
        if self._is_boundary_chain_residual(side, event, suppress):
            self._count_boundary_chain_residual(side, key)
            return
        if key == self._last_chain_key[side]:
            self._counters["chain_repeated_frame_count"] += 1
            return
        self._last_chain_key[side] = key
        self._counters["chain_candidate_update_count"] += 1
        mechanism = str(getattr(event, "mechanism", "") or "")
        kind = _MECHANISM_KIND.get(mechanism)
        if kind is None:
            self._counters["chain_ignored_mechanism_update_count"] += 1
            return
        self._resolver.push(_chain_observation(side, t_sec, event, kind, mechanism))
        active = [item for item in self._resolver.active() if item.side.lower() == side]
        if len(active) != 1:
            self._counters["chain_update_suppressed_count"] += 1
            return
        chain_id = self._canonical_chain_id(
            game_idx, side, event, active[0].chain_id,
        )
        self._emit_chain_start(frame_idx, t_sec, game_idx, side, event, chain_id)
        self._emit_all_clear_signals(frame_idx, t_sec, game_idx, side, event, chain_id)

    def _is_boundary_chain_residual(
        self, side: str, event: object, suppress: bool,
    ) -> bool:
        trigger = _chain_trigger(event)
        residual = self._boundary_chain_trigger[side]
        if suppress:
            self._boundary_chain_trigger[side] = trigger
            return True
        if residual is None or trigger != residual:
            self._boundary_chain_trigger[side] = None
            return False
        return True

    def _count_boundary_chain_residual(
        self, side: str, key: tuple[Any, ...],
    ) -> None:
        if key == self._last_chain_key[side]:
            self._counters["chain_repeated_frame_count"] += 1
            return
        self._last_chain_key[side] = key
        self._counters["chain_candidate_update_count"] += 1
        self._counters["chain_update_suppressed_count"] += 1
        self._counters["boundary_chain_residual_suppressed_count"] += 1

    def _canonical_chain_id(
        self, game_idx: int, side: str, event: object, resolver_id: int,
    ) -> int:
        aliased = self._resolver_chain_aliases.get(resolver_id)
        if aliased is not None:
            return aliased
        physical_key = (game_idx, side, _chain_trigger(event))
        canonical = self._physical_chain_ids.get(physical_key)
        if canonical is None:
            canonical = resolver_id
            self._physical_chain_ids[physical_key] = canonical
        else:
            self._counters["chain_reidentified_update_count"] += 1
        self._resolver_chain_aliases[resolver_id] = canonical
        return canonical

    def _emit_chain_start(
        self, frame_idx: int, t_sec: float, game_idx: int, side: str,
        event: object, chain_id: int,
    ) -> None:
        if chain_id in self._emitted_chain_ids:
            self._counters["chain_update_suppressed_count"] += 1
            return
        self._emitted_chain_ids.add(chain_id)
        self._counters["chain_started_count"] += 1
        payload = _chain_payload(event)
        payload["resolver_chain_ordinal"] = chain_id
        self._rows.append(PhysicalObservationRow(
            frame_idx, t_sec, game_idx, side, "chain_detected", payload,
        ))

    def _emit_all_clear_signals(
        self, frame_idx: int, t_sec: float, game_idx: int, side: str,
        event: object, chain_id: int,
    ) -> None:
        bonus = int(getattr(event, "all_clear_bonus_applied"))
        if bonus > 0 and chain_id not in self._all_clear_consumed_ids:
            self._all_clear_consumed_ids.add(chain_id)
            self._append_all_clear(
                frame_idx, t_sec, game_idx, side, event, chain_id,
                "all_clear_consumed_candidate", bonus,
            )
        if bool(getattr(event, "is_all_clear")) and chain_id not in self._all_clear_gained_ids:
            self._all_clear_gained_ids.add(chain_id)
            self._append_all_clear(
                frame_idx, t_sec, game_idx, side, event, chain_id,
                "all_clear_gained_candidate", 0,
            )

    def _append_all_clear(
        self, frame_idx: int, t_sec: float, game_idx: int, side: str,
        event: object, chain_id: int, observation_type: str, bonus: int,
    ) -> None:
        counter = observation_type.replace("_candidate", "_count")
        self._counters[counter] += 1
        self._rows.append(PhysicalObservationRow(
            frame_idx, t_sec, game_idx, side, observation_type,
            _all_clear_payload(event, chain_id, bonus),
        ))

    def _observe_landing(
        self, frame_idx: int, t_sec: float, game_idx: int,
        side: str, state: str, grid: Grid | None, raw_grid: Grid | None,
        opponent_chain_active: bool,
    ) -> None:
        previous = self._last_state[side]
        if state == OJAMA_FALL_STATE and previous != OJAMA_FALL_STATE:
            self._counters["ojama_fall_entry_count"] += 1
            self._replace_open_landing(
                frame_idx, t_sec, game_idx, side, opponent_chain_active,
            )
        candidate = self._landing[side]
        if candidate is None or state != STABLE_STATE or grid is None:
            return
        self._rows.append(_landing_row(
            frame_idx, t_sec, game_idx, side, candidate, grid, raw_grid,
        ))
        self._counters["landing_completed_count"] += 1
        self._landing[side] = None

    def _replace_open_landing(
        self, frame_idx: int, t_sec: float, game_idx: int, side: str,
        opponent_chain_active: bool,
    ) -> None:
        current = self._landing[side]
        if current is not None:
            self._rows.append(_incomplete_landing_row(
                frame_idx, t_sec, current.game_idx, side, current,
                "new_fall_before_stable",
            ))
        before = self._last_stable[side]
        immediate = before is not None and before.frame_idx == self._last_observed_frame
        self._landing[side] = LandingCandidate(
            frame_idx, t_sec, game_idx, before, immediate,
            opponent_chain_active,
        )

    def _close_boundary_landings(
        self, frame_idx: int, t_sec: float, game_idx: int,
    ) -> None:
        close_frame = frame_idx if self._last_observed_frame is None else self._last_observed_frame
        close_sec = t_sec if self._last_observed_sec is None else self._last_observed_sec
        for side in SIDE_VALUES:
            candidate = self._landing[side]
            if candidate is not None:
                self._rows.append(_incomplete_landing_row(
                    close_frame, close_sec, candidate.game_idx, side, candidate,
                    "game_boundary_before_stable",
                ))
            self._landing[side] = None
            self._last_stable[side] = None
            self._last_state[side] = None
            self._has_ever_placed[side] = False
        self._resolver.push(ChainObservation(
            side="BOTH", t_sec=t_sec, kind=ObservationKind.MATCH_BOUNDARY,
        ))

    def _remember_boundary_chain_residuals(
        self, sides: Sequence[tuple[str, object, object | None, object | None]],
    ) -> None:
        current = {
            side: event for side, _state, _board, event in sides
            if side in SIDE_VALUES
        }
        for side in SIDE_VALUES:
            event = current.get(side)
            previous = self._last_chain_key[side]
            self._boundary_chain_trigger[side] = (
                _chain_trigger(event) if event is not None
                else None if previous is None else float(previous[0])
            )
            self._last_chain_key[side] = None

    def _clear_changed_boundary_trigger(self, side: str, event: object) -> None:
        residual = self._boundary_chain_trigger[side]
        if residual is not None and _chain_trigger(event) != residual:
            self._boundary_chain_trigger[side] = None

    def sidecar_value(
        self, processing_start_frame: int, processing_end_frame_exclusive: int,
    ) -> dict[str, Any]:
        """収集器が新規ファイルへ書くJSON互換値を返す。"""
        _validate_processing_range(processing_start_frame, processing_end_frame_exclusive)
        rows = [*self._rows, *self._open_landing_rows()]
        counters = _completed_counters(
            self._counters, rows, self._inspected_side_count,
        )
        return {
            "schema_version": PHYSICAL_SIDECAR_VERSION,
            "observer_version": PHYSICAL_OBSERVER_VERSION,
            "processing_start_frame": processing_start_frame,
            "processing_end_frame_exclusive": processing_end_frame_exclusive,
            "observed_frame_count": self._observed_frame_count,
            "inspected_side_count": self._inspected_side_count,
            "row_count": len(rows),
            "open_landing_count": sum(value is not None for value in self._landing.values()),
            "counters": counters,
            "rows": [_row_value(row) for row in rows],
        }

    def _open_landing_rows(self) -> list[PhysicalObservationRow]:
        if self._last_observed_frame is None or self._last_observed_sec is None:
            return []
        return [
            _incomplete_landing_row(
                self._last_observed_frame, self._last_observed_sec, candidate.game_idx,
                side, candidate, "processing_end_before_stable",
            )
            for side, candidate in self._landing.items()
            if candidate is not None
        ]


def _chain_event_key(event: object | None) -> tuple[Any, ...] | None:
    if event is None:
        return None
    return (
        round(float(getattr(event, "trigger_sec")), 6),
        round(float(getattr(event, "end_sec")), 6),
        int(getattr(event, "chain_count")),
        int(getattr(event, "total_score")),
        str(getattr(event, "mechanism", "") or ""),
    )


def _chain_trigger(event: object) -> float:
    return round(float(getattr(event, "trigger_sec")), 6)


def _chain_payload(event: object) -> dict[str, Any]:
    return {
        "trigger_sec": float(getattr(event, "trigger_sec")),
        "projected_end_sec": float(getattr(event, "end_sec")),
        "chain_count": int(getattr(event, "chain_count")),
        "total_erased": int(getattr(event, "total_erased")),
        "total_score": int(getattr(event, "total_score")),
        "base_score": int(getattr(event, "base_score")),
        "all_clear_bonus_applied": int(getattr(event, "all_clear_bonus_applied")),
        "is_all_clear": bool(getattr(event, "is_all_clear")),
        "mechanism": str(getattr(event, "mechanism", "") or "unknown"),
        "score_estimated": bool(getattr(event, "score_estimated", False)),
        "observation_state": "physics_reconstructed_not_independently_observed",
    }


def _chain_observation(
    side: str, t_sec: float, event: object,
    kind: ObservationKind, mechanism: str,
) -> ChainObservation:
    return ChainObservation(
        side=side.upper(), t_sec=t_sec, kind=kind,
        chain_count=int(getattr(event, "chain_count")),
        total_score=int(getattr(event, "total_score")),
        mechanism=mechanism,
    )


def _all_clear_payload(event: object, chain_id: int, bonus: int) -> dict[str, Any]:
    payload = {
        "resolver_chain_ordinal": chain_id,
        "chain_count": int(getattr(event, "chain_count")),
        "mechanism": str(getattr(event, "mechanism", "") or "unknown"),
        "observation_state": "physics_reconstructed_not_independently_observed",
    }
    if bonus > 0:
        payload["all_clear_bonus_amount"] = bonus
    return payload


def _landing_row(
    frame_idx: int, t_sec: float, game_idx: int, side: str,
    candidate: LandingCandidate, after: Grid, raw_after: Grid | None,
) -> PhysicalObservationRow:
    before = candidate.before
    if before is None:
        payload = _landing_base_payload(candidate, None, after)
        payload.update({"comparison_state": "missing_before_board"})
    else:
        payload = _landing_base_payload(candidate, before, after)
        payload.update(_landing_differences(before.grid, after))
        payload.update(_raw_landing_evidence(before, after, raw_after))
        if not candidate.before_is_immediate_previous_observation:
            payload["comparison_state"] = "stale_before_board"
        elif payload.get("confirmed_raw_agreement") is False:
            payload["comparison_state"] = "unexpected_board_change"
    return PhysicalObservationRow(
        frame_idx, t_sec, candidate.game_idx, side, "landing_board_compared", payload,
    )


def _landing_base_payload(
    candidate: LandingCandidate, before: StableBoardEvidence | None, after: Grid,
) -> dict[str, Any]:
    return {
        "fall_start_frame": candidate.start_frame,
        "fall_start_sec": candidate.start_sec,
        "before_frame": None if before is None else before.frame_idx,
        "before_sec": None if before is None else before.t_sec,
        "before_grid": None if before is None else _grid_lists(before.grid),
        "after_grid": _grid_lists(after),
        "before_is_immediate_previous_observation": (
            candidate.before_is_immediate_previous_observation
        ),
        "opponent_chain_active_at_fall_start": (
            candidate.opponent_chain_active_at_start
        ),
    }


def _landing_differences(before: Grid, after: Grid) -> dict[str, Any]:
    before_counts, after_counts = _board_counts(before), _board_counts(after)
    differences = {
        key: after_counts[key] - before_counts[key] for key in before_counts
    }
    expected = (
        differences["color"] == 0 and differences["unknown"] == 0
        and differences["garbage"] > 0
        and differences["occupied"] == differences["garbage"]
    )
    return {
        "before_counts": before_counts,
        "after_counts": after_counts,
        "differences": differences,
        "comparison_state": _comparison_state(differences, expected),
    }


def _raw_landing_evidence(
    before: StableBoardEvidence, after: Grid, raw_after: Grid | None,
) -> dict[str, Any]:
    if before.raw_grid is None or raw_after is None:
        return {"raw_observation_state": "not_available"}
    raw = _landing_differences(before.raw_grid, raw_after)
    return {
        "raw_before_grid": _grid_lists(before.raw_grid),
        "raw_after_grid": _grid_lists(raw_after),
        "raw_differences": raw["differences"],
        "raw_comparison_state": raw["comparison_state"],
        "confirmed_raw_agreement": before.grid == before.raw_grid and after == raw_after,
        "raw_observation_state": "direct_cnn_stable_frame",
    }


def _comparison_state(differences: Mapping[str, int], expected: bool) -> str:
    if expected:
        return "consistent_garbage_addition"
    if all(value == 0 for value in differences.values()):
        return "zero_garbage_change"
    return "unexpected_board_change"


def _incomplete_landing_row(
    frame_idx: int, t_sec: float, game_idx: int, side: str,
    candidate: LandingCandidate, reason: str,
) -> PhysicalObservationRow:
    return PhysicalObservationRow(
        frame_idx, t_sec, game_idx, side, "landing_incomplete",
        {
            "fall_start_frame": candidate.start_frame,
            "fall_start_sec": candidate.start_sec,
            "before_frame": None if candidate.before is None else candidate.before.frame_idx,
            "reason": reason,
        },
    )


def _board_counts(grid: Grid) -> dict[str, int]:
    flat = [value for row in grid for value in row]
    return {
        "color": sum(value in {1, 2, 3, 4, 5} for value in flat),
        "garbage": sum(value == 9 for value in flat),
        "unknown": sum(value == 10 for value in flat),
        "occupied": sum(value != 0 for value in flat),
    }


def _grid_value(board: object | None) -> Grid | None:
    if board is None:
        return None
    value = board.to_dict()["grid"] if hasattr(board, "to_dict") else board
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("盤面は13行6列でなければなりません")
    return _validated_grid(value)


def _validated_grid(value: Sequence[object]) -> Grid:
    if len(value) != BOARD_ROWS:
        raise ValueError("盤面は13行でなければなりません")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("盤面行は配列でなければなりません")
        values = tuple(_cell_value(cell) for cell in row)
        if len(values) != BOARD_COLUMNS:
            raise ValueError("盤面は6列でなければなりません")
        rows.append(values)
    return tuple(rows)


def _cell_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("盤面セルは整数でなければなりません")
    if value not in ALLOWED_CELL_VALUES:
        raise ValueError("盤面セルが許可範囲外です")
    return value


def _state_value(value: object) -> str:
    state = getattr(value, "value", value)
    if not isinstance(state, str) or not state:
        raise ValueError("盤面状態は空でない文字列でなければなりません")
    return state


def _other_side(side: str) -> str:
    """相手側のラベルを返す。side検証は直後の既存経路で行う。"""
    return "p2" if side == "p1" else "p1"


def _chain_event_in_progress(event: object | None, t_sec: float) -> bool:
    """残留イベントではなく、物理終了予測前の連鎖だけを進行中とする。"""
    if event is None:
        return False
    end_sec = getattr(event, "end_sec", None)
    if isinstance(end_sec, bool) or not isinstance(end_sec, (int, float)):
        return False
    return float(end_sec) >= t_sec


def _validate_side(side: str, seen: set[str]) -> None:
    if side not in SIDE_VALUES:
        raise ValueError("sideはp1またはp2でなければなりません")
    if side in seen:
        raise ValueError("同じフレームのsideが重複しています")


def _validate_position(frame_idx: int, t_sec: float, game_idx: int) -> None:
    if frame_idx < 0 or t_sec < 0.0 or game_idx < 0:
        raise ValueError("フレーム・時刻・試合番号は0以上でなければなりません")


def _validate_processing_range(start: int, end: int) -> None:
    if start < 0 or end <= start:
        raise ValueError("処理フレーム範囲が不正です")


def _zero_counters() -> dict[str, int]:
    return {
        "preplacement_side_frame_count": 0,
        "eligible_side_frame_count": 0,
        "preplacement_chain_event_present_frame_count": 0,
        "preplacement_ojama_fall_state_frame_count": 0,
        "chain_event_present_frame_count": 0,
        "chain_repeated_frame_count": 0,
        "chain_candidate_update_count": 0,
        "chain_started_count": 0,
        "chain_update_suppressed_count": 0,
        "chain_reidentified_update_count": 0,
        "boundary_chain_residual_suppressed_count": 0,
        "match_evidence_rejected_side_frame_count": 0,
        "chain_ignored_mechanism_update_count": 0,
        "all_clear_gained_count": 0,
        "all_clear_consumed_count": 0,
        "ojama_fall_entry_count": 0,
        "landing_completed_count": 0,
    }


def _completed_counters(
    counters: Mapping[str, int], rows: Sequence[PhysicalObservationRow], inspected: int,
) -> dict[str, int]:
    result = dict(counters)
    for reason in (
        "new_fall_before_stable", "game_boundary_before_stable",
        "processing_end_before_stable",
    ):
        result[f"landing_incomplete_{reason}_count"] = _row_payload_count(
            rows, "landing_incomplete", "reason", reason,
        )
    for state in (
        "consistent_garbage_addition", "zero_garbage_change",
        "unexpected_board_change", "stale_before_board", "missing_before_board",
    ):
        result[f"landing_{state}_count"] = _row_payload_count(
            rows, "landing_board_compared", "comparison_state", state,
        )
    incomplete = sum(
        value for key, value in result.items()
        if key.startswith("landing_incomplete_")
    )
    result["landing_incomplete_count"] = incomplete
    _validate_counter_equations(result, inspected)
    return result


def _row_payload_count(
    rows: Sequence[PhysicalObservationRow], observation_type: str,
    key: str, value: str,
) -> int:
    return sum(
        row.observation_type == observation_type and row.payload.get(key) == value
        for row in rows
    )


def _validate_counter_equations(counters: Mapping[str, int], inspected: int) -> None:
    chain_parts = (
        counters["chain_repeated_frame_count"]
        + counters["chain_candidate_update_count"]
    )
    update_parts = (
        counters["chain_started_count"]
        + counters["chain_update_suppressed_count"]
        + counters["chain_ignored_mechanism_update_count"]
    )
    landing_parts = counters["landing_completed_count"] + counters["landing_incomplete_count"]
    if counters["chain_event_present_frame_count"] != chain_parts:
        raise ValueError("連鎖観測母数の内訳が一致しません")
    if counters["chain_candidate_update_count"] != update_parts:
        raise ValueError("連鎖候補更新の内訳が一致しません")
    diagnosed = (
        counters["chain_reidentified_update_count"]
        + counters["boundary_chain_residual_suppressed_count"]
    )
    if diagnosed > counters["chain_update_suppressed_count"]:
        raise ValueError("連鎖抑制の診断内訳が抑制総数を超えています")
    if counters["ojama_fall_entry_count"] != landing_parts:
        raise ValueError("おじゃま落下入口の内訳が一致しません")
    eligibility = (
        counters["preplacement_side_frame_count"]
        + counters["eligible_side_frame_count"]
    )
    if eligibility != inspected:
        raise ValueError("物理観測の利用可否母数が検査side数と一致しません")


def _grid_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def _row_value(row: PhysicalObservationRow) -> dict[str, Any]:
    value = asdict(row)
    value["payload"] = dict(row.payload)
    return value


__all__ = [
    "EventPhysicalRecorder",
    "PHYSICAL_OBSERVER_VERSION",
    "PHYSICAL_SIDECAR_VERSION",
    "PhysicalObservationRow",
]
