"""おじゃま会計の単調カウンタを非消費型の観測サイドカーへ記録する。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from src.chain_detector import (
    CHAIN_MECHANISM_BASELINE,
    CHAIN_MECHANISM_FORMULA,
    CHAIN_MECHANISM_FORMULA_READ,
)
from src.chain_id_resolver import (
    FINALIZED_SOURCE_SCORE_OCR_DIFF,
    ChainIdResolver,
    ChainObservation,
    ObservationKind,
    ResolvedChain,
)
from src.exchange_episode_tracker import classify_gross_counter_delta
from src.ojama_accounting import (
    AttackFinalizationCounters,
    GrossOjamaCounters,
    OjamaAccountingTracker,
)


ACCOUNTING_SIDECAR_VERSION = "event-accounting-sidecar/v1"
ACCOUNTING_OBSERVER_VERSION = "event-accounting-observer/v1"
_MECHANISM_KIND = {
    CHAIN_MECHANISM_FORMULA: ObservationKind.FORMULA_STEP,
    CHAIN_MECHANISM_FORMULA_READ: ObservationKind.FORMULA_STEP,
    CHAIN_MECHANISM_BASELINE: ObservationKind.CHAIN_SETTLED,
}


@dataclass(frozen=True, slots=True)
class AttackFinalizationObservation:
    """一つのsideでscore OCR会計が確定した攻撃。"""

    side: str
    side_ordinal: int
    chain_total_score: int
    generated_amount: int
    effective_rate: int
    leftover_before: int
    leftover_after: int
    resolver_chain_ordinal: int = 0


@dataclass(frozen=True, slots=True)
class ProvisionalAttackObservation:
    """掛け算式の成長観測から得た、会計へ未採用の暫定攻撃値。"""

    side: str
    resolver_chain_ordinal: int
    update_ordinal: int
    chain_count: int
    provisional_score: int
    provisional_generated_amount: int
    effective_rate: int
    mechanism: str


@dataclass(frozen=True, slots=True)
class AccountingDeltaObservation:
    """一処理フレームで実際に変化したgross会計。"""

    frame_idx: int
    pending_before_p1: int
    pending_before_p2: int
    pending_after_p1: int
    pending_after_p2: int
    deltas: dict[str, int]
    attacks: tuple[AttackFinalizationObservation, ...]
    provisional_updates: tuple[ProvisionalAttackObservation, ...]
    conservation_residual_p1: int
    conservation_residual_p2: int
    formal_boundary: bool


class EventAccountingRecorder:
    """累積値を読むだけで、会計本体を消費・変更せず差分を記録する。"""

    def __init__(self) -> None:
        self._previous_gross = _zero_gross()
        self._previous_final = _zero_finalization()
        self._previous_pending = (0, 0)
        self._rows: list[AccountingDeltaObservation] = []
        self._observed_frame_count = 0
        self._inspected_side_count = 0
        self._resolver = ChainIdResolver()
        self._last_chain_key: dict[str, tuple[Any, ...] | None] = {
            "p1": None, "p2": None,
        }
        self._last_game_idx: int | None = None
        self._resolved_cursor = 0
        self._provisional_values: dict[int, tuple[int, int]] = {}
        self._provisional_update_counts: dict[int, int] = {}
        self._last_mechanism: dict[str, str] = {"p1": "", "p2": ""}
        self._chain_mechanisms: dict[int, str] = {}

    def observe(
        self, frame_idx: int, t_sec: float, tracker: OjamaAccountingTracker,
        *, game_idx: int = 0,
        chain_events: tuple[tuple[str, object | None], ...] = (),
    ) -> None:
        """現在値を読み、変化があるフレームだけ追記する。"""
        gross = tracker.get_gross_counters(t_sec)
        final = tracker.get_attack_finalization_counters(t_sec)
        snapshot = tracker.get_snapshot(t_sec)
        pending = (snapshot.pending_p1_uncapped, snapshot.pending_p2_uncapped)
        row = self._classify(
            frame_idx, t_sec, game_idx, chain_events, tracker, gross, final, pending,
        )
        if row is not None:
            self._rows.append(row)
        self._previous_gross, self._previous_final = gross, final
        self._previous_pending = pending
        self._observed_frame_count += 1
        self._inspected_side_count += 2

    def _classify(
        self, frame_idx: int, t_sec: float, game_idx: int,
        chain_events: tuple[tuple[str, object | None], ...],
        tracker: OjamaAccountingTracker, gross: GrossOjamaCounters,
        final: AttackFinalizationCounters, pending: tuple[int, int],
    ) -> AccountingDeltaObservation | None:
        classified = classify_gross_counter_delta(
            self._previous_gross, gross, self._previous_pending, pending, 0,
        )
        deltas = _gross_deltas(self._previous_gross, gross)
        attacks = _attack_deltas(self._previous_final, final, classified)
        formal_boundary = self._advance_boundary(game_idx, t_sec)
        self._observe_chain_events(chain_events, t_sec)
        provisionals = self._new_provisionals(tracker, t_sec)
        attacks = self._resolve_attacks(attacks, t_sec)
        residuals = (
            _exact_integer(classified.conservation_residual_p1),
            _exact_integer(classified.conservation_residual_p2),
        )
        if (not any(deltas.values()) and not attacks and not provisionals
                and not formal_boundary
                and residuals == (0, 0)):
            return None
        return AccountingDeltaObservation(
            frame_idx, *self._previous_pending, *pending, deltas, attacks, provisionals,
            residuals[0], residuals[1], formal_boundary,
        )

    def _advance_boundary(self, game_idx: int, t_sec: float) -> bool:
        changed = self._last_game_idx is not None and game_idx != self._last_game_idx
        if changed:
            self._resolver.push(ChainObservation(
                side="BOTH", t_sec=t_sec, kind=ObservationKind.MATCH_BOUNDARY,
            ))
        self._last_game_idx = game_idx
        return changed

    def _observe_chain_events(
        self, chain_events: tuple[tuple[str, object | None], ...], t_sec: float,
    ) -> None:
        for side, event in chain_events:
            key = _chain_event_key(event)
            if event is None or key == self._last_chain_key[side]:
                continue
            self._last_chain_key[side] = key
            mechanism = str(getattr(event, "mechanism", "") or "")
            kind = _MECHANISM_KIND.get(mechanism)
            if kind is None:
                continue
            self._last_mechanism[side] = mechanism
            self._resolver.push(ChainObservation(
                side=side.upper(), t_sec=t_sec, kind=kind,
                chain_count=int(getattr(event, "chain_count")),
                total_score=int(getattr(event, "total_score")),
                mechanism=mechanism,
            ))

    def _new_provisionals(
        self, tracker: OjamaAccountingTracker, t_sec: float,
    ) -> tuple[ProvisionalAttackObservation, ...]:
        result: list[ProvisionalAttackObservation] = []
        for active in self._resolver.active():
            if not active.growth_observed:
                continue
            current = (active.step_count, active.provisional_score)
            if self._provisional_values.get(active.chain_id) == current:
                continue
            self._provisional_values[active.chain_id] = current
            update = self._provisional_update_counts.get(active.chain_id, 0) + 1
            self._provisional_update_counts[active.chain_id] = update
            side = active.side.lower()
            rate = _effective_rate_for_tracker(tracker, t_sec)
            mechanism = self._chain_mechanisms.setdefault(
                active.chain_id, self._last_mechanism[side],
            )
            result.append(ProvisionalAttackObservation(
                side, active.chain_id, update, active.step_count,
                active.provisional_score, active.provisional_score // rate,
                rate, mechanism,
            ))
        return tuple(result)

    def _resolve_attacks(
        self, attacks: tuple[AttackFinalizationObservation, ...], t_sec: float,
    ) -> tuple[AttackFinalizationObservation, ...]:
        for attack in attacks:
            self._resolver.push(ChainObservation(
                side=attack.side.upper(), t_sec=t_sec,
                kind=ObservationKind.SCORE_FINALIZE,
                total_score=attack.chain_total_score,
                mechanism="score_ocr_generated_delta",
            ))
        resolved = self._resolver.resolved()
        new = resolved[self._resolved_cursor:]
        self._resolved_cursor = len(resolved)
        finalized = [item for item in new if item.was_finalized]
        return _attach_resolved_chain_ordinals(attacks, finalized)

    def sidecar_value(
        self, processing_start_frame: int, processing_end_frame_exclusive: int,
    ) -> dict[str, Any]:
        """収集器が新規ファイルへ書くJSON互換値を返す。"""
        _validate_processing_range(processing_start_frame, processing_end_frame_exclusive)
        return {
            "schema_version": ACCOUNTING_SIDECAR_VERSION,
            "observer_version": ACCOUNTING_OBSERVER_VERSION,
            "processing_start_frame": processing_start_frame,
            "processing_end_frame_exclusive": processing_end_frame_exclusive,
            "observed_frame_count": self._observed_frame_count,
            "inspected_side_count": self._inspected_side_count,
            "nonzero_row_count": len(self._rows),
            "initial_gross_counters": _counter_values(_zero_gross()),
            "final_gross_counters": _counter_values(self._previous_gross),
            "final_pending_uncapped": {
                "p1": self._previous_pending[0], "p2": self._previous_pending[1],
            },
            "rows": [_row_value(row) for row in self._rows],
        }


def _zero_gross() -> GrossOjamaCounters:
    return GrossOjamaCounters(0.0, *(0 for _ in range(12)))


def _zero_finalization() -> AttackFinalizationCounters:
    return AttackFinalizationCounters(0.0, *(0 for _ in range(12)))


def _chain_event_key(event: object | None) -> tuple[Any, ...] | None:
    if event is None:
        return None
    return (
        round(float(getattr(event, "trigger_sec")), 3),
        getattr(event, "mechanism", None),
        int(getattr(event, "chain_count")),
        int(getattr(event, "total_score")),
    )


def _effective_rate_for_tracker(
    tracker: OjamaAccountingTracker, t_sec: float,
) -> int:
    rate = tracker.get_effective_rate(t_sec)
    if rate <= 0:
        raise ValueError("おじゃま換算率が0以下です")
    return rate


def _attach_resolved_chain_ordinals(
    attacks: tuple[AttackFinalizationObservation, ...],
    resolved: list[ResolvedChain],
) -> tuple[AttackFinalizationObservation, ...]:
    authoritative = [
        item for item in resolved
        if item.finalized_source == FINALIZED_SOURCE_SCORE_OCR_DIFF
    ]
    result: list[AttackFinalizationObservation] = []
    for attack in attacks:
        matches = [item for item in authoritative if item.side.lower() == attack.side]
        if len(matches) != 1:
            raise ValueError("確定攻撃を一つの物理連鎖IDへ対応付けられません")
        match = matches[0]
        authoritative.remove(match)
        result.append(replace(attack, resolver_chain_ordinal=match.chain_id))
    if authoritative:
        raise ValueError("score OCR確定連鎖に対応する攻撃会計がありません")
    return tuple(result)


def _counter_values(counters: GrossOjamaCounters) -> dict[str, int]:
    values = asdict(counters)
    values.pop("t_sec")
    return {str(key): int(value) for key, value in values.items()}


def _gross_deltas(
    previous: GrossOjamaCounters, current: GrossOjamaCounters,
) -> dict[str, int]:
    before, after = _counter_values(previous), _counter_values(current)
    deltas = {key: after[key] - before[key] for key in before}
    if any(value < 0 for value in deltas.values()):
        raise ValueError("gross累積カウンタが減少しました")
    return deltas


def _attack_deltas(
    previous: AttackFinalizationCounters, current: AttackFinalizationCounters,
    classified: Any,
) -> tuple[AttackFinalizationObservation, ...]:
    attacks: list[AttackFinalizationObservation] = []
    for side, expected in (("p1", classified.generated_by_1p),
                           ("p2", classified.generated_by_2p)):
        item = _attack_delta(previous, current, side, int(expected))
        if item is not None:
            attacks.append(item)
    return tuple(attacks)


def _attack_delta(
    previous: AttackFinalizationCounters, current: AttackFinalizationCounters,
    side: str, expected_generated: int,
) -> AttackFinalizationObservation | None:
    before = int(getattr(previous, f"finalized_count_{side}"))
    after = int(getattr(current, f"finalized_count_{side}"))
    if after - before not in {0, 1}:
        raise ValueError("一処理フレームで同じ側の攻撃が複数確定しました")
    if after == before:
        if expected_generated:
            raise ValueError("生成量増分に対応する確定攻撃がありません")
        return None
    generated = int(getattr(current, f"generated_{side}"))
    if generated != expected_generated:
        raise ValueError("確定攻撃量とgross生成量増分が一致しません")
    return AttackFinalizationObservation(
        side, after, int(getattr(current, f"chain_total_score_{side}")),
        generated, int(getattr(current, f"effective_rate_{side}")),
        int(getattr(current, f"leftover_before_{side}")),
        int(getattr(current, f"leftover_after_{side}")),
    )


def _exact_integer(value: float) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise ValueError("保存則残差が整数ではありません")
    return int(rounded)


def _row_value(row: AccountingDeltaObservation) -> dict[str, Any]:
    return {
        "frame_idx": row.frame_idx,
        "pending_before": {"p1": row.pending_before_p1, "p2": row.pending_before_p2},
        "pending_after": {"p1": row.pending_after_p1, "p2": row.pending_after_p2},
        "deltas": dict(row.deltas),
        "attack_finalizations": [asdict(item) for item in row.attacks],
        "provisional_updates": [asdict(item) for item in row.provisional_updates],
        "conservation_residual": {
            "p1": row.conservation_residual_p1, "p2": row.conservation_residual_p2,
        },
        "formal_boundary": row.formal_boundary,
    }


def _validate_processing_range(start: int, end: int) -> None:
    if start < 0 or end <= start:
        raise ValueError("処理フレーム範囲が不正です")


__all__ = [
    "ACCOUNTING_OBSERVER_VERSION",
    "ACCOUNTING_SIDECAR_VERSION",
    "AccountingDeltaObservation",
    "AttackFinalizationObservation",
    "EventAccountingRecorder",
    "ProvisionalAttackObservation",
]
