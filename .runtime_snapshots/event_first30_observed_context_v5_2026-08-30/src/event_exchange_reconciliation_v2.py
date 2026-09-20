"""途中火力と実着地を結ぶ、既定OFFの交換会計v2。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.event_accounting_adapter_v1 import (
    AccountingRow,
    AttackFinalization,
    EventAccountingSidecar,
    ProvisionalAttack,
)
from src.event_physical_adapter_v1 import EventPhysicalSidecar
from src.event_physical_observer_v1 import PhysicalObservationRow
from src.exchange_ledger import (
    FINALIZE_SOURCE_SCORE_OCR_DIFF,
    OJAMA_MAX_DROP_PER_TURN,
    EventKind,
    ExchangeEvent,
    ExchangeLedger,
    PhysicalContext,
    Side,
)


DEFAULT_VIDEO_FPS = 30.0
BOARD_CELL_COUNT = 78
BOARD_WIDTH = 6
SIDE_BY_LABEL = {"p1": Side.P1, "p2": Side.P2}
ACCEPTED_LANDING_STATUSES = frozenset({
    "exact_match", "capacity_limited_match", "partial_visible_match",
})
TIMELINE_PRIORITY = {
    "boundary": 0,
    "provisional": 1,
    "finalize": 2,
    "physical": 3,
}


@dataclass(frozen=True, slots=True)
class ReconciledLandingV2:
    """一回の実着地と、その直前に利用可能だった交換量の照合。"""

    frame_idx: int
    game_idx: int
    recipient: str
    visible_amount: int
    expected_drop_amount: int
    gross_landed_amount: int
    mutual_canceled_amount: int
    sender_outstanding_before: int
    recipient_outstanding_before: int
    evidence: str
    status: str


@dataclass(frozen=True, slots=True)
class ReconciledTimelineRowV2:
    """学習・人手レビューに使える、交換会計の因果的な一行。"""

    frame_idx: int
    game_idx: int
    event_type: str
    side: str | None
    chain_id: int | None
    observed_amount: int
    generated_delta: int
    mutual_canceled_amount: int
    gross_landed_amount: int
    visible_landed_amount: int
    p1_outstanding_after: int
    p2_outstanding_after: int
    net_after: int
    observation_trusted: bool
    trusted: bool
    evidence: str
    status: str


@dataclass(frozen=True, slots=True)
class ExchangeReconciliationReportV2:
    """旧手数drainを使わず、途中火力と実着地だけで再構成した結果。"""

    provisional_update_count: int
    finalization_count: int
    left_context_unverified: bool
    physical_landing_row_count: int
    exact_physical_candidate_count: int
    occupancy_corroborated_candidate_count: int
    corroborated_zero_count: int
    deferred_during_active_attack_count: int
    exact_match_count: int
    capacity_limited_match_count: int
    partial_visible_match_count: int
    conflict_count: int
    ignored_nonexact_physical_count: int
    game_index_mismatch_count: int
    late_confirmation_count: int
    late_correction_count: int
    held_finalization_count: int
    late_held_finalization_count: int
    uncertainty_started_count: int
    uncertainty_resolved_by_growth_count: int
    untrusted_observation_row_count: int
    uncertain_timeline_row_count: int
    first_uncertain_frame: int | None
    final_state_trusted: bool
    legacy_modeled_canceled_amount_ignored: int
    legacy_modeled_dropped_amount_ignored: int
    total_generated: int
    total_finalized_generated: int
    total_provisional_only_generated: int
    total_canceled: int
    total_gross_landed: int
    total_visible_landed: int
    total_boundary_expired: int
    total_outstanding: int
    conservation_residual: int
    oversettled_total: int
    net_raw: int
    landings: tuple[ReconciledLandingV2, ...]
    timeline_rows: tuple[ReconciledTimelineRowV2, ...]

    def to_json_value(self) -> dict[str, Any]:
        """監査成果物へ保存できるJSON互換値を返す。"""
        value = asdict(self)
        value["schema_version"] = "event-exchange-reconciliation/v2"
        return value


@dataclass(slots=True)
class _ChainState:
    side: Side
    game_idx: int
    available_amount: int = 0
    finalized: bool = False
    settled: bool = False


class EventExchangeReconcilerV2:
    """会計と物理観測を時系列で統合する。既存本番会計には接続しない。"""

    def __init__(self, *, fps: float = DEFAULT_VIDEO_FPS) -> None:
        if fps <= 0.0:
            raise ValueError("fpsは0より大きい必要があります")
        self._fps = fps
        self._ledger = ExchangeLedger(defer_max_sec_close=True)
        self._chains: dict[int, _ChainState] = {}
        self._game_idx = 0
        self._event_seq = 0
        self._landings: list[ReconciledLandingV2] = []
        self._timeline_rows: list[ReconciledTimelineRowV2] = []
        self._provisional_count = 0
        self._finalization_count = 0
        self._physical_row_count = 0
        self._raw_exact_candidate_count = 0
        self._occupancy_candidate_count = 0
        self._corroborated_zero_count = 0
        self._deferred_during_active_attack_count = 0
        self._ignored_nonexact_count = 0
        self._game_mismatch_count = 0
        self._late_confirmation_count = 0
        self._late_correction_count = 0
        self._held_finalization_count = 0
        self._late_held_finalization_count = 0
        self._total_canceled = 0
        self._total_gross_landed = 0
        self._total_boundary_expired = 0
        self._uncertain = False
        self._left_context_active = False
        self._uncertainty_started_count = 0
        self._uncertainty_resolved_by_growth_count = 0
        self._uncertainty_causes: set[str] = set()
        self._first_uncertain_frame: int | None = None

    def reconcile(
        self, accounting: EventAccountingSidecar, physical: EventPhysicalSidecar,
    ) -> ExchangeReconciliationReportV2:
        """同じ処理範囲の二つのサイドカーを統合する。"""
        _validate_matching_ranges(accounting, physical)
        if accounting.processing_start_frame != 0:
            self._uncertain = True
            self._left_context_active = True
            self._first_uncertain_frame = accounting.processing_start_frame
        for _frame, _priority, kind, value in _timeline(accounting, physical):
            self._apply_timeline_item(kind, value)
        return self._report(accounting)

    def _apply_timeline_item(self, kind: str, value: object) -> None:
        if kind == "boundary":
            self._advance_boundary(int(value))
        elif kind == "provisional":
            row, attack = value  # type: ignore[misc]
            self._observe_provisional(row, attack)
        elif kind == "finalize":
            row, attack = value  # type: ignore[misc]
            self._observe_finalization(row, attack)
        elif kind == "physical":
            self._observe_physical_landing(value)  # type: ignore[arg-type]

    def _advance_boundary(self, frame_idx: int) -> None:
        was_uncertain = self._uncertain
        expired = self._current_outstanding()
        self._total_boundary_expired += expired
        self._game_idx += 1
        self._ledger.push(ExchangeEvent(
            EventKind.TSUMO_PLACED, Side.P1, self._time(frame_idx),
            source="event_exchange_v2_formal_boundary",
        ), self._context())
        self._record_timeline(
            frame_idx, "formal_boundary", None, None, expired,
            status="unresolved_expired" if expired else "clean",
            observation_trusted=True,
            trusted=not was_uncertain,
        )
        self._uncertain = False
        self._left_context_active = False
        self._uncertainty_causes.clear()

    def _observe_provisional(
        self, row: AccountingRow, attack: ProvisionalAttack,
    ) -> None:
        self._provisional_count += 1
        side = SIDE_BY_LABEL[attack.side]
        state = self._chains.get(attack.resolver_chain_ordinal)
        if state is None:
            state = _ChainState(side, self._game_idx)
            self._chains[attack.resolver_chain_ordinal] = state
        if state.side is not side or state.game_idx != self._game_idx:
            raise ValueError("連鎖IDがsideまたは試合を跨いで再利用されました")
        amount = attack.provisional_generated_amount
        if amount < state.available_amount:
            raise ValueError("途中火力は同じ連鎖内で減少できません")
        kind = EventKind.FIRE if state.available_amount == 0 else EventKind.STEP
        delta = amount - state.available_amount
        self._push_attack(kind, side, row.frame_idx, attack.resolver_chain_ordinal, delta)
        state.available_amount = amount
        if delta > 0:
            self._resolve_growth_uncertainty(attack.resolver_chain_ordinal)
        mutual = self._cancel_mutual_outstanding(row.frame_idx)
        self._record_timeline(
            row.frame_idx, "provisional_update", side, attack.resolver_chain_ordinal,
            amount, generated_delta=delta, mutual=mutual,
            evidence=attack.mechanism, status="observed",
            observation_trusted=True,
        )

    def _observe_finalization(
        self, row: AccountingRow, attack: AttackFinalization,
    ) -> None:
        self._finalization_count += 1
        side = SIDE_BY_LABEL[attack.side]
        state = self._chains.get(attack.resolver_chain_ordinal)
        if state is None:
            state = _ChainState(side, self._game_idx)
            self._chains[attack.resolver_chain_ordinal] = state
        if state.side is not side or state.game_idx != self._game_idx:
            raise ValueError("確定攻撃の連鎖IDが観測履歴と一致しません")
        was_settled = state.settled
        previous_amount = state.available_amount
        self._ledger.push(ExchangeEvent(
            EventKind.FINALIZE, side, self._time(row.frame_idx),
            amount=float(attack.generated_amount),
            chain_id=attack.resolver_chain_ordinal,
            source=FINALIZE_SOURCE_SCORE_OCR_DIFF,
        ), self._context())
        accepted_amount = _exact_int(
            self._ledger.amount_of(attack.resolver_chain_ordinal)
        )
        finalized_amount = self._ledger.finalized_amount_of(
            attack.resolver_chain_ordinal
        )
        held = accepted_amount != attack.generated_amount
        if held:
            self._held_finalization_count += 1
            self._late_held_finalization_count += int(was_settled)
        elif was_settled and previous_amount == accepted_amount:
            self._late_confirmation_count += 1
        elif was_settled:
            self._late_correction_count += 1
        state.available_amount = accepted_amount
        state.finalized = finalized_amount is not None
        mutual = self._cancel_mutual_outstanding(row.frame_idx)
        late_status = (
            _held_finalization_status(was_settled) if held
            else _finalization_status(was_settled, previous_amount, accepted_amount)
        )
        self._record_timeline(
            row.frame_idx, "finalization", side, attack.resolver_chain_ordinal,
            attack.generated_amount,
            generated_delta=accepted_amount - previous_amount,
            mutual=mutual, evidence=FINALIZE_SOURCE_SCORE_OCR_DIFF,
            status=late_status,
            observation_trusted=not held,
        )

    def _push_attack(
        self, kind: EventKind, side: Side, frame_idx: int, chain_id: int, amount: int,
    ) -> None:
        self._ledger.push(ExchangeEvent(
            kind, side, self._time(frame_idx), amount=float(amount),
            chain_id=chain_id, source="event_exchange_v2_provisional",
        ), self._context())

    def _observe_physical_landing(self, row: PhysicalObservationRow) -> None:
        self._physical_row_count += 1
        recipient = SIDE_BY_LABEL[row.side]
        sender_total, recipient_total, expected = self._landing_balances(recipient)
        candidate = _landing_candidate(row)
        if candidate is None:
            self._observe_missing_candidate(row, recipient, expected)
            return
        visible, evidence = candidate
        if evidence == "raw_exact":
            self._raw_exact_candidate_count += 1
        else:
            self._occupancy_candidate_count += 1
        if row.game_idx != self._game_idx:
            self._start_uncertainty(row.frame_idx)
            self._game_mismatch_count += 1
            self._append_landing(
                row, visible, evidence, "game_index_mismatch",
            )
            self._record_timeline(
                row.frame_idx, "physical_landing", SIDE_BY_LABEL[row.side], None,
                visible, evidence=evidence,
                status="game_index_mismatch",
            )
            return
        status = _landing_status(visible, expected, row)
        gross = 0
        mutual = 0
        if status in ACCEPTED_LANDING_STATUSES:
            mutual = min(sender_total, recipient_total)
            gross = _gross_landing_amount(status, visible, expected)
            self._settle_exchange(row, recipient, mutual, gross)
        if status == "partial_visible_match" or status not in ACCEPTED_LANDING_STATUSES:
            self._start_uncertainty(row.frame_idx)
        elif self._can_restore_trust(status):
            self._uncertain = False
            self._uncertainty_causes.clear()
        self._append_landing(
            row, visible, evidence, status, sender_total, recipient_total, expected,
        )
        accepted = status in ACCEPTED_LANDING_STATUSES
        self._record_timeline(
            row.frame_idx, "physical_landing", recipient, None, visible,
            mutual=mutual, gross_landed=gross,
            visible_landed=visible if accepted else 0,
            evidence=evidence, status=status,
            observation_trusted=True,
        )

    def _landing_balances(self, recipient: Side) -> tuple[int, int, int]:
        sender_total = self._outstanding(recipient.other)
        recipient_total = self._outstanding(recipient)
        net = sender_total - recipient_total
        return sender_total, recipient_total, min(
            max(net, 0), OJAMA_MAX_DROP_PER_TURN,
        )

    def _observe_missing_candidate(
        self, row: PhysicalObservationRow, recipient: Side, expected: int,
    ) -> None:
        if _is_corroborated_zero(row):
            self._corroborated_zero_count += 1
            self._record_timeline(
                row.frame_idx, "physical_landing", recipient, None, 0,
                evidence="raw_occupancy_zero", status="corroborated_zero",
                observation_trusted=True,
            )
            return
        if self._is_active_unfinalized_attack(row, recipient, expected):
            self._deferred_during_active_attack_count += 1
            self._ignored_nonexact_count += 1
            self._record_timeline(
                row.frame_idx, "physical_landing", recipient, None, 0,
                evidence="opponent_chain_active_unfinalized_only",
                status="deferred_during_active_attack",
            )
            return
        if expected > 0 or _has_unexplained_positive_change(row):
            cause = (
                self._growth_uncertainty_cause(recipient.other, row.frame_idx)
                if expected > 0 else None
            )
            self._start_uncertainty(row.frame_idx, cause=cause)
        self._ignored_nonexact_count += 1
        self._record_ignored_physical(row)

    def _is_active_unfinalized_attack(
        self, row: PhysicalObservationRow, recipient: Side, expected: int,
    ) -> bool:
        """未完走の相手連鎖だけが供給源なら、その時点の着地ではない。"""
        return (
            expected > 0
            and row.payload.get("opponent_chain_active_at_fall_start") is True
            and self._finalized_outstanding(recipient.other) == 0
        )

    def _settle_exchange(
        self, row: PhysicalObservationRow, recipient: Side, mutual: int, gross: int,
    ) -> None:
        t_sec = self._time(row.frame_idx)
        self._allocate(recipient.other, mutual, EventKind.CANCEL, recipient.other, t_sec)
        self._allocate(recipient, mutual, EventKind.CANCEL, recipient, t_sec)
        self._allocate(recipient.other, gross, EventKind.LAND, recipient, t_sec)
        self._total_canceled += mutual * 2
        self._total_gross_landed += gross

    def _cancel_mutual_outstanding(self, frame_idx: int) -> int:
        """同時に存在する両者の火力を、観測時点で因果的に相殺する。"""
        mutual = min(self._outstanding(Side.P1), self._outstanding(Side.P2))
        if mutual <= 0:
            return 0
        t_sec = self._time(frame_idx)
        self._allocate(Side.P1, mutual, EventKind.CANCEL, Side.P1, t_sec)
        self._allocate(Side.P2, mutual, EventKind.CANCEL, Side.P2, t_sec)
        self._total_canceled += mutual * 2
        return mutual

    def _record_ignored_physical(self, row: PhysicalObservationRow) -> None:
        differences = row.payload.get("differences")
        visible = differences.get("garbage", 0) if isinstance(differences, dict) else 0
        if isinstance(visible, bool) or not isinstance(visible, int):
            visible = 0
        status = str(row.payload.get("comparison_state", "unclassified"))
        self._record_timeline(
            row.frame_idx, "physical_landing", SIDE_BY_LABEL[row.side], None,
            max(visible, 0), evidence="untrusted", status=f"ignored_{status}",
        )

    def _start_uncertainty(self, frame_idx: int, *, cause: str | None = None) -> None:
        was_uncertain = self._uncertain
        self._uncertainty_causes.add(cause or f"physical:{frame_idx}")
        self._uncertain = True
        if not was_uncertain:
            self._uncertainty_started_count += 1
            if self._first_uncertain_frame is None:
                self._first_uncertain_frame = frame_idx

    def _growth_uncertainty_cause(self, sender: Side, frame_idx: int) -> str:
        candidates = [
            chain_id for chain_id in self._open_chain_ids(sender)
            if not self._chains[chain_id].finalized
            and self._ledger.outstanding_of(chain_id) > 0.0
        ]
        if len(candidates) == 1:
            return f"growing_chain:{candidates[0]}"
        return f"physical:{frame_idx}"

    def _resolve_growth_uncertainty(self, chain_id: int) -> None:
        cause = f"growing_chain:{chain_id}"
        if cause not in self._uncertainty_causes:
            return
        self._uncertainty_causes.remove(cause)
        self._uncertainty_resolved_by_growth_count += 1
        if not self._uncertainty_causes and not self._left_context_active:
            self._uncertain = False

    def _can_restore_trust(self, landing_status: str) -> bool:
        return (
            landing_status in {"exact_match", "capacity_limited_match"}
            and self._current_outstanding() == 0
            and not self._left_context_active
        )

    def _record_timeline(
        self, frame_idx: int, event_type: str, side: Side | None,
        chain_id: int | None, observed: int, *, generated_delta: int = 0,
        mutual: int = 0, gross_landed: int = 0, visible_landed: int = 0,
        evidence: str = "", status: str, observation_trusted: bool = False,
        trusted: bool | None = None,
    ) -> None:
        p1 = self._outstanding(Side.P1)
        p2 = self._outstanding(Side.P2)
        self._timeline_rows.append(ReconciledTimelineRowV2(
            frame_idx, self._game_idx, event_type,
            None if side is None else side.name.lower(), chain_id,
            observed, generated_delta, mutual, gross_landed, visible_landed,
            p1, p2, p1 - p2,
            observation_trusted,
            not self._uncertain if trusted is None else trusted, evidence, status,
        ))

    def _allocate(
        self, owner: Side, amount: int, kind: EventKind, event_side: Side, t_sec: float,
    ) -> None:
        remaining = float(amount)
        for chain_id in self._open_chain_ids(owner):
            if remaining <= 0.0:
                break
            take = min(remaining, self._ledger.outstanding_of(chain_id))
            if take <= 0.0:
                continue
            self._event_seq += 1
            self._ledger.push(ExchangeEvent(
                kind, event_side, t_sec, amount=take, chain_id=chain_id,
                source="event_exchange_v2_physical", seq=self._event_seq,
            ), self._context())
            self._chains[chain_id].settled = True
            remaining -= take
        if remaining > 1e-9:
            raise ValueError("実着地を帰属できる途中火力が不足しています")

    def _append_landing(
        self, row: PhysicalObservationRow, visible: int, evidence: str, status: str,
        sender: int = 0, recipient: int = 0, expected: int = 0,
    ) -> None:
        accepted = status in ACCEPTED_LANDING_STATUSES
        mutual = min(sender, recipient) if accepted else 0
        gross = _gross_landing_amount(status, visible, expected) if accepted else 0
        self._landings.append(ReconciledLandingV2(
            row.frame_idx, row.game_idx, row.side, visible, expected, gross,
            mutual, sender, recipient, evidence, status,
        ))

    def _open_chain_ids(self, side: Side) -> list[int]:
        return [
            chain_id for chain_id in self._ledger.open_chain_ids(side)
            if self._chains[chain_id].game_idx == self._game_idx
        ]

    def _outstanding(self, side: Side) -> int:
        return _exact_int(sum(
            self._ledger.outstanding_of(chain_id)
            for chain_id in self._open_chain_ids(side)
        ))

    def _finalized_outstanding(self, side: Side) -> int:
        return _exact_int(sum(
            self._ledger.outstanding_of(chain_id)
            for chain_id in self._open_chain_ids(side)
            if self._chains[chain_id].finalized
        ))

    def _current_outstanding(self) -> int:
        return self._outstanding(Side.P1) + self._outstanding(Side.P2)

    def _context(self) -> PhysicalContext:
        return PhysicalContext(game_idx=self._game_idx)

    def _time(self, frame_idx: int) -> float:
        return frame_idx / self._fps

    def _report(
        self, accounting: EventAccountingSidecar,
    ) -> ExchangeReconciliationReportV2:
        snapshot = self._ledger.snapshot(self._context())
        statuses = [item.status for item in self._landings]
        generated = sum(state.available_amount for state in self._chains.values())
        finalized = sum(
            state.available_amount for state in self._chains.values() if state.finalized
        )
        visible = sum(
            item.visible_amount for item in self._landings
            if item.status in ACCEPTED_LANDING_STATUSES
        )
        outstanding = self._current_outstanding()
        residual = (
            generated - self._total_canceled - self._total_gross_landed
            - self._total_boundary_expired - outstanding
        )
        return ExchangeReconciliationReportV2(
            self._provisional_count, self._finalization_count,
            accounting.processing_start_frame != 0, self._physical_row_count,
            self._raw_exact_candidate_count, self._occupancy_candidate_count,
            self._corroborated_zero_count,
            self._deferred_during_active_attack_count,
            statuses.count("exact_match"),
            statuses.count("capacity_limited_match"),
            statuses.count("partial_visible_match"),
            sum(status not in ACCEPTED_LANDING_STATUSES for status in statuses),
            self._ignored_nonexact_count, self._game_mismatch_count,
            self._late_confirmation_count, self._late_correction_count,
            self._held_finalization_count,
            self._late_held_finalization_count,
            self._uncertainty_started_count,
            self._uncertainty_resolved_by_growth_count,
            sum(not row.observation_trusted for row in self._timeline_rows),
            sum(not row.trusted for row in self._timeline_rows),
            self._first_uncertain_frame, not self._uncertain,
            _legacy_total(accounting, "offset_uncapped"),
            _legacy_total(accounting, "dropped_uncapped"),
            generated, finalized, generated - finalized,
            self._total_canceled, self._total_gross_landed, visible,
            self._total_boundary_expired, outstanding, residual,
            _exact_int(snapshot.oversettled_total), _exact_int(snapshot.net_raw),
            tuple(self._landings), tuple(self._timeline_rows),
        )


def reconcile_event_exchange_v2(
    accounting: EventAccountingSidecar, physical: EventPhysicalSidecar,
    *, fps: float = DEFAULT_VIDEO_FPS,
) -> ExchangeReconciliationReportV2:
    """新しい交換会計を一回実行する公開関数。"""
    return EventExchangeReconcilerV2(fps=fps).reconcile(accounting, physical)


def validate_exchange_reconciliation_report_v2(
    report: ExchangeReconciliationReportV2,
) -> None:
    """成果物内部の保存則と時系列集計が一致することを検査する。"""
    rows = report.timeline_rows
    checks = {
        "保存則残差": report.conservation_residual,
        "超過決済": report.oversettled_total,
        "生成量": sum(row.generated_delta for row in rows) - report.total_generated,
        "相殺量": (
            2 * sum(row.mutual_canceled_amount for row in rows)
            - report.total_canceled
        ),
        "実着地量": (
            sum(row.gross_landed_amount for row in rows)
            - report.total_gross_landed
        ),
        "可視着地量": (
            sum(row.visible_landed_amount for row in rows)
            - report.total_visible_landed
        ),
        "境界失効量": (
            sum(row.observed_amount for row in rows if row.event_type == "formal_boundary")
            - report.total_boundary_expired
        ),
        "不確実行数": (
            sum(not row.trusted for row in rows)
            - report.uncertain_timeline_row_count
        ),
        "観測不可信行数": (
            sum(not row.observation_trusted for row in rows)
            - report.untrusted_observation_row_count
        ),
    }
    failures = {name: value for name, value in checks.items() if value != 0}
    if failures:
        raise ValueError(f"交換会計v2の内部集計が一致しません: {failures}")
    _validate_timeline_end(report)


def _validate_timeline_end(report: ExchangeReconciliationReportV2) -> None:
    rows = report.timeline_rows
    if any(current.frame_idx > following.frame_idx
           for current, following in zip(rows, rows[1:])):
        raise ValueError("交換会計v2の時系列が逆行しています")
    if not rows:
        if report.total_outstanding != 0 or report.net_raw != 0:
            raise ValueError("時系列が空なのに未決着量があります")
        return
    last = rows[-1]
    if last.p1_outstanding_after + last.p2_outstanding_after != report.total_outstanding:
        raise ValueError("最終時系列行と未決着合計が一致しません")
    if last.net_after != report.net_raw:
        raise ValueError("最終時系列行と純残量が一致しません")


def _timeline(
    accounting: EventAccountingSidecar, physical: EventPhysicalSidecar,
) -> list[tuple[int, int, str, object]]:
    """収集順の境界→途中値→確定値→物理盤面を同一フレームでも維持する。"""
    items: list[tuple[int, int, str, object]] = []
    for row in accounting.rows:
        if row.formal_boundary:
            items.append((
                row.frame_idx, TIMELINE_PRIORITY["boundary"],
                "boundary", row.frame_idx,
            ))
        items.extend((row.frame_idx, TIMELINE_PRIORITY["provisional"],
                      "provisional", (row, item))
                     for item in row.provisional_updates)
        items.extend((row.frame_idx, TIMELINE_PRIORITY["finalize"],
                      "finalize", (row, item)) for item in row.attacks)
    items.extend((row.frame_idx, TIMELINE_PRIORITY["physical"], "physical", row)
                 for row in physical.rows
                 if row.observation_type == "landing_board_compared")
    return sorted(items, key=lambda item: (item[0], item[1]))


def _landing_candidate(row: PhysicalObservationRow) -> tuple[int, str] | None:
    payload = row.payload
    if payload.get("before_is_immediate_previous_observation") is not True:
        return None
    differences = payload.get("differences")
    if not isinstance(differences, dict):
        return None
    amount = differences.get("garbage")
    if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
        return None
    if not _is_garbage_only_addition(differences, amount):
        return None
    if payload.get("confirmed_raw_agreement") is True:
        return amount, "raw_exact"
    if _occupancy_corroborates(payload, amount):
        return amount, "raw_occupancy_corroborated"
    return None


def _is_garbage_only_addition(differences: dict[str, object], amount: int) -> bool:
    return (
        differences.get("color") == 0
        and differences.get("unknown") == 0
        and differences.get("occupied") == amount
    )


def _occupancy_corroborates(payload: dict[str, Any], amount: int) -> bool:
    """色分類が揺れても、生画像の占有増加が同量なら下限量を認める。"""
    raw = payload.get("raw_differences")
    if payload.get("before_is_immediate_previous_observation") is not True:
        return False
    if not isinstance(raw, dict) or raw.get("occupied") != amount:
        return False
    values = [raw.get(name) for name in ("color", "garbage", "unknown")]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return False
    return sum(values) == amount


def _is_corroborated_zero(row: PhysicalObservationRow) -> bool:
    payload = row.payload
    differences = payload.get("differences")
    if payload.get("before_is_immediate_previous_observation") is not True:
        return False
    names = ("color", "garbage", "unknown", "occupied")
    if not isinstance(differences, dict):
        return False
    if any(differences.get(name) != 0 for name in names):
        return False
    if payload.get("confirmed_raw_agreement") is True:
        return True
    raw = payload.get("raw_differences")
    if not isinstance(raw, dict) or raw.get("occupied") != 0:
        return False
    values = [raw.get(name) for name in ("color", "garbage", "unknown")]
    valid = all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    )
    return valid and sum(values) == 0


def _has_unexplained_positive_change(row: PhysicalObservationRow) -> bool:
    """着地予定0でも会計欠落を疑う、確定盤面上の正方向変化を判定する。"""
    payload = row.payload
    if payload.get("before_is_immediate_previous_observation") is not True:
        return False
    differences = payload.get("differences")
    if not isinstance(differences, dict):
        return False
    values = [differences.get(name) for name in ("garbage", "occupied")]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return False
    return any(value > 0 for value in values)


def _landing_status(
    visible: int, expected: int, row: PhysicalObservationRow,
) -> str:
    if expected <= 0:
        return "direction_conflict"
    if visible == expected:
        return "exact_match"
    visible_range = _visible_landing_range(row, expected)
    if (visible < expected and visible_range is not None
            and visible_range[0] <= visible <= visible_range[1]):
        return "capacity_limited_match"
    if 0 < visible < expected:
        return "partial_visible_match"
    return "amount_conflict"


def _finalization_status(settled: bool, previous: int, confirmed: int) -> str:
    if not settled:
        return "finalized"
    if previous == confirmed:
        return "late_confirmation"
    return "late_correction"


def _held_finalization_status(settled: bool) -> str:
    return "late_finalization_held" if settled else "finalization_held"


def _gross_landing_amount(status: str, visible: int, expected: int) -> int:
    if status == "partial_visible_match":
        return visible
    return expected


def _visible_landing_range(
    row: PhysicalObservationRow, gross: int,
) -> tuple[int, int] | None:
    """列ごとの空きを使い、端数列が未知でも取り得る可視着地範囲を返す。"""
    grid = row.payload.get("before_grid")
    if not isinstance(grid, list) or len(grid) * BOARD_WIDTH != BOARD_CELL_COUNT:
        return None
    if any(not isinstance(line, list) or len(line) != BOARD_WIDTH for line in grid):
        return None
    if any(isinstance(cell, bool) or not isinstance(cell, int)
           for line in grid for cell in line):
        return None
    rooms = [sum(line[col] == 0 for line in grid) for col in range(BOARD_WIDTH)]
    base, remainder = divmod(gross, BOARD_WIDTH)
    base_visible = sum(min(room, base) for room in rooms)
    extra_gains = sorted(int(room > base) for room in rooms)
    minimum = base_visible + sum(extra_gains[:remainder])
    maximum = base_visible + sum(extra_gains[-remainder:] if remainder else ())
    return minimum, maximum


def _legacy_total(sidecar: EventAccountingSidecar, prefix: str) -> int:
    return sum(
        row.deltas[f"{prefix}_{side}"]
        for row in sidecar.rows for side in ("p1", "p2")
    )


def _validate_matching_ranges(
    accounting: EventAccountingSidecar, physical: EventPhysicalSidecar,
) -> None:
    accounting_range = (
        accounting.processing_start_frame, accounting.processing_end_frame_exclusive,
    )
    physical_range = (
        physical.processing_start_frame, physical.processing_end_frame_exclusive,
    )
    if accounting_range != physical_range:
        raise ValueError("会計と物理観測の処理範囲が一致しません")


def _exact_int(value: float) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise ValueError("交換会計量が整数ではありません")
    return int(rounded)


__all__ = [
    "EventExchangeReconcilerV2",
    "ExchangeReconciliationReportV2",
    "ReconciledLandingV2",
    "ReconciledTimelineRowV2",
    "reconcile_event_exchange_v2",
    "validate_exchange_reconciliation_report_v2",
]
