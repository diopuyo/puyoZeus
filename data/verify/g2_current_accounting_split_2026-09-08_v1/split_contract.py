"""現在候補と過去消去を分離する私有 CPU 契約。実証拠・本番接続は未実装。

policy は証拠ごとに拒否例外または None を返す。bool は認証に使わない。
単一 writer の不変 state 交換だけを保証し、外部 pipeline の原子性は保証しない。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Callable, Protocol

ROWS, COLS, COLORS = 13, 6, 5
EMPTY, OJAMA = 0, 9
VALID_CELLS = frozenset((EMPTY, 1, 2, 3, 4, 5, OJAMA))
Grid = tuple[tuple[int, ...], ...]
Counts = tuple[int, ...]


class Rejected(ValueError):
    """証拠・scope・時系列・会計の契約違反。"""


@dataclass(frozen=True)
class Scope:
    source_sha256: str
    run_id: str
    side: str
    game_id: str
    reset_epoch: int


@dataclass(frozen=True)
class Clock:
    frame: int
    time_sec: float
    sequence: int


@dataclass(frozen=True)
class BaselineEvidence:
    scope: Scope
    available_at: Clock
    counts: Counts
    evidence_id: str
    covers_through: Clock


@dataclass(frozen=True)
class CurrentEvidence:
    scope: Scope
    action: int
    observed_at: Clock
    available_at: Clock
    state: str
    grid: Grid
    evidence_id: str


@dataclass(frozen=True)
class OriginEvidence:
    scope: Scope
    origin_id: str
    action: int
    observed_at: Clock
    available_at: Clock
    before_grid: Grid
    predicted_final: Grid
    erased: Counts
    prediction_sha256: str
    event_identity: str


@dataclass(frozen=True)
class PlacementEvidence:
    scope: Scope
    event_id: str
    action: int
    available_at: Clock
    added: Counts
    occurred_at: Clock


@dataclass(frozen=True)
class JournalEntry:
    event_id: str
    kind: str
    delta: Counts
    action: int
    available_at: Clock


@dataclass(frozen=True)
class CounterClaim:
    counts: Counts
    revision: int
    history: tuple[JournalEntry, ...]
    consumed_ids: tuple[str, ...]


@dataclass(frozen=True)
class SettlementEvidence:
    scope: Scope
    origin: OriginEvidence
    available_at: Clock
    claim: CounterClaim
    evidence_id: str


@dataclass(frozen=True)
class Debt:
    origin: OriginEvidence
    baseline: CounterClaim


@dataclass(frozen=True)
class CurrentSlot:
    grid: Grid
    action: int
    observed_at: Clock
    revision: int
    evidence_id: str
    available: bool = True


@dataclass(frozen=True)
class OwnerState:
    scope: Scope
    action: int = 0
    clock: Clock | None = None
    current: CurrentSlot | None = None
    baseline: Counts | None = None
    counter: Counts | None = None
    counter_revision: int = 0
    history: tuple[JournalEntry, ...] = ()
    debts: tuple[Debt, ...] = ()
    consumed_ids: tuple[str, ...] = ()
    baseline_through: Clock | None = None
    origins: tuple[OriginEvidence, ...] = ()
    action_since: Clock | None = None

    @property
    def accounting_available(self) -> bool:
        return self.baseline is not None and not self.debts

    @property
    def live_permission(self) -> bool:
        return False


class EvidencePolicy(Protocol):
    def authorize(self, purpose: str, evidence: object, state: OwnerState) -> None:
        """外部証拠 producer は未接続。人工 CPU fixture 以外を認定しない。"""
        ...


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Rejected(reason)


def integer(value: object) -> bool:
    return type(value) is int and value >= 0


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def validate_scope(value: Scope) -> None:
    require(type(value) is Scope, "scope_type")
    require(value.side in ("1P", "2P"), "side")
    require(integer(value.reset_epoch), "reset_epoch")
    for item in (value.source_sha256, value.run_id, value.game_id):
        require(type(item) is str and bool(item), "scope_identity")
    require(len(value.source_sha256) == 64, "source_sha256")
    require(all(c in "0123456789abcdef" for c in value.source_sha256), "source_sha256")


def validate_clock(value: Clock) -> None:
    require(type(value) is Clock, "clock_type")
    require(integer(value.frame) and integer(value.sequence), "clock_integer")
    require(type(value.time_sec) in (int, float), "clock_time_type")
    require(math.isfinite(value.time_sec) and value.time_sec >= 0, "clock_time")


def validate_grid(value: Grid) -> None:
    require(type(value) is tuple and len(value) == ROWS, "grid_rows")
    for row in value:
        require(type(row) is tuple and len(row) == COLS, "grid_cols")
        require(all(type(c) is int and c in VALID_CELLS for c in row), "grid_cell")


def validate_counts(value: Counts) -> None:
    require(type(value) is tuple and len(value) == COLORS, "counts_shape")
    require(all(integer(c) for c in value), "counts_value")


def available(point: Clock, now: Clock) -> None:
    validate_clock(point)
    require(point.frame <= now.frame and point.time_sec <= now.time_sec
            and point.sequence <= now.sequence, "future_evidence")


def color_counts(grid: Grid) -> Counts:
    return tuple(sum(row.count(color) for row in grid) for color in range(1, COLORS + 1))


def prediction_digest(origin: OriginEvidence) -> str:
    return digest((origin.before_grid, origin.predicted_final, origin.erased))


def origin_key(origin: OriginEvidence) -> tuple[Scope, int, str]:
    """便宜 ID・予測版を除く occurrence。実 producer の発行は未接続。"""
    return origin.scope, origin.action, origin.event_identity


def validate_origin(origin: OriginEvidence) -> None:
    require(type(origin) is OriginEvidence, "origin_type")
    validate_scope(origin.scope)
    require(type(origin.origin_id) is str and bool(origin.origin_id), "origin_id")
    require(type(origin.event_identity) is str and bool(origin.event_identity), "event_identity")
    require(integer(origin.action), "origin_action")
    validate_clock(origin.observed_at)
    validate_clock(origin.available_at)
    available(origin.observed_at, origin.available_at)
    validate_grid(origin.before_grid)
    validate_grid(origin.predicted_final)
    validate_counts(origin.erased)
    difference = tuple(a - b for a, b in zip(color_counts(origin.before_grid),
                                            color_counts(origin.predicted_final)))
    require(difference == origin.erased and sum(difference) > 0, "origin_erase")
    require(origin.prediction_sha256 == prediction_digest(origin), "prediction_digest")


def counter_claim(state: OwnerState) -> CounterClaim:
    require(state.counter is not None, "unknown_baseline")
    return CounterClaim(state.counter, state.counter_revision, state.history, state.consumed_ids)


def validate_claim(claim: CounterClaim) -> None:
    require(type(claim) is CounterClaim, "claim_type")
    validate_counts(claim.counts)
    require(integer(claim.revision), "claim_revision")
    require(type(claim.history) is tuple and type(claim.consumed_ids) is tuple, "claim_tuple")
    require(all(type(item) is str for item in claim.consumed_ids), "claim_consumed")
    for entry in claim.history:
        require(type(entry) is JournalEntry and integer(entry.action), "claim_entry")
        require(type(entry.delta) is tuple and len(entry.delta) == COLORS
                and all(type(c) is int for c in entry.delta), "claim_delta")
        validate_clock(entry.available_at)


def validate_accounting(state: OwnerState) -> None:
    if state.baseline is None:
        require(state.counter is None and not state.history and not state.debts,
                "unknown_accounting_changed")
        return
    validate_counts(state.baseline)
    total = state.baseline
    seen: set[str] = set()
    consumed: list[str] = []
    for entry in state.history:
        require(entry.event_id not in seen, "duplicate_history")
        seen.add(entry.event_id)
        require(entry.kind in ("placement", "settlement"), "unknown_history_kind")
        total = tuple(a + b for a, b in zip(total, entry.delta))
        validate_counts(total)
        if entry.kind == "settlement":
            consumed.append(entry.event_id)
    require(total == state.counter and state.counter_revision == len(state.history), "counter_history")
    require(tuple(consumed) == state.consumed_ids, "consumed_history")
    for debt in state.debts:
        require(debt.origin.origin_id not in seen, "debt_consumed")
        revision = debt.baseline.revision
        require(debt.baseline.history == state.history[:revision], "debt_history_prefix")
        require(debt.baseline.consumed_ids == tuple(e.event_id for e in state.history[:revision]
                if e.kind == "settlement"), "debt_consumed_prefix")
        prefix = state.baseline
        for entry in state.history[:revision]:
            prefix = tuple(a + b for a, b in zip(prefix, entry.delta))
        require(prefix == debt.baseline.counts, "debt_counter_prefix")
        require(all(a >= b for a, b in zip(prefix, debt.origin.erased)), "origin_prefix_underflow")


class SplitOwner:
    """単一 scope の私有 owner。reset は別 owner、未清算旧 owner は廃棄しない。"""

    def __init__(self, scope: Scope, policy: EvidencePolicy | None = None) -> None:
        validate_scope(scope)
        self._state = OwnerState(copy.deepcopy(scope))
        self._policy = policy
        self._busy = False
        self._reentered = False

    @property
    def state(self) -> OwnerState:
        return self._state

    def _authorize(self, purpose: str, evidence: object, state: OwnerState) -> None:
        require(self._policy is not None, "policy_unconnected")
        result = self._policy.authorize(purpose, copy.deepcopy(evidence), copy.deepcopy(state))
        require(result is None, "policy_boolean_or_payload_not_proof")
        require(not self._reentered, "policy_reentry")

    def _apply(self, now: Clock, build: Callable[[OwnerState], OwnerState]) -> OwnerState:
        if self._busy:
            self._reentered = True
            raise Rejected("reentry")
        self._busy, self._reentered = True, False
        try:
            validate_clock(now)
            old = self._state
            if old.clock is not None:
                available(old.clock, now)
                require(now.sequence > old.clock.sequence, "duplicate_clock")
            validate_accounting(old)
            proposed = build(old)
            validate_accounting(proposed)
            require(not self._reentered, "reentry")
            proposed = replace(proposed, clock=copy.deepcopy(now))
            self._state = proposed
            return proposed
        finally:
            self._busy = False

    def _scope(self, scope: Scope, state: OwnerState) -> None:
        validate_scope(scope)
        require(scope == state.scope, "scope_mismatch")

    def establish_baseline(self, evidence: BaselineEvidence, now: Clock) -> OwnerState:
        def build(state: OwnerState) -> OwnerState:
            require(type(evidence) is BaselineEvidence, "baseline_type")
            self._scope(evidence.scope, state)
            validate_counts(evidence.counts)
            available(evidence.available_at, now)
            available(evidence.covers_through, evidence.available_at)
            require(state.baseline is None, "baseline_already_set")
            require(type(evidence.evidence_id) is str and bool(evidence.evidence_id), "evidence_id")
            self._authorize("baseline", evidence, state)
            return replace(state, baseline=evidence.counts, counter=evidence.counts,
                           baseline_through=evidence.covers_through)
        return self._apply(now, build)

    def register_debt(self, origin: OriginEvidence, now: Clock) -> OwnerState:
        def build(state: OwnerState) -> OwnerState:
            validate_origin(origin)
            self._scope(origin.scope, state)
            available(origin.available_at, now)
            require(origin.action == state.action, "registration_action")
            require(not state.debts, "multiple_outstanding_origins_not_supported")
            require(all(origin_key(item) != origin_key(origin) for item in state.origins), "origin_occurrence_duplicate")
            require(origin.origin_id not in state.consumed_ids
                    and all(d.origin.origin_id != origin.origin_id for d in state.debts), "origin_duplicate")
            claim = counter_claim(state)
            available(state.baseline_through, origin.observed_at)
            if state.action_since is not None:
                available(state.action_since, origin.observed_at)
            for entry in state.history:
                available(entry.available_at, origin.observed_at)
            require(all(a >= b for a, b in zip(claim.counts, origin.erased)), "origin_underflow")
            self._authorize("origin", origin, state)
            saved = copy.deepcopy(origin)
            return replace(state, debts=state.debts + (Debt(saved, claim),), origins=state.origins + (saved,))
        return self._apply(now, build)

    def advance_action(self, action: int, now: Clock) -> OwnerState:
        def build(state: OwnerState) -> OwnerState:
            require(integer(action) and action > state.action, "action_not_forward")
            held = replace(state.current, available=False) if state.current else None
            return replace(state, action=action, current=held, action_since=now)
        return self._apply(now, build)

    def recover_current(self, evidence: CurrentEvidence, now: Clock) -> OwnerState:
        def build(state: OwnerState) -> OwnerState:
            require(type(evidence) is CurrentEvidence, "current_type")
            self._scope(evidence.scope, state)
            require(integer(evidence.action) and evidence.action == state.action, "current_action")
            validate_clock(evidence.observed_at)
            available(evidence.available_at, now)
            require(evidence.observed_at == evidence.available_at == now, "current_not_fresh")
            require(evidence.state == "STABLE", "current_not_stable")
            validate_grid(evidence.grid)
            require(type(evidence.evidence_id) is str and bool(evidence.evidence_id), "evidence_id")
            require(all(d.origin.action < state.action for d in state.debts), "current_not_new_action")
            require(state.current is None or state.current.action < state.action, "current_same_action")
            self._authorize("current", evidence, state)
            revision = 1 if state.current is None else state.current.revision + 1
            slot = CurrentSlot(copy.deepcopy(evidence.grid), state.action, now, revision, evidence.evidence_id)
            return replace(state, current=slot)
        return self._apply(now, build)

    def add_placement(self, evidence: PlacementEvidence, now: Clock) -> OwnerState:
        def build(state: OwnerState) -> OwnerState:
            require(type(evidence) is PlacementEvidence, "placement_type")
            self._scope(evidence.scope, state)
            available(evidence.available_at, now)
            available(evidence.occurred_at, evidence.available_at)
            require(state.baseline_through is not None, "unknown_baseline")
            available(state.baseline_through, evidence.occurred_at)
            require(evidence.occurred_at.sequence > state.baseline_through.sequence, "placement_in_baseline")
            if state.action_since is not None:
                available(state.action_since, evidence.occurred_at)
            validate_counts(evidence.added)
            require(sum(evidence.added) == 2, "placement_not_pair")
            require(integer(evidence.action) and evidence.action == state.action, "placement_action")
            require(type(evidence.event_id) is str and bool(evidence.event_id), "placement_id")
            require(all(e.event_id != evidence.event_id and not (e.kind == "placement"
                    and e.action == state.action) for e in state.history), "placement_duplicate")
            require(all(d.origin.origin_id != evidence.event_id for d in state.debts), "identity_collision")
            claim = counter_claim(state)
            self._authorize("placement", evidence, state)
            entry = JournalEntry(evidence.event_id, "placement", evidence.added, state.action, now)
            counter = tuple(a + b for a, b in zip(claim.counts, evidence.added))
            return replace(state, counter=counter, counter_revision=state.counter_revision + 1,
                           history=state.history + (entry,))
        return self._apply(now, build)

    def settle(self, evidence: SettlementEvidence, now: Clock) -> OwnerState:
        def build(state: OwnerState) -> OwnerState:
            require(type(evidence) is SettlementEvidence, "settlement_type")
            self._scope(evidence.scope, state)
            validate_origin(evidence.origin)
            available(evidence.available_at, now)
            available(evidence.origin.available_at, evidence.available_at)
            require(evidence.origin.scope == state.scope, "origin_scope")
            validate_claim(evidence.claim)
            require(evidence.claim == counter_claim(state), "stale_counter_claim")
            require(type(evidence.evidence_id) is str and bool(evidence.evidence_id), "evidence_id")
            debt = next((d for d in state.debts if d.origin == evidence.origin), None)
            require(debt is not None and evidence.origin.origin_id not in state.consumed_ids, "unknown_or_consumed_origin")
            counter = tuple(a - b for a, b in zip(evidence.claim.counts, evidence.origin.erased))
            validate_counts(counter)
            self._authorize("settlement", evidence, state)
            entry = JournalEntry(evidence.origin.origin_id, "settlement", tuple(-c for c in evidence.origin.erased),
                                 evidence.origin.action, now)
            return replace(state, counter=counter, counter_revision=state.counter_revision + 1,
                           history=state.history + (entry,), debts=tuple(d for d in state.debts if d is not debt),
                           consumed_ids=state.consumed_ids + (evidence.origin.origin_id,))
        return self._apply(now, build)
