"""会計観測サイドカーを攻撃・相殺・送付・着地の出来事へ変換する。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.chain_detector import CHAIN_MECHANISM_FORMULA, CHAIN_MECHANISM_FORMULA_READ
from src.event_accounting_observer_v1 import (
    ACCOUNTING_OBSERVER_VERSION,
    ACCOUNTING_SIDECAR_VERSION,
)
from src.event_snapshot_adapter_v1 import VideoTimeBase
from src.event_source_v1 import SCHEMA_VERSION
from src.ojama_accounting import GrossOjamaCounters


ACCOUNTING_ADAPTER_VERSION = "event-accounting-adapter/v1.2"
COUNTER_NAMES = tuple(
    name for name in GrossOjamaCounters.__dataclass_fields__ if name != "t_sec"
)
SIDE_ORDER = ("p1", "p2")


class EventAccountingAdapterError(ValueError):
    """会計観測を安全に出来事化できない場合の例外。"""


@dataclass(frozen=True, slots=True)
class AttackFinalization:
    side: str
    side_ordinal: int
    chain_total_score: int
    generated_amount: int
    effective_rate: int
    leftover_before: int
    leftover_after: int
    resolver_chain_ordinal: int


@dataclass(frozen=True, slots=True)
class ProvisionalAttack:
    side: str
    resolver_chain_ordinal: int
    update_ordinal: int
    chain_count: int
    provisional_score: int
    provisional_generated_amount: int
    effective_rate: int
    mechanism: str


@dataclass(frozen=True, slots=True)
class AccountingRow:
    frame_idx: int
    pending_before: tuple[int, int]
    pending_after: tuple[int, int]
    deltas: dict[str, int]
    attacks: tuple[AttackFinalization, ...]
    provisional_updates: tuple[ProvisionalAttack, ...] = ()
    conservation_residual: tuple[int, int] = (0, 0)
    observation_window_start_frame: int = 0
    formal_boundary: bool = False


@dataclass(frozen=True, slots=True)
class EventAccountingSidecar:
    processing_start_frame: int
    processing_end_frame_exclusive: int
    observed_frame_count: int
    inspected_side_count: int
    initial_gross: dict[str, int]
    final_gross: dict[str, int]
    final_pending: tuple[int, int]
    rows: tuple[AccountingRow, ...]


@dataclass(slots=True)
class _Lot:
    lot_id: str
    sender: str
    recipient: str
    remaining: int
    created_frame: int


class _EventBuilder:
    """FIFO帰属を保ちながら会計行を出来事列へ展開する。"""

    def __init__(self, sidecar: EventAccountingSidecar, source_video_id: str,
                 build_id: str, attempt_id: str, time_base: VideoTimeBase) -> None:
        self.sidecar = sidecar
        self.source_video_id = source_video_id
        self.build_id = build_id
        self.attempt_id = attempt_id
        self.time_base = time_base
        self.events: list[dict[str, Any]] = []
        self.lots: list[_Lot] = []
        self.fall_ordinals = {"p1": 0, "p2": 0}
        self.provisional_tokens: dict[
            tuple[str, int], list[tuple[str, int, int]]
        ] = {}
        self.open_provisional_keys: dict[str, list[tuple[str, int]]] = {
            "p1": [], "p2": [],
        }
        self.accounting_source_order = 0

    def build(self) -> tuple[dict[str, Any], ...]:
        for row in self.sidecar.rows:
            self._build_row(row)
        last_frame = self.sidecar.processing_end_frame_exclusive - 1
        for side in SIDE_ORDER:
            self._close_provisionals(last_frame, side, "processing_end")
        return tuple(self.events)

    def _build_row(self, row: AccountingRow) -> None:
        self.accounting_source_order = 0
        if row.formal_boundary:
            for side in SIDE_ORDER:
                self._close_provisionals(row.frame_idx, side, "match_boundary")
        attacks = {item.side: item for item in row.attacks}
        self._validate_row_triggers(row, attacks)
        for provisional in row.provisional_updates:
            self._add_provisional(row, provisional)
        for side in SIDE_ORDER:
            if side in attacks:
                self._finalize_attack(row, attacks[side])
        for side in SIDE_ORDER:
            if row.deltas[f"boundary_resets_{side}"]:
                self._expire_at_boundary(row, side)
        for recipient in SIDE_ORDER:
            amount = row.deltas[f"dropped_uncapped_{recipient}"]
            if amount:
                self._complete_fall(row, recipient, amount)

    @staticmethod
    def _validate_row_triggers(
        row: AccountingRow, attacks: Mapping[str, AttackFinalization],
    ) -> None:
        for side in SIDE_ORDER:
            if row.deltas[f"offset_uncapped_{side}"] and side not in attacks:
                raise EventAccountingAdapterError("相殺量に対応する確定攻撃がありません")
            if (row.deltas[f"boundary_wiped_uncapped_{side}"]
                    and not row.deltas[f"boundary_resets_{side}"]):
                raise EventAccountingAdapterError("境界失効量に対応する境界リセットがありません")

    def _finalize_attack(self, row: AccountingRow, attack: AttackFinalization) -> None:
        side, generated = attack.side, attack.generated_amount
        canceled = row.deltas[f"offset_uncapped_{side}"]
        sent = generated - canceled
        if sent < 0:
            raise EventAccountingAdapterError("相殺量が生成量を超えています")
        related = tuple(self.open_provisional_keys[side])
        self.open_provisional_keys[side].clear()
        resolver_key = (side, attack.resolver_chain_ordinal)
        if related and resolver_key not in related:
            self._close_provisional_keys(
                row.frame_idx, related, "resolver_mismatch_before_attack",
            )
            related = ()
        attack_id = self._final_attack_id(side, attack.side_ordinal)
        self.events.append(self._attack_event(row, attack, attack_id, related, sent))
        if canceled:
            allocations = self._allocate(side, canceled)
            self.events.append(self._cancel_event(
                row, side, canceled, sent, attack_id, allocations,
            ))
        if sent:
            self._send(row, side, sent, attack_id)

    def _add_provisional(
        self, row: AccountingRow, provisional: ProvisionalAttack,
    ) -> None:
        chain = provisional.resolver_chain_ordinal
        token = (
            f"{self.build_id}:provisional-{provisional.side}-{chain:06d}-"
            f"{provisional.update_ordinal:06d}"
        )
        key = (provisional.side, chain)
        if key not in self.provisional_tokens:
            self.open_provisional_keys[provisional.side].append(key)
        self.provisional_tokens.setdefault(key, []).append(
            (token, provisional.provisional_generated_amount, provisional.effective_rate)
        )
        event = self._row_event(row, "attack_provisional_updated", provisional.side)
        event["assertion"] = {"state": "provisional", "value_form": "exact"}
        event["relations"].update({
            "attack_id": self._provisional_attack_id(provisional.side, chain),
            "chain_id": self._chain_id(provisional.side, chain),
        })
        event["payload"] = {
            "provisional_observation_id": token,
            "update_ordinal": provisional.update_ordinal,
            "chain_count": provisional.chain_count,
            "provisional_score": provisional.provisional_score,
            "provisional_generated_amount": provisional.provisional_generated_amount,
            "effective_rate": provisional.effective_rate,
            "mechanism": provisional.mechanism,
        }
        event["missing_information"] = [
            "leftover_not_applied_to_provisional",
            "all_clear_bonus_not_applied_to_provisional",
        ]
        self.events.append(event)

    def _send(self, row: AccountingRow, side: str, amount: int, attack_id: str) -> None:
        recipient = _other(side)
        lot_id = f"{self.build_id}:lot-{recipient}-{len(self.lots) + 1:06d}"
        self.lots.append(_Lot(lot_id, side, recipient, amount, row.frame_idx))
        event = self._row_event(row, "garbage_sent", side)
        event["relations"].update({"attack_id": attack_id, "garbage_lot_ids": [lot_id]})
        event["payload"] = {
            "sent_amount": amount, "recipient": recipient,
            "garbage_lot_id": lot_id,
            "derivation_state": "accounting_model",
        }
        _mark_modeled(event, "recipient_pending_board_not_observed")
        self.events.append(event)

    def _complete_fall(self, row: AccountingRow, recipient: str, amount: int) -> None:
        reservation_id = self._next_fall_reservation_id(recipient)
        older, older_amount = self._allocate_partial(
            recipient, amount, created_before=row.frame_idx,
        )
        if older_amount:
            self._append_completed_fall(
                row, recipient, older_amount, reservation_id, older,
            )
        unresolved = amount - older_amount
        if unresolved <= 0:
            return
        current, current_amount = self._allocate_partial(
            recipient, unresolved, created_at=row.frame_idx,
        )
        if current_amount != unresolved:
            missing = unresolved - current_amount
            raise EventAccountingAdapterError(
                f"{recipient}向け会計量{amount}のうち{missing}を送付元へ帰属できません"
            )
        self._append_same_frame_ambiguity(
            row, recipient, unresolved, reservation_id, current,
        )

    def _next_fall_reservation_id(self, recipient: str) -> str:
        self.fall_ordinals[recipient] += 1
        return (
            f"{self.build_id}:fall-{recipient}-{self.fall_ordinals[recipient]:06d}"
        )

    def _append_completed_fall(
        self, row: AccountingRow, recipient: str, amount: int,
        reservation_id: str, allocations: list[dict[str, Any]],
    ) -> None:
        event = self._row_event(row, "garbage_fall_completed", recipient)
        event["relations"].update({
            "fall_reservation_id": reservation_id,
            "garbage_lot_ids": [item["garbage_lot_id"] for item in allocations],
        })
        event["payload"] = {
            "fall_reservation_id": reservation_id,
            "modeled_landed_amount": amount,
            "lot_allocations": allocations,
            "board_difference_state": "unobserved",
            "derivation_state": "placement_drain_rule",
        }
        _mark_modeled(
            event, "actual_landed_amount_not_observed",
            "uncompleted_amount_not_observed", "landing_board_difference_not_observed",
        )
        self.events.append(event)

    def _append_same_frame_ambiguity(
        self, row: AccountingRow, recipient: str, amount: int,
        drain_id: str, allocations: list[dict[str, Any]],
    ) -> None:
        event = self._row_event(
            row, "garbage_same_frame_order_ambiguous", recipient,
        )
        event["relations"].update({
            "counter_drain_id": drain_id,
            "garbage_lot_ids": [item["garbage_lot_id"] for item in allocations],
        })
        event["payload"] = {
            "counter_drain_id": drain_id,
            "ambiguous_amount": amount,
            "lot_allocations": allocations,
            "counter_source": "dropped_uncapped",
            "ordering_state": "same_frame_send_or_drain_order_unresolved",
            "model_input_allowed": False,
        }
        _mark_modeled(
            event, "exact_send_or_drain_order_not_observed",
            "actual_landing_not_established",
        )
        self.events.append(event)

    def _expire_at_boundary(self, row: AccountingRow, recipient: str) -> None:
        amount = row.deltas[f"boundary_wiped_uncapped_{recipient}"]
        if amount <= 0:
            return
        allocations = self._allocate(recipient, amount)
        current_lots = {
            lot.lot_id for lot in self.lots if lot.created_frame == row.frame_idx
        }
        uses_current_lot = any(
            item["garbage_lot_id"] in current_lots for item in allocations
        )
        event = self._row_event(row, "garbage_expired_at_boundary", recipient)
        event["relations"]["garbage_lot_ids"] = [
            item["garbage_lot_id"] for item in allocations
        ]
        event["payload"] = {
            "expired_amount": amount, "lot_allocations": allocations,
            "derivation_state": "boundary_pending_reset_model",
            "within_window_order_state": (
                "inferred_send_before_reset" if uses_current_lot
                else "no_same_window_lot_consumed"
            ),
        }
        missing = ["boundary_board_difference_not_observed"]
        if uses_current_lot:
            missing.append("exact_send_reset_order_not_observed")
        _mark_modeled(event, *missing)
        self.events.append(event)

    def _allocate(
        self, recipient: str, amount: int, *, landing_frame: int | None = None,
    ) -> list[dict[str, Any]]:
        allocations, allocated = self._allocate_partial(
            recipient, amount, created_before=landing_frame,
        )
        if allocated != amount:
            remaining = amount - allocated
            raise EventAccountingAdapterError(
                f"{recipient}向け会計量{amount}のうち{remaining}を送付元へ帰属できません"
            )
        return allocations

    def _allocate_partial(
        self, recipient: str, amount: int, *,
        created_before: int | None = None, created_at: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        remaining = amount
        allocations: list[dict[str, Any]] = []
        for lot in self.lots:
            if lot.recipient != recipient or lot.remaining <= 0:
                continue
            if created_before is not None and lot.created_frame >= created_before:
                continue
            if created_at is not None and lot.created_frame != created_at:
                continue
            used = min(remaining, lot.remaining)
            lot.remaining -= used
            remaining -= used
            allocations.append({"garbage_lot_id": lot.lot_id, "amount": used})
            if remaining == 0:
                break
        return allocations, amount - remaining

    def _attack_event(
        self, row: AccountingRow, attack: AttackFinalization,
        attack_id: str, related: tuple[tuple[str, int], ...], sent: int,
    ) -> dict[str, Any]:
        event = self._row_event(row, "attack_finalized", attack.side)
        chain_ids = [self._chain_id(side, ordinal) for side, ordinal in related]
        resolver_chain_id = self._chain_id(
            attack.side, attack.resolver_chain_ordinal,
        )
        event["relations"]["attack_id"] = attack_id
        event["relations"]["observed_chain_ids"] = chain_ids
        event["relations"]["resolver_chain_id_diagnostic"] = resolver_chain_id
        if len(chain_ids) == 1:
            event["relations"]["chain_id"] = chain_ids[0]
        rate_state = self._set_revision_relation(event, attack, related)
        relation_state = _chain_relation_state(len(chain_ids))
        event["payload"] = {
            "attack_id": attack_id, "observed_chain_ids": chain_ids,
            "chain_relation_state": relation_state,
            "revision_rate_relation_state": rate_state,
            "generated_amount": attack.generated_amount,
            "chain_total_score": attack.chain_total_score,
            "effective_rate": attack.effective_rate,
            "all_clear_bonus_state": "unobserved",
            "leftover_before": attack.leftover_before,
            "leftover_after": attack.leftover_after,
            "conversion_validation_state": "validated_at_sidecar_ingestion",
            "distribution_state": "modeled_in_followup_events",
        }
        event["missing_information"] = [
            "chain_start_not_yet_wired", "all_clear_bonus_not_separately_observed",
        ]
        if chain_ids:
            event["checks"].append(_membership_check(
                "resolver_chain_is_observed_candidate", resolver_chain_id, chain_ids,
            ))
        return event

    def _set_revision_relation(
        self, event: dict[str, Any], attack: AttackFinalization,
        related: tuple[tuple[str, int], ...],
    ) -> str:
        targets = [
            target for key in related
            for target in self.provisional_tokens.pop(key, [])
        ]
        if not targets:
            return "no_provisional_observation"
        rates = {rate for _, _, rate in targets}
        action = (
            "confirms" if len(related) == 1
            and targets[-1][1] == attack.generated_amount
            and rates == {attack.effective_rate}
            and attack.leftover_before == 0 else "corrects"
        )
        event["relations"]["revision"] = {
            "action": action,
            "target_event_ids": [],
            "target_observation_ids": [token for token, _, _ in targets],
        }
        return (
            "same_effective_rate" if rates == {attack.effective_rate}
            else "effective_rate_changed_before_finalization"
        )

    def _cancel_event(
        self, row: AccountingRow, side: str, canceled: int, sent: int,
        attack_id: str, allocations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        event = self._row_event(row, "garbage_cancelled", side)
        event["relations"].update({
            "attack_id": attack_id,
            "garbage_lot_ids": [item["garbage_lot_id"] for item in allocations],
        })
        event["payload"] = {
            "cancelled_amount": canceled, "sent_after_cancel_amount": sent,
            "lot_allocations": allocations,
            "derivation_state": "accounting_model",
        }
        _mark_modeled(event, "visible_pending_icons_not_independently_observed")
        return event

    def _close_provisionals(self, frame: int, side: str, reason: str) -> None:
        """権威値へ結び付かなかった途中観測を黙って捨てずに閉じる。"""
        keys = tuple(self.open_provisional_keys[side])
        self.open_provisional_keys[side].clear()
        self._close_provisional_keys(frame, keys, reason)

    def _close_provisional_keys(
        self, frame: int, keys: Sequence[tuple[str, int]], reason: str,
    ) -> None:
        for key in keys:
            targets = self.provisional_tokens.pop(key, [])
            if targets:
                self.events.append(self._unfinalized_event(frame, key, targets, reason))

    def _unfinalized_event(
        self, frame: int, key: tuple[str, int],
        targets: list[tuple[str, int, int]], reason: str,
    ) -> dict[str, Any]:
        side, chain = key
        event = self._base_event(frame, "attack_provisional_unfinalized", side)
        event["assertion"] = {"state": "provisional", "value_form": "exact"}
        event["relations"].update({
            "attack_id": self._provisional_attack_id(side, chain),
            "chain_id": self._chain_id(side, chain),
            "unfinalized_observation_ids": [token for token, _, _ in targets],
        })
        event["payload"] = {
            "close_reason": reason,
            "provisional_update_count": len(targets),
            "last_provisional_generated_amount": targets[-1][1],
            "last_provisional_effective_rate": targets[-1][2],
        }
        event["missing_information"] = ["authoritative_score_finalization_not_observed"]
        return event

    def _row_event(
        self, row: AccountingRow, event_type: str, side: str,
    ) -> dict[str, Any]:
        return self._base_event(
            row.frame_idx, event_type, side,
            earliest_frame=row.observation_window_start_frame,
        )

    def _base_event(
        self, frame: int, event_type: str, side: str,
        *, earliest_frame: int | None = None,
    ) -> dict[str, Any]:
        milliseconds = self.time_base.frame_to_ms(frame)
        earliest = frame if earliest_frame is None else earliest_frame
        self.accounting_source_order += 1
        return {
            "record_kind": "event", "schema_version": SCHEMA_VERSION,
            "source_video_id": self.source_video_id, "build_id": self.build_id,
            "attempt_id": self.attempt_id, "seq": 0, "event_id": f"{self.build_id}:0",
            "availability_batch_id": f"{self.build_id}:unsequenced",
            "batch_index": 0, "batch_size": 1, "event_type": event_type, "side": side,
            "timing": _interval_timing(
                earliest, frame, self.time_base.frame_to_ms(earliest), milliseconds,
            ),
            "assertion": {"state": "confirmed", "value_form": "exact"},
            "evidence": [_accounting_evidence(frame, event_type)], "checks": [],
            "missing_information": [],
            "relations": {
                "revision": {"action": "none", "target_event_ids": []},
                "accounting_source_order": self.accounting_source_order,
            },
            "heavy_evidence_refs": [], "payload": {},
        }

    def _provisional_attack_id(self, side: str, chain_ordinal: int) -> str:
        return f"{self.build_id}:provisional-attack-{side}-{chain_ordinal:06d}"

    def _final_attack_id(self, side: str, side_ordinal: int) -> str:
        return f"{self.build_id}:final-attack-{side}-{side_ordinal:06d}"

    def _chain_id(self, side: str, chain_ordinal: int) -> str:
        return f"{self.build_id}:chain-{side}-{chain_ordinal:06d}"


def load_event_accounting_sidecar(path: Path) -> EventAccountingSidecar:
    return parse_event_accounting_sidecar_bytes(path.read_bytes())


def parse_event_accounting_sidecar_bytes(payload: bytes) -> EventAccountingSidecar:
    """同じバイト列を厳格に解釈し、保存則と累積値を再検査する。"""
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EventAccountingAdapterError("会計サイドカーがUTF-8 JSONではありません") from error
    if not isinstance(value, dict):
        raise EventAccountingAdapterError("会計サイドカーはJSONオブジェクトが必要です")
    return _parse_sidecar(value)


def build_event_accounting_events(
    sidecar: EventAccountingSidecar, *, source_video_id: str, build_id: str,
    attempt_id: str, time_base: VideoTimeBase,
) -> tuple[dict[str, Any], ...]:
    """確定会計をID付き出来事へ変換する。clamp損失は安全に変換できず拒否する。"""
    if sidecar.final_gross["clamp_loss_p1"] or sidecar.final_gross["clamp_loss_p2"]:
        raise EventAccountingAdapterError("sanity clamp損失がある会計は採用できません")
    return _EventBuilder(
        sidecar, source_video_id, build_id, attempt_id, time_base,
    ).build()


def _parse_sidecar(value: Mapping[str, Any]) -> EventAccountingSidecar:
    if value.get("schema_version") != ACCOUNTING_SIDECAR_VERSION:
        raise EventAccountingAdapterError("会計サイドカーの仕様版が不正です")
    if value.get("observer_version") != ACCOUNTING_OBSERVER_VERSION:
        raise EventAccountingAdapterError("会計観測器の版が不正です")
    start = _nonnegative_int(value.get("processing_start_frame"), "処理開始")
    end = _nonnegative_int(value.get("processing_end_frame_exclusive"), "処理終了")
    observed = _nonnegative_int(value.get("observed_frame_count"), "観測フレーム数")
    inspected = _nonnegative_int(value.get("inspected_side_count"), "検査side数")
    if end <= start or observed <= 0 or inspected != observed * 2:
        raise EventAccountingAdapterError("処理範囲または検査母数が不正です")
    initial = _counter_mapping(value.get("initial_gross_counters"))
    final = _counter_mapping(value.get("final_gross_counters"))
    if any(initial.values()):
        raise EventAccountingAdapterError("会計観測はゼロ初期値から開始する必要があります")
    rows = _parse_rows(value.get("rows"), start, end)
    pending = _pending_mapping(value.get("final_pending_uncapped"), "最終pending")
    sidecar = EventAccountingSidecar(
        start, end, observed, inspected, initial, final, pending, rows,
    )
    _validate_sidecar_totals(sidecar, value.get("nonzero_row_count"))
    return sidecar


def _parse_rows(value: Any, start: int, end: int) -> tuple[AccountingRow, ...]:
    if not isinstance(value, list):
        raise EventAccountingAdapterError("会計行は配列でなければなりません")
    rows_list: list[AccountingRow] = []
    earliest = start
    for item in value:
        row = _parse_row(item, earliest)
        rows_list.append(row)
        earliest = row.frame_idx + 1
    rows = tuple(rows_list)
    frames = [row.frame_idx for row in rows]
    if any(not start <= frame < end for frame in frames):
        raise EventAccountingAdapterError("会計行が処理範囲外です")
    if frames != sorted(set(frames)):
        raise EventAccountingAdapterError("会計行のフレーム順または一意性が不正です")
    return rows


def _parse_row(value: Any, earliest_frame: int) -> AccountingRow:
    if not isinstance(value, dict):
        raise EventAccountingAdapterError("会計行はオブジェクトでなければなりません")
    before = _pending_mapping(value.get("pending_before"), "変化前pending")
    after = _pending_mapping(value.get("pending_after"), "変化後pending")
    deltas = _counter_mapping(value.get("deltas"))
    attacks = _parse_attacks(value.get("attack_finalizations"))
    provisionals = _parse_provisionals(value.get("provisional_updates", []))
    residual = _pending_mapping(value.get("conservation_residual"), "保存則残差", signed=True)
    formal_boundary = _formal_boundary_value(value, deltas)
    if residual != (0, 0):
        raise EventAccountingAdapterError("保存則残差が0ではありません")
    _validate_row_equation(before, after, deltas)
    return AccountingRow(
        _nonnegative_int(value.get("frame_idx"), "frame_idx"),
        before, after, deltas, attacks, provisionals, residual, earliest_frame,
        formal_boundary,
    )


def _formal_boundary_value(
    value: Mapping[str, Any], deltas: Mapping[str, int],
) -> bool:
    raw = value.get("formal_boundary")
    if raw is None:
        return any(deltas[f"boundary_resets_{side}"] > 0 for side in SIDE_ORDER)
    if not isinstance(raw, bool):
        raise EventAccountingAdapterError("正式境界フラグが真偽値ではありません")
    return raw


def _parse_attacks(value: Any) -> tuple[AttackFinalization, ...]:
    if not isinstance(value, list):
        raise EventAccountingAdapterError("確定攻撃は配列でなければなりません")
    attacks = tuple(_parse_attack(item) for item in value)
    sides = [item.side for item in attacks]
    if sides != [side for side in SIDE_ORDER if side in sides] or len(set(sides)) != len(sides):
        raise EventAccountingAdapterError("確定攻撃のside順または一意性が不正です")
    return attacks


def _parse_attack(value: Any) -> AttackFinalization:
    if not isinstance(value, dict) or value.get("side") not in SIDE_ORDER:
        raise EventAccountingAdapterError("確定攻撃のsideが不正です")
    fields = ("side_ordinal", "chain_total_score", "generated_amount",
              "effective_rate", "leftover_before", "leftover_after",
              "resolver_chain_ordinal")
    numbers = [_nonnegative_int(value.get(field), field) for field in fields]
    attack = AttackFinalization(str(value["side"]), *numbers)
    if (attack.side_ordinal <= 0 or attack.effective_rate <= 0
            or attack.resolver_chain_ordinal <= 0):
        raise EventAccountingAdapterError("確定攻撃の通番、連鎖IDまたは換算率が不正です")
    if (attack.generated_amount * attack.effective_rate + attack.leftover_after
            != attack.chain_total_score + attack.leftover_before):
        raise EventAccountingAdapterError("得点から攻撃量への換算式が一致しません")
    if attack.leftover_after >= attack.effective_rate:
        raise EventAccountingAdapterError("確定攻撃の換算余りが有効レート以上です")
    return attack


def _parse_provisionals(value: Any) -> tuple[ProvisionalAttack, ...]:
    if not isinstance(value, list):
        raise EventAccountingAdapterError("暫定攻撃更新は配列でなければなりません")
    result = tuple(_parse_provisional(item) for item in value)
    keys = [(item.side, item.resolver_chain_ordinal) for item in result]
    if keys != sorted(keys, key=lambda item: (SIDE_ORDER.index(item[0]), item[1])):
        raise EventAccountingAdapterError("暫定攻撃更新の順序が不正です")
    return result


def _parse_provisional(value: Any) -> ProvisionalAttack:
    if not isinstance(value, dict) or value.get("side") not in SIDE_ORDER:
        raise EventAccountingAdapterError("暫定攻撃更新のsideが不正です")
    fields = (
        "resolver_chain_ordinal", "update_ordinal", "chain_count",
        "provisional_score", "provisional_generated_amount", "effective_rate",
    )
    numbers = [_nonnegative_int(value.get(field), field) for field in fields]
    mechanism = str(value.get("mechanism", ""))
    item = ProvisionalAttack(str(value["side"]), *numbers, mechanism)
    if (item.resolver_chain_ordinal <= 0 or item.update_ordinal <= 0
            or item.effective_rate <= 0
            or mechanism not in {CHAIN_MECHANISM_FORMULA, CHAIN_MECHANISM_FORMULA_READ}):
        raise EventAccountingAdapterError("暫定攻撃更新のID、換算率または経路が不正です")
    if item.provisional_generated_amount != item.provisional_score // item.effective_rate:
        raise EventAccountingAdapterError("暫定攻撃量の換算式が一致しません")
    return item


def _validate_row_equation(
    before: tuple[int, int], after: tuple[int, int], deltas: Mapping[str, int],
) -> None:
    expected_p1 = (
        deltas["generated_p2"] - deltas["offset_uncapped_p2"]
        - deltas["offset_uncapped_p1"] - deltas["dropped_uncapped_p1"]
        - deltas["boundary_wiped_uncapped_p1"] - deltas["clamp_loss_p1"]
    )
    expected_p2 = (
        deltas["generated_p1"] - deltas["offset_uncapped_p1"]
        - deltas["offset_uncapped_p2"] - deltas["dropped_uncapped_p2"]
        - deltas["boundary_wiped_uncapped_p2"] - deltas["clamp_loss_p2"]
    )
    if (after[0] - before[0], after[1] - before[1]) != (expected_p1, expected_p2):
        raise EventAccountingAdapterError("会計行の保存式を再計算できません")


def _validate_sidecar_totals(sidecar: EventAccountingSidecar, row_count: Any) -> None:
    if _nonnegative_int(row_count, "非ゼロ行数") != len(sidecar.rows):
        raise EventAccountingAdapterError("非ゼロ行数が一致しません")
    cumulative = dict(sidecar.initial_gross)
    pending = (0, 0)
    ordinals = {"p1": 0, "p2": 0}
    leftovers = {"p1": {0}, "p2": {0}}
    provisional_state: dict[tuple[str, int], ProvisionalAttack] = {}
    resolved_chains: set[tuple[str, int]] = set()
    for row in sidecar.rows:
        if row.pending_before != pending:
            raise EventAccountingAdapterError("会計行のpendingが連続していません")
        for key in COUNTER_NAMES:
            cumulative[key] += row.deltas[key]
        _validate_provisional_row(row, provisional_state, resolved_chains)
        _validate_attack_row(row, ordinals, resolved_chains, leftovers)
        pending = row.pending_after
    if cumulative != sidecar.final_gross or pending != sidecar.final_pending:
        raise EventAccountingAdapterError("会計行の累積終値が説明値と一致しません")


def _validate_attack_row(
    row: AccountingRow, ordinals: dict[str, int],
    resolved_chains: set[tuple[str, int]], leftovers: dict[str, set[int]],
) -> None:
    by_side = {item.side: item for item in row.attacks}
    for side in SIDE_ORDER:
        generated = row.deltas[f"generated_{side}"]
        attack = by_side.get(side)
        if attack is None and generated > 0:
            raise EventAccountingAdapterError("生成量と確定攻撃行が対応しません")
        reset = row.deltas[f"boundary_resets_{side}"] > 0
        if attack is None:
            if reset:
                leftovers[side] = {0}
            continue
        ordinals[side] += 1
        if attack.side_ordinal != ordinals[side] or attack.generated_amount != generated:
            raise EventAccountingAdapterError("確定攻撃通番または生成量が不正です")
        allowed_before = leftovers[side] | ({0} if reset else set())
        if attack.leftover_before not in allowed_before:
            raise EventAccountingAdapterError("確定攻撃の換算余りが連続していません")
        next_values = {attack.leftover_after} if not reset else set()
        if reset and attack.leftover_before == 0:
            next_values.add(attack.leftover_after)
        if reset and attack.leftover_before in leftovers[side]:
            next_values.add(0)
        leftovers[side] = next_values
        key = (side, attack.resolver_chain_ordinal)
        if key in resolved_chains:
            raise EventAccountingAdapterError("同じ物理連鎖IDが複数回確定しています")
        resolved_chains.add(key)


def _validate_provisional_row(
    row: AccountingRow,
    state: dict[tuple[str, int], ProvisionalAttack],
    resolved_chains: set[tuple[str, int]],
) -> None:
    for item in row.provisional_updates:
        key = (item.side, item.resolver_chain_ordinal)
        if key in resolved_chains:
            raise EventAccountingAdapterError("確定後の物理連鎖へ暫定更新があります")
        previous = state.get(key)
        if previous is None:
            if item.update_ordinal != 1:
                raise EventAccountingAdapterError("暫定攻撃更新の初回通番が1ではありません")
        elif (item.update_ordinal != previous.update_ordinal + 1
              or item.chain_count < previous.chain_count
              or item.provisional_score < previous.provisional_score
              or item.provisional_generated_amount < previous.provisional_generated_amount):
            raise EventAccountingAdapterError("暫定攻撃更新が単調ではありません")
        state[key] = item


def _counter_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(COUNTER_NAMES):
        raise EventAccountingAdapterError("grossカウンタの列が一致しません")
    return {name: _nonnegative_int(value[name], name) for name in COUNTER_NAMES}


def _pending_mapping(value: Any, name: str, signed: bool = False) -> tuple[int, int]:
    if not isinstance(value, dict) or set(value) != {"p1", "p2"}:
        raise EventAccountingAdapterError(f"{name}の列が不正です")
    parser = _integer if signed else _nonnegative_int
    return parser(value["p1"], f"{name}.p1"), parser(value["p2"], f"{name}.p2")


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EventAccountingAdapterError(f"{name}は整数でなければなりません")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise EventAccountingAdapterError(f"{name}は0以上でなければなりません")
    return result


def _other(side: str) -> str:
    return "p2" if side == "p1" else "p1"


def _chain_relation_state(count: int) -> str:
    if count == 0:
        return "no_provisional_chain_observed"
    if count == 1:
        return "single_observed_chain"
    return "ambiguous_multiple_observed_chains"


def _interval_timing(
    earliest_frame: int, latest_frame: int,
    earliest_ms: int, latest_ms: int,
) -> dict[str, int]:
    return {
        "occurred_earliest_frame": earliest_frame,
        "occurred_earliest_ms": earliest_ms,
        "occurred_latest_frame": latest_frame,
        "occurred_latest_ms": latest_ms,
        "available_frame": latest_frame,
        "available_ms": latest_ms,
    }


def _accounting_evidence(frame: int, event_type: str) -> dict[str, Any]:
    evidence_type = {
        "attack_provisional_updated": "chain_formula_observation",
        "attack_provisional_unfinalized": "processing_boundary_diagnostic",
        "attack_finalized": "score_ocr_attack_finalization",
        "garbage_cancelled": "accounting_model_derivation",
        "garbage_sent": "accounting_model_derivation",
        "garbage_fall_completed": "tsumo_settled_drain_rule",
        "garbage_same_frame_order_ambiguous": "same_frame_accounting_order_ambiguity",
        "garbage_expired_at_boundary": "match_boundary_pending_reset",
    }.get(event_type, "accounting_model_derivation")
    return {
        "evidence_type": evidence_type,
        "method_id": "OjamaAccountingTracker",
        "method_version": ACCOUNTING_ADAPTER_VERSION,
        "from_frame": frame, "to_frame": frame,
    }


def _mark_modeled(event: dict[str, Any], *missing: str) -> None:
    """盤面で独立観測していない会計派生値を確定値へ昇格させない。"""
    event["assertion"] = {"state": "provisional", "value_form": "exact"}
    event["missing_information"] = list(missing)


def _membership_check(
    check_id: str, observed: str, candidates: Sequence[str],
) -> dict[str, Any]:
    matched = observed in candidates
    return {
        "check_id": check_id, "check_version": ACCOUNTING_ADAPTER_VERSION,
        "result": "pass" if matched else "fail", "observed": observed,
        "expected": list(candidates), "difference": 0 if matched else 1,
        "tolerance": 0, "reason_codes": [] if matched else ["resolver_mismatch"],
    }


__all__ = [
    "ACCOUNTING_ADAPTER_VERSION",
    "AccountingRow",
    "AttackFinalization",
    "EventAccountingAdapterError",
    "EventAccountingSidecar",
    "build_event_accounting_events",
    "load_event_accounting_sidecar",
    "parse_event_accounting_sidecar_bytes",
]
