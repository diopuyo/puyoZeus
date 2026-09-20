"""会計出来事を含む完了試行の保存則・訂正関係を独立に再検査する。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.event_accounting_adapter_v1 import (
    EventAccountingAdapterError,
    EventAccountingSidecar,
    parse_event_accounting_sidecar_bytes,
)
from src.event_pilot_analysis_v1 import load_completed_run_events


EXPECTED_CONFIG_VERSION = "event-snapshot-recognition-config/4"
ACCOUNTING_EVENT_TYPES = frozenset({
    "attack_provisional_updated", "attack_provisional_unfinalized",
    "attack_finalized", "garbage_cancelled",
    "garbage_sent", "garbage_fall_completed",
    "garbage_same_frame_order_ambiguous", "garbage_expired_at_boundary",
})
SETTLEMENT_EVENT_TYPES = frozenset({
    "garbage_cancelled", "garbage_fall_completed",
    "garbage_same_frame_order_ambiguous", "garbage_expired_at_boundary",
})
GROSS_EVENT_SPECS = (
    ("generated", (("attack_finalized", "generated_amount"),)),
    ("offset_uncapped", (("garbage_cancelled", "cancelled_amount"),)),
    ("dropped_uncapped", (
        ("garbage_fall_completed", "modeled_landed_amount"),
        ("garbage_same_frame_order_ambiguous", "ambiguous_amount"),
    )),
    ("boundary_wiped_uncapped", (
        ("garbage_expired_at_boundary", "expired_amount"),
    )),
)


class AccountingPilotValidationError(ValueError):
    """完了試行の会計保存則または訂正関係が不正な場合の例外。"""


def validate_accounting_pilot_run(run_dir: Path) -> dict[str, Any]:
    """一試行を読み直し、攻撃・量ID・訂正・検査母数を再計算する。"""
    manifest, all_events = load_completed_run_events(run_dir)
    return validate_accounting_pilot_loaded_run(run_dir, manifest, all_events)


def validate_accounting_pilot_loaded_run(
    run_dir: Path, manifest: dict[str, Any], all_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """直前に完成検査した共有出来事列から会計を再計算する。"""
    config = _load_object(run_dir / "recognition-config.json")
    sidecar, sidecar_sha, residual_audit = _embedded_accounting(config)
    events = [event for event in all_events if event["event_type"] in ACCOUNTING_EVENT_TYPES]
    checked, unchecked = _validate_event_checks(events)
    _validate_attack_conservation(events)
    gross_totals = _validate_event_gross_totals(events, sidecar)
    outstanding = _validate_lot_conservation(events)
    expected_pending = {"p1": sidecar.final_pending[0], "p2": sidecar.final_pending[1]}
    if outstanding != expected_pending:
        raise AccountingPilotValidationError("量IDの残量が最終pendingと一致しません")
    revisions = _validate_revisions(events)
    _validate_provisional_closure(events)
    counts = Counter(str(event["event_type"]) for event in events)
    relation_counts = _finalized_relation_counts(events)
    frame_span = sidecar.processing_end_frame_exclusive - sidecar.processing_start_frame
    sampling = _sampling_metadata(config, frame_span, sidecar.observed_frame_count)
    return {
        "run_dir": str(run_dir.resolve()),
        "build_id": manifest["build_id"], "attempt_id": manifest["attempt_id"],
        "accounting_sidecar_sha256": sidecar_sha,
        "accounting_event_count": len(events),
        "accounting_event_type_counts": dict(sorted(counts.items())),
        "inspected_side_count": sidecar.inspected_side_count,
        "sampled_frame_count": sidecar.observed_frame_count,
        "processing_frame_span": frame_span,
        "sampled_frame_coverage_ratio": sidecar.observed_frame_count / frame_span,
        **sampling,
        "nonzero_accounting_row_count": len(sidecar.rows),
        "conservation_residual_nonzero_count": residual_audit["nonzero_count"],
        "conservation_residual_observation_side_count": residual_audit["denominator"],
        "conservation_residual_materialized_side_count": residual_audit["materialized"],
        "conservation_residual_inferred_zero_side_count": residual_audit["inferred_zero"],
        "events_with_checks_count": checked,
        "events_without_checks_count": unchecked,
        "final_pending_uncapped": expected_pending,
        "lot_outstanding": outstanding,
        "gross_event_totals": gross_totals,
        "revision_confirms_count": revisions["confirms"],
        "revision_corrects_count": revisions["corrects"],
        "revision_target_count": revisions["targets"],
        "provisional_update_count": revisions["provisional_updates"],
        "unfinalized_provisional_chain_count": revisions["unfinalized_chains"],
        "unfinalized_provisional_target_count": revisions["unfinalized_targets"],
        "finalized_chain_relation_counts": relation_counts,
    }


def _finalized_relation_counts(events: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(event["payload"].get("chain_relation_state", "missing"))
        for event in events if event["event_type"] == "attack_finalized"
    )
    return dict(sorted(counts.items()))


def _embedded_accounting(
    config: dict[str, Any],
) -> tuple[EventAccountingSidecar, str, dict[str, int]]:
    if config.get("format_version") != EXPECTED_CONFIG_VERSION:
        raise AccountingPilotValidationError("認識設定の仕様版が会計観測対応ではありません")
    embedded = config.get("event_accounting_input")
    if not isinstance(embedded, dict) or embedded.get("format") != "embedded-utf8-json":
        raise AccountingPilotValidationError("会計サイドカーが自己完結していません")
    content, expected_sha = embedded.get("content_utf8"), embedded.get("sha256")
    if not isinstance(content, str) or not isinstance(expected_sha, str):
        raise AccountingPilotValidationError("埋込み会計サイドカーが不正です")
    payload = content.encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise AccountingPilotValidationError("埋込み会計サイドカーの要約が一致しません")
    try:
        raw_value = json.loads(content)
    except json.JSONDecodeError as error:
        raise AccountingPilotValidationError(
            "埋込み会計サイドカーがJSONではありません"
        ) from error
    audit = _raw_residual_audit(raw_value)
    try:
        sidecar = parse_event_accounting_sidecar_bytes(payload)
    except EventAccountingAdapterError as error:
        raise AccountingPilotValidationError(
            "会計サイドカーを拒否しました: 保存則残差="
            f"{audit['nonzero_count']}/{audit['denominator']}: {error}"
        ) from error
    return sidecar, expected_sha, audit


def _raw_residual_audit(value: Any) -> dict[str, int]:
    """採用ゲート前の生JSONから残差件数と全標本母数を数える。"""
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise AccountingPilotValidationError("残差監査対象の生JSONが不正です")
    observed = value.get("observed_frame_count")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
        raise AccountingPilotValidationError("残差監査の観測母数が不正です")
    residuals: list[int] = []
    for row in value["rows"]:
        residual = row.get("conservation_residual") if isinstance(row, dict) else None
        if not isinstance(residual, dict) or set(residual) != {"p1", "p2"}:
            raise AccountingPilotValidationError("残差監査の行が不正です")
        values = [residual[side] for side in ("p1", "p2")]
        if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
            raise AccountingPilotValidationError("残差監査値が整数ではありません")
        residuals.extend(values)
    if len(residuals) > observed * 2:
        raise AccountingPilotValidationError("残差監査行が観測母数を超えています")
    return {
        "nonzero_count": sum(int(item != 0) for item in residuals),
        "denominator": observed * 2,
        "materialized": len(residuals),
        "inferred_zero": observed * 2 - len(residuals),
    }


def _validate_event_checks(events: Sequence[dict[str, Any]]) -> tuple[int, int]:
    checked = 0
    for event in events:
        checks = event.get("checks", [])
        if any(check.get("result") != "pass" for check in checks):
            raise AccountingPilotValidationError("会計出来事に失敗検査があります")
        checked += int(bool(checks))
    return checked, len(events) - checked


def _validate_attack_conservation(events: Sequence[dict[str, Any]]) -> None:
    attacks = {
        str(event["relations"]["attack_id"]): int(event["payload"]["generated_amount"])
        for event in events if event["event_type"] == "attack_finalized"
    }
    if len(attacks) != sum(event["event_type"] == "attack_finalized" for event in events):
        raise AccountingPilotValidationError("攻撃IDが重複しています")
    distributed = {attack_id: 0 for attack_id in attacks}
    for event in events:
        if event["event_type"] == "garbage_cancelled":
            _add_attack_amount(distributed, event, "cancelled_amount")
        elif event["event_type"] == "garbage_sent":
            _add_attack_amount(distributed, event, "sent_amount")
    if distributed != attacks:
        raise AccountingPilotValidationError("生成量が相殺量と送付量に保存されていません")


def _validate_event_gross_totals(
    events: Sequence[dict[str, Any]], sidecar: EventAccountingSidecar,
) -> dict[str, dict[str, int]]:
    """出来事量を再合計し、sidecarの累積終値と突き合わせる。"""
    results: dict[str, dict[str, int]] = {}
    for counter_prefix, event_specs in GROSS_EVENT_SPECS:
        for side in ("p1", "p2"):
            observed = _gross_event_total(events, side, event_specs)
            expected = sidecar.final_gross[f"{counter_prefix}_{side}"]
            if observed != expected:
                raise AccountingPilotValidationError(
                    f"{counter_prefix}の出来事合計がgross終値と一致しません"
                )
            results[f"{counter_prefix}_{side}"] = {
                "event_total": observed,
                "gross_total": expected,
                "difference": observed - expected,
            }
    return results


def _gross_event_total(
    events: Sequence[dict[str, Any]], side: str,
    event_specs: Sequence[tuple[str, str]],
) -> int:
    return sum(
        int(event["payload"][payload_key])
        for event_type, payload_key in event_specs
        for event in events
        if event["event_type"] == event_type and event["side"] == side
    )


def _add_attack_amount(
    totals: dict[str, int], event: dict[str, Any], payload_key: str,
) -> None:
    attack_id = str(event["relations"].get("attack_id", ""))
    if attack_id not in totals:
        raise AccountingPilotValidationError("配分先の攻撃IDが存在しません")
    totals[attack_id] += int(event["payload"][payload_key])


def _validate_lot_conservation(events: Sequence[dict[str, Any]]) -> dict[str, int]:
    lots: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["event_type"] == "garbage_sent":
            _register_lot(lots, event)
        elif event["event_type"] in SETTLEMENT_EVENT_TYPES:
            _consume_allocations(lots, event)
    return {
        side: sum(item["remaining"] for item in lots.values() if item["recipient"] == side)
        for side in ("p1", "p2")
    }


def _register_lot(lots: dict[str, dict[str, Any]], event: dict[str, Any]) -> None:
    lot_id = str(event["payload"]["garbage_lot_id"])
    if lot_id in lots:
        raise AccountingPilotValidationError("おじゃま量IDが重複しています")
    lots[lot_id] = {
        "recipient": str(event["payload"]["recipient"]),
        "remaining": int(event["payload"]["sent_amount"]),
        "created_frame": int(event["timing"]["available_frame"]),
    }


def _consume_allocations(lots: dict[str, dict[str, Any]], event: dict[str, Any]) -> None:
    if event["event_type"] == "garbage_same_frame_order_ambiguous":
        _validate_same_frame_ambiguity(event)
    expected = _independent_fifo_allocations(lots, event)
    recorded = [
        {"garbage_lot_id": str(item["garbage_lot_id"]), "amount": int(item["amount"])}
        for item in event["payload"].get("lot_allocations", [])
    ]
    if recorded != expected:
        raise AccountingPilotValidationError("記録された量ID配分が独立FIFO再計算と違います")


def _independent_fifo_allocations(
    lots: dict[str, dict[str, Any]], event: dict[str, Any],
) -> list[dict[str, Any]]:
    if event["event_type"] == "garbage_same_frame_order_ambiguous":
        return _same_frame_fifo_allocations(lots, event)
    recipient = str(event["side"])
    remaining, result = _settlement_amount(event), []
    for lot_id, lot in lots.items():
        if lot["recipient"] != recipient or lot["remaining"] <= 0:
            continue
        _reject_same_frame_landing(event, lot)
        used = min(remaining, int(lot["remaining"]))
        lot["remaining"] -= used
        remaining -= used
        result.append({"garbage_lot_id": lot_id, "amount": used})
        if remaining == 0:
            break
    if remaining:
        raise AccountingPilotValidationError("独立FIFO再計算で決済元が不足しています")
    return result


def _same_frame_fifo_allocations(
    lots: dict[str, dict[str, Any]], event: dict[str, Any],
) -> list[dict[str, Any]]:
    recipient = str(event["side"])
    frame = int(event["timing"]["available_frame"])
    older = any(
        lot["recipient"] == recipient and lot["remaining"] > 0
        and lot["created_frame"] < frame for lot in lots.values()
    )
    if older:
        raise AccountingPilotValidationError("順序不明量より古い量IDが未決済です")
    remaining, result = _settlement_amount(event), []
    for lot_id, lot in lots.items():
        if (lot["recipient"] != recipient or lot["remaining"] <= 0
                or lot["created_frame"] != frame):
            continue
        used = min(remaining, int(lot["remaining"]))
        lot["remaining"] -= used
        remaining -= used
        result.append({"garbage_lot_id": lot_id, "amount": used})
        if remaining == 0:
            break
    if remaining:
        raise AccountingPilotValidationError("同一フレームの順序不明量IDが不足しています")
    return result


def _validate_same_frame_ambiguity(event: dict[str, Any]) -> None:
    payload = event["payload"]
    if payload.get("model_input_allowed") is not False:
        raise AccountingPilotValidationError("順序不明量が学習入力から隔離されていません")
    if payload.get("counter_source") != "dropped_uncapped":
        raise AccountingPilotValidationError("順序不明量の累積器由来が不正です")
    expected = "same_frame_send_or_drain_order_unresolved"
    if payload.get("ordering_state") != expected:
        raise AccountingPilotValidationError("順序不明量の状態が不正です")


def _reject_same_frame_landing(event: dict[str, Any], lot: dict[str, Any]) -> None:
    if (event["event_type"] == "garbage_fall_completed"
            and lot["created_frame"] >= int(event["timing"]["available_frame"])):
        raise AccountingPilotValidationError("同一フレーム新規送付を着地へ帰属しています")


def _settlement_amount(event: dict[str, Any]) -> int:
    key = {
        "garbage_cancelled": "cancelled_amount",
        "garbage_fall_completed": "modeled_landed_amount",
        "garbage_same_frame_order_ambiguous": "ambiguous_amount",
        "garbage_expired_at_boundary": "expired_amount",
    }[str(event["event_type"])]
    return int(event["payload"][key])


def _validate_revisions(events: Sequence[dict[str, Any]]) -> dict[str, int]:
    by_id = {str(event["event_id"]): event for event in events}
    counts = {"confirms": 0, "corrects": 0, "targets": 0,
              "provisional_updates": 0, "unfinalized_chains": 0,
              "unfinalized_targets": 0}
    for event in events:
        if event["event_type"] != "attack_finalized":
            continue
        revision = event["relations"]["revision"]
        action = str(revision["action"])
        if action == "none":
            continue
        if action not in {"confirms", "corrects"}:
            raise AccountingPilotValidationError("確定攻撃の訂正動作が不正です")
        targets = [by_id.get(str(target)) for target in revision["target_event_ids"]]
        _validate_revision_targets(event, targets, action)
        counts[action] += 1
        counts["targets"] += len(targets)
    counts["provisional_updates"] = sum(
        event["event_type"] == "attack_provisional_updated" for event in events
    )
    closures = [
        event for event in events
        if event["event_type"] == "attack_provisional_unfinalized"
    ]
    counts["unfinalized_chains"] = len(closures)
    counts["unfinalized_targets"] = sum(
        len(event["relations"].get("unfinalized_event_ids", [])) for event in closures
    )
    return counts


def _validate_revision_targets(
    finalized: dict[str, Any], targets: list[dict[str, Any] | None], action: str,
) -> None:
    if not targets or any(target is None for target in targets):
        raise AccountingPilotValidationError("訂正対象が存在しません")
    concrete = [target for target in targets if target is not None]
    observed_chain_ids = set(finalized["relations"].get("observed_chain_ids", []))
    if any(target["event_type"] != "attack_provisional_updated" for target in concrete):
        raise AccountingPilotValidationError("訂正対象が暫定攻撃ではありません")
    if any(target["relations"].get("chain_id") not in observed_chain_ids
           for target in concrete):
        raise AccountingPilotValidationError("訂正対象の連鎖候補が関係一覧にありません")
    if any(int(target["seq"]) >= int(finalized["seq"]) for target in concrete):
        raise AccountingPilotValidationError("訂正対象が確定行より後です")
    provisional = int(concrete[-1]["payload"]["provisional_generated_amount"])
    exact = int(finalized["payload"]["generated_amount"])
    target_rates = {
        int(target["payload"]["effective_rate"]) for target in concrete
    }
    exact_rate = int(finalized["payload"]["effective_rate"])
    rate_state = (
        "same_effective_rate" if target_rates == {exact_rate}
        else "effective_rate_changed_before_finalization"
    )
    if finalized["payload"].get("revision_rate_relation_state") != rate_state:
        raise AccountingPilotValidationError("暫定値の換算率関係を再計算できません")
    expected = (
        "confirms" if len(observed_chain_ids) == 1 and provisional == exact
        and int(finalized["payload"]["leftover_before"]) == 0
        and rate_state == "same_effective_rate"
        else "corrects"
    )
    if action != expected:
        raise AccountingPilotValidationError("暫定値と確定値に対する訂正動作が違います")


def _validate_provisional_closure(events: Sequence[dict[str, Any]]) -> None:
    provisional_ids = {
        str(event["event_id"]) for event in events
        if event["event_type"] == "attack_provisional_updated"
    }
    resolved: list[str] = []
    for event in events:
        if event["event_type"] == "attack_finalized":
            resolved.extend(event["relations"]["revision"].get("target_event_ids", []))
        elif event["event_type"] == "attack_provisional_unfinalized":
            resolved.extend(event["relations"].get("unfinalized_event_ids", []))
    if len(resolved) != len(set(resolved)) or set(resolved) != provisional_ids:
        raise AccountingPilotValidationError("暫定攻撃に重複または未処理の観測があります")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AccountingPilotValidationError("認識設定がJSONオブジェクトではありません")
    return value


def _sampling_metadata(
    config: dict[str, Any], frame_span: int, sampled_frames: int,
) -> dict[str, Any]:
    tokens = config.get("collection_tokens", [])
    if not isinstance(tokens, list) or any(not isinstance(item, str) for item in tokens):
        raise AccountingPilotValidationError("収集設定の引数列が不正です")
    requested = _option_after(tokens, "--sample-interval")
    return {
        "requested_sample_interval_sec": requested,
        "normalize_fps_30_enabled": "--normalize-fps-30" in tokens,
        "effective_decoded_frames_per_sample": frame_span / sampled_frames,
    }


def _option_after(tokens: list[str], option: str) -> str | None:
    if option not in tokens:
        return None
    index = tokens.index(option)
    if index + 1 >= len(tokens):
        raise AccountingPilotValidationError(f"{option}の値がありません")
    return tokens[index + 1]


__all__ = ["AccountingPilotValidationError", "validate_accounting_pilot_run"]
