"""出来事原本v1から当時版の学習用状態・正解・対応表を作る。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.event_source_v1 import (
    CommittedBatch,
    SemanticEventHasher,
    iter_committed_batches,
)
from src.event_causal_exchange_v1 import (
    CausalExchangeReplay,
    settled_pending_disagreement,
)


LEARNING_TABLE_VERSION = "event-learning-tables/v1"
RUN_COMPLETE_VERSION = "event-run-complete/1"
POSTHOC_EVENT_TYPES = frozenset({"winner_observed", "official_game_assignment"})
BOUNDARY_EVENT_TYPE = "match_boundary_evidence"
BOARD_EVENT_TYPE = "stable_board_observed"
ONLINE_EVENT_TYPES = frozenset({
    BOARD_EVENT_TYPE, BOUNDARY_EVENT_TYPE, "garbage_landing_board_compared",
    "garbage_landing_observation_incomplete", "attack_provisional_updated",
    "attack_provisional_unfinalized", "attack_finalized", "chain_started",
    "garbage_sent", "garbage_cancelled", "garbage_fall_completed",
    "garbage_expired_at_boundary", "garbage_same_frame_order_ambiguous",
    "all_clear_gained", "all_clear_consumed",
})
KNOWN_EVENT_TYPES = ONLINE_EVENT_TYPES | POSTHOC_EVENT_TYPES
SIDES = ("p1", "p2")
BOARD_CELL_COUNT = 13 * 6
HASH_CHUNK_BYTES = 1024 * 1024
MODEL_FORBIDDEN_STATE_KEYS = frozenset(
    {"winner_side", "p1_won", "won", "official_game_number", "game_key"}
)


class EventLearningTableError(ValueError):
    """学習表へ安全に変換できない場合の例外。"""


@dataclass(frozen=True, slots=True)
class LearningTables:
    """一つの元映像から得た三つの派生表。"""

    states: tuple[dict[str, Any], ...]
    labels: tuple[dict[str, Any], ...]
    event_links: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class _ReplayState:
    boards: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending: dict[str, int] = field(
        default_factory=lambda: {side: 0 for side in SIDES}
    )
    provisional: dict[str, dict[str, int] | None] = field(
        default_factory=lambda: {side: None for side in SIDES}
    )
    chain_active: dict[str, bool] = field(
        default_factory=lambda: {side: False for side in SIDES}
    )
    all_clear: dict[str, bool] = field(
        default_factory=lambda: {side: False for side in SIDES}
    )
    batch_ambiguous_amount: int = 0
    batch_boundary_expiry_gap: int = 0
    batch_disallowed_event_count: int = 0
    online_segment_index: int = 0
    causal_exchange: CausalExchangeReplay = field(default_factory=CausalExchangeReplay)

    def reset_for_boundary(self) -> None:
        self.online_segment_index += 1
        self.boards.clear()
        self.pending = {side: 0 for side in SIDES}
        self.provisional = {side: None for side in SIDES}
        self.chain_active = {side: False for side in SIDES}
        self.all_clear = {side: False for side in SIDES}
        self.batch_ambiguous_amount = 0
        self.batch_boundary_expiry_gap = 0
        self.batch_disallowed_event_count = 0
        self.causal_exchange.reset()


@dataclass(frozen=True, slots=True)
class _OfficialAssignment:
    event_id: str
    from_sequence: int
    to_sequence: int
    game_number: int
    winner_side: str
    winner_event_id: str


def build_learning_tables(
    batches: Iterable[CommittedBatch], *, fold: int, tier: str,
    source_group_id: str = "", context_quarantined_segments: Collection[int] = (),
    online_context_quarantine_starts: Mapping[int, int] | None = None,
) -> LearningTables:
    """一括更新を原子的に再生し、同じ状態集合へ正解を事後対応付けする。"""

    replay = _ReplayState()
    quarantined_segments = frozenset(int(value) for value in context_quarantined_segments)
    if any(value < 0 for value in quarantined_segments):
        raise EventLearningTableError("隔離する局所試合番号は0以上でなければなりません")
    online_starts = _validated_online_quarantine_starts(
        quarantined_segments, online_context_quarantine_starts,
    )
    states: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    assignment_events: list[Mapping[str, Any]] = []
    winner_events: dict[str, Mapping[str, Any]] = {}
    for batch in batches:
        _validate_event_allowlist(batch)
        _collect_posthoc_events(batch, assignment_events, winner_events)
        if _is_posthoc_only(batch):
            continue
        if _contains_boundary(batch):
            replay.reset_for_boundary()
            continue
        _apply_batch(replay, batch)
        if set(replay.boards) != set(SIDES):
            continue
        training_quarantined = replay.online_segment_index in quarantined_segments
        online_quarantined = _is_online_quarantined(
            replay.online_segment_index, int(batch.events[-1]["seq"]),
            training_quarantined, online_starts,
        )
        state = _state_row(
            replay, batch, fold=fold, tier=tier, source_group_id=source_group_id,
            context_quarantined=online_quarantined,
            training_context_quarantined=training_quarantined,
        )
        _reject_forbidden_state_keys(state)
        states.append(state)
        links.extend(_event_link_rows(state["state_id"], batch))
    assignments = [
        _assignment_from_event(event, winner_events) for event in assignment_events
    ]
    labels = _label_rows(states, assignments)
    return LearningTables(tuple(states), tuple(labels), tuple(links))


def _validated_online_quarantine_starts(
    segments: frozenset[int], starts: Mapping[int, int] | None,
) -> dict[int, int] | None:
    """学習用の全体隔離とは独立に、表示用の隔離開始を検証する。

    空の辞書を明示した場合は、正解データからは除外しても表示は止めない。
    盤面で物理的な受領量を確認でき、送付元との対応付けだけが欠ける場合に
    この区別が必要になる。
    """

    if starts is None:
        return None
    normalized = {int(game): int(sequence) for game, sequence in starts.items()}
    if any(game < 0 or sequence < 0 for game, sequence in normalized.items()):
        raise EventLearningTableError("当時版隔離の試合番号または通番が負です")
    return dict(sorted(normalized.items()))


def _is_online_quarantined(
    segment: int, through_sequence: int, training_quarantined: bool,
    starts: Mapping[int, int] | None,
) -> bool:
    if starts is None:
        return training_quarantined
    start = starts.get(segment)
    return start is not None and through_sequence >= start


def load_completed_run_batches(run_dir: Path) -> tuple[CommittedBatch, ...]:
    """完了検査を通った実行の採用部品だけを通番順に読む。"""

    manifest = _load_json_object(run_dir / "manifest.json")
    validation = _load_json_object(run_dir / "validation.json")
    complete = _load_json_object(run_dir / "COMPLETE")
    _validate_completion_hashes(run_dir, complete)
    parts = manifest.get("parts")
    part_rows = parts if isinstance(parts, list) else [parts]
    expected = 0
    batches: list[CommittedBatch] = []
    hasher = SemanticEventHasher()
    seen_batch_ids: set[str] = set()
    last_position = (-1, -1)
    for part in part_rows:
        if not isinstance(part, Mapping) or not isinstance(part.get("relative_path"), str):
            raise EventLearningTableError("manifestの採用部品一覧が不正です")
        part_path = run_dir / part["relative_path"]
        if _file_sha256(part_path) != part.get("physical_sha256"):
            raise EventLearningTableError("採用部品の物理要約値が一致しません")
        current = tuple(iter_committed_batches(part_path, expected_first_seq=expected))
        if not current:
            raise EventLearningTableError("採用部品に確定一括更新がありません")
        _validate_part_summary(part, current)
        batches.extend(current)
        for batch in current:
            last_position = _validate_cross_part_batch(
                batch, manifest, seen_batch_ids, last_position,
            )
            for event in batch.events:
                hasher.add(event)
        expected = int(current[-1].events[-1]["seq"]) + 1
    _validate_loaded_summary(manifest, validation, expected, hasher.hexdigest())
    return tuple(batches)


def _validate_completion_hashes(
    run_dir: Path, complete: Mapping[str, Any],
) -> None:
    if complete.get("format_version") != RUN_COMPLETE_VERSION:
        raise EventLearningTableError("完了印の形式版が不正です")
    expected = {
        "manifest_sha256": _file_sha256(run_dir / "manifest.json"),
        "validation_sha256": _file_sha256(run_dir / "validation.json"),
    }
    if any(complete.get(key) != value for key, value in expected.items()):
        raise EventLearningTableError("完了印と説明・検査ファイルの要約値が一致しません")


def _validate_loaded_summary(
    manifest: Mapping[str, Any], validation: Mapping[str, Any],
    event_count: int, semantic_sha256: str,
) -> None:
    expected = {
        "adopted_event_count": event_count,
        "semantic_sha256": semantic_sha256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise EventLearningTableError("manifestと採用出来事列が一致しません")
    validation_expected = {
        "valid": True, "event_count": event_count,
        "semantic_sha256": semantic_sha256,
    }
    if any(validation.get(key) != value for key, value in validation_expected.items()):
        raise EventLearningTableError("完成検査と読取り出来事件数が一致しません")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_part_summary(
    part: Mapping[str, Any], batches: Sequence[CommittedBatch],
) -> None:
    expected = {
        "batch_count": len(batches),
        "event_count": sum(len(batch.events) for batch in batches),
        "first_seq": int(batches[0].events[0]["seq"]),
        "last_seq": int(batches[-1].events[-1]["seq"]),
    }
    if any(part.get(key) != value for key, value in expected.items()):
        raise EventLearningTableError("manifestの部品要約と採用部品が一致しません")


def _validate_cross_part_batch(
    batch: CommittedBatch, manifest: Mapping[str, Any], seen_ids: set[str],
    last_position: tuple[int, int],
) -> tuple[int, int]:
    first = batch.events[0]
    source = manifest.get("source", {})
    expected_identity = (
        source.get("source_video_id"), manifest.get("build_id"), manifest.get("attempt_id"),
    )
    identity = (first.get("source_video_id"), first.get("build_id"), first.get("attempt_id"))
    position = (int(first["timing"]["available_frame"]), int(first["timing"]["available_ms"]))
    if identity != expected_identity or position[0] <= last_position[0] or position[1] < last_position[1]:
        raise EventLearningTableError("部品間の識別情報または公開位置が不正です")
    if batch.batch_id in seen_ids:
        raise EventLearningTableError("部品間で一括更新IDが重複しています")
    seen_ids.add(batch.batch_id)
    return position


def build_learning_tables_from_run(
    run_dir: Path, *, fold: int, tier: str, source_group_id: str = "",
    context_quarantined_segments: Collection[int] = (),
    online_context_quarantine_starts: Mapping[int, int] | None = None,
) -> LearningTables:
    """完成原本一件を読んで学習用三表を作る。"""

    return build_learning_tables(
        load_completed_run_batches(run_dir), fold=fold, tier=tier,
        source_group_id=source_group_id,
        context_quarantined_segments=context_quarantined_segments,
        online_context_quarantine_starts=online_context_quarantine_starts,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EventLearningTableError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise EventLearningTableError(f"JSONオブジェクトではありません: {path}")
    return value


def _collect_posthoc_events(
    batch: CommittedBatch, assignments: list[Mapping[str, Any]],
    winners: dict[str, Mapping[str, Any]],
) -> None:
    for event in batch.events:
        if event["event_type"] == "official_game_assignment":
            assignments.append(event)
        elif event["event_type"] == "winner_observed":
            event_id = str(event["event_id"])
            if event_id in winners:
                raise EventLearningTableError("勝者観測IDが重複しています")
            winners[event_id] = event


def _assignment_from_event(
    event: Mapping[str, Any], winners: Mapping[str, Mapping[str, Any]],
) -> _OfficialAssignment:
    payload = event.get("payload", {})
    side = payload.get("winner_side")
    if event.get("assertion", {}).get("state") != "confirmed" or side not in {"1P", "2P"}:
        raise EventLearningTableError("公式勝敗対応が確定値ではありません")
    winner_id = _verified_winner_id(event, winners, str(side))
    assignment = _OfficialAssignment(
        str(event["event_id"]), _integer(payload, "from_sequence"),
        _integer(payload, "to_sequence"), _integer(payload, "official_game_number"),
        str(side), winner_id,
    )
    if assignment.to_sequence < assignment.from_sequence or assignment.game_number < 1:
        raise EventLearningTableError("公式勝敗対応の範囲または試合番号が不正です")
    return assignment


def _verified_winner_id(
    assignment: Mapping[str, Any], winners: Mapping[str, Mapping[str, Any]],
    winner_side: str,
) -> str:
    causes = assignment.get("relations", {}).get("verified_cause_event_ids", [])
    matched = [str(event_id) for event_id in causes if str(event_id) in winners]
    if len(matched) != 1:
        raise EventLearningTableError("公式勝敗対応の勝者観測原因が一意ではありません")
    winner = winners[matched[0]]
    payload_side = winner.get("payload", {}).get("winner_side")
    expected_event_side = "p1" if winner_side == "1P" else "p2"
    if payload_side != winner_side or winner.get("side") != expected_event_side:
        raise EventLearningTableError("公式勝敗対応と勝者観測のsideが一致しません")
    if winner.get("assertion", {}).get("state") != "confirmed":
        raise EventLearningTableError("勝者観測が確定値ではありません")
    return matched[0]


def _validate_event_allowlist(batch: CommittedBatch) -> None:
    for event in batch.events:
        event_type = str(event["event_type"])
        if event_type not in KNOWN_EVENT_TYPES:
            raise EventLearningTableError(f"未登録の出来事種類です: {event_type}")
        posthoc = "available_only_posthoc" in event.get("missing_information", [])
        if posthoc and event_type not in POSTHOC_EVENT_TYPES:
            raise EventLearningTableError("事後情報がonline入力種類に混入しています")


def _is_posthoc_only(batch: CommittedBatch) -> bool:
    return all(event["event_type"] in POSTHOC_EVENT_TYPES for event in batch.events)


def _contains_boundary(batch: CommittedBatch) -> bool:
    return any(event["event_type"] == BOUNDARY_EVENT_TYPE for event in batch.events)


def _apply_batch(replay: _ReplayState, batch: CommittedBatch) -> None:
    replay.batch_ambiguous_amount = 0
    replay.batch_boundary_expiry_gap = 0
    replay.batch_disallowed_event_count = 0
    for event in batch.events:
        event_type = str(event["event_type"])
        replay.causal_exchange.observe(event)
        if event.get("payload", {}).get("model_input_allowed") is False:
            replay.batch_disallowed_event_count += 1
        if event_type == BOARD_EVENT_TYPE:
            side = _side(event)
            replay.boards[side] = dict(event)
            # 確定盤面が戻った時点で、物理的な連鎖処理は終わっている。
            # 攻撃量はOCRの確定が遅れる場合があるため、provisional は
            # attack_finalized / unfinalized まで保持する。
            replay.chain_active[side] = False
        elif event_type == "garbage_sent":
            _increase_pending(replay, event)
        elif event_type == "garbage_expired_at_boundary":
            _expire_pending(replay, event)
        elif event_type in _PENDING_DECREASE_KEYS:
            _decrease_pending(replay, event, _PENDING_DECREASE_KEYS[event_type])
        elif event_type == "garbage_same_frame_order_ambiguous":
            amount = _integer(event.get("payload", {}), "ambiguous_amount")
            _decrease_pending_amount(replay, _side(event), amount)
            replay.batch_ambiguous_amount += amount
        else:
            _apply_non_balance_event(replay, event)


_PENDING_DECREASE_KEYS = {
    "garbage_cancelled": "cancelled_amount",
    "garbage_fall_completed": "modeled_landed_amount",
}


def _increase_pending(replay: _ReplayState, event: Mapping[str, Any]) -> None:
    payload = event.get("payload", {})
    recipient = payload.get("recipient")
    if recipient not in SIDES:
        raise EventLearningTableError("送付先sideが不正です")
    replay.pending[str(recipient)] += _integer(payload, "sent_amount")


def _decrease_pending(
    replay: _ReplayState, event: Mapping[str, Any], amount_key: str,
) -> None:
    amount = _integer(event.get("payload", {}), amount_key)
    _decrease_pending_amount(replay, _side(event), amount)


def _decrease_pending_amount(replay: _ReplayState, side: str, amount: int) -> None:
    if amount > replay.pending[side]:
        raise EventLearningTableError(
            f"未解決量が負になります: {side} {replay.pending[side]} - {amount}"
        )
    replay.pending[side] -= amount


def _expire_pending(replay: _ReplayState, event: Mapping[str, Any]) -> None:
    """正式境界後に遅着する旧試合の消滅量を新試合へ持ち込まない。"""

    side = _side(event)
    amount = _integer(event.get("payload", {}), "expired_amount")
    before = replay.pending[side]
    replay.pending[side] = max(0, before - amount)
    replay.batch_boundary_expiry_gap += max(0, amount - before)


def _apply_non_balance_event(replay: _ReplayState, event: Mapping[str, Any]) -> None:
    event_type = str(event["event_type"])
    side = event.get("side")
    if side not in SIDES:
        return
    if event_type == "attack_provisional_updated":
        replay.provisional[str(side)] = _provisional_values(event)
        # 原本によって chain_started が欠けても、途中攻撃量の観測自体が
        # 連鎖中である直接証拠になる。
        replay.chain_active[str(side)] = True
    elif event_type in {"attack_finalized", "attack_provisional_unfinalized"}:
        replay.provisional[str(side)] = None
        replay.chain_active[str(side)] = False
    elif event_type == "chain_started":
        replay.chain_active[str(side)] = True
    elif event_type == "all_clear_gained":
        replay.all_clear[str(side)] = True
    elif event_type == "all_clear_consumed":
        replay.all_clear[str(side)] = False


def _provisional_values(event: Mapping[str, Any]) -> dict[str, int]:
    payload = event.get("payload", {})
    return {
        "generated_amount": _integer(payload, "provisional_generated_amount"),
        "score": _integer(payload, "provisional_score"),
        "chain_count": _integer(payload, "chain_count"),
    }


def _state_row(
    replay: _ReplayState, batch: CommittedBatch, *, fold: int, tier: str,
    source_group_id: str, context_quarantined: bool,
    training_context_quarantined: bool,
) -> dict[str, Any]:
    first = batch.events[0]
    last = batch.events[-1]
    build_id = str(first["build_id"])
    through_seq = int(last["seq"])
    row: dict[str, Any] = {
        "schema_version": LEARNING_TABLE_VERSION,
        "state_id": f"{build_id}:state-after-{through_seq:012d}",
        "source_video_id": str(first["source_video_id"]),
        "source_group_id": source_group_id or str(first["source_video_id"]),
        "build_id": build_id, "partition_fold": fold, "tier": tier,
        "availability_batch_id": batch.batch_id, "through_sequence": through_seq,
        "batch_first_sequence": int(first["seq"]),
        "online_segment_index": replay.online_segment_index,
        "available_frame": int(first["timing"]["available_frame"]),
        "available_ms": int(first["timing"]["available_ms"]),
        "trigger_event_count": len(batch.events),
    }
    for side in SIDES:
        row.update(_board_fields(side, replay.boards[side]))
        row.update(_unresolved_fields(side, replay))
    row.update(_usability_fields(
        replay, context_quarantined, training_context_quarantined,
    ))
    return row


def _board_fields(side: str, event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event.get("payload", {})
    grid = _flatten_grid(payload.get("grid"), "grid")
    mask = _flatten_grid(payload.get("unknown_mask"), "unknown_mask")
    context = payload.get("observed_context", {})
    if not isinstance(context, Mapping):
        raise EventLearningTableError("observed_contextがオブジェクトではありません")
    fields = {
        f"a_{side}_grid": grid, f"a_{side}_unknown_mask": mask,
        f"a_{side}_color_count": _integer(payload, "color_cell_count"),
        f"a_{side}_garbage_count": _integer(payload, "garbage_cell_count"),
        f"a_{side}_occupied_count": _integer(payload, "occupied_cell_count"),
        f"a_{side}_unknown_count": _integer(payload, "unknown_cell_count"),
    }
    fields.update(_context_fields(side, context))
    return fields


def _context_fields(side: str, context: Mapping[str, Any]) -> dict[str, Any]:
    next_pair = _pair_values(context.get("next_pair"))
    double_next = _pair_values(context.get("double_next_pair"))
    missing = context.get("missing_fields", [])
    if not isinstance(missing, list) or any(not isinstance(item, str) for item in missing):
        raise EventLearningTableError("missing_fieldsが文字列配列ではありません")
    return {
        f"a_{side}_score": _optional_int(context.get("score")),
        f"a_{side}_next_first": next_pair[0], f"a_{side}_next_second": next_pair[1],
        f"a_{side}_double_next_first": double_next[0],
        f"a_{side}_double_next_second": double_next[1],
        f"a_{side}_tsumo_count": _optional_int(context.get("tsumo_count")),
        f"a_{side}_all_clear_pending": _optional_bool(context.get("all_clear_pending")),
        f"a_{side}_stable_confidence": _optional_int(
            context.get("stable_persistence_confidence")
        ),
        f"a_{side}_chain_mechanism": _optional_text(context.get("chain_mechanism")),
        f"quality_{side}_match_end_locked": _optional_bool(context.get("match_end_locked")),
        f"quality_{side}_post_match_lockdown": _optional_bool(
            context.get("post_match_lockdown_active")
        ),
        f"quality_{side}_missing_fields": sorted(missing),
    }


def _unresolved_fields(side: str, replay: _ReplayState) -> dict[str, Any]:
    provisional = replay.provisional[side]
    return {
        f"b_{side}_pending_garbage": replay.pending[side],
        f"b_{side}_chain_active": replay.chain_active[side],
        f"b_{side}_all_clear_event_state": replay.all_clear[side],
        f"b_{side}_provisional_generated": _value_or_none(provisional, "generated_amount"),
        f"b_{side}_provisional_score": _value_or_none(provisional, "score"),
        f"b_{side}_provisional_chain_count": _value_or_none(provisional, "chain_count"),
        f"b_{side}_causal_pending_garbage": replay.causal_exchange.pending(side),
        f"b_{side}_causal_effective_rate": replay.causal_exchange.rate(side),
        "b_causal_exchange_usable": replay.causal_exchange.usable,
        "b_causal_observed_attack_balance": replay.causal_exchange.observed_balance(),
        "quality_causal_quarantined_finalized_count": (
            replay.causal_exchange.quarantined_finalized_count
        ),
        "quality_causal_quarantined_finalized_amount": (
            replay.causal_exchange.quarantined_finalized_amount
        ),
        "quality_causal_quarantined_landing_count": (
            replay.causal_exchange.quarantined_landing_count
        ),
        "quality_causal_quarantined_landing_amount": (
            replay.causal_exchange.quarantined_landing_amount
        ),
    }


def _usability_fields(
    replay: _ReplayState, context_quarantined: bool,
    training_context_quarantined: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    for side in SIDES:
        context = replay.boards[side].get("payload", {}).get("observed_context", {})
        if context.get("match_end_locked") is True:
            reasons.append(f"{side}_match_end_locked")
        if context.get("post_match_lockdown_active") is True:
            reasons.append(f"{side}_post_match_lockdown_active")
    a_usable = not reasons
    b_reasons = list(reasons)
    if not replay.causal_exchange.usable:
        b_reasons.extend(replay.causal_exchange.reason_codes)
    settled_disagreement = _settled_pending_disagreement(replay)
    if settled_disagreement:
        b_reasons.append("causal_ledger_pending_disagrees_while_settled")
    if replay.batch_ambiguous_amount:
        b_reasons.append("same_frame_accounting_order_ambiguous")
    if replay.batch_boundary_expiry_gap:
        b_reasons.append("boundary_expiry_after_formal_reset")
    if replay.batch_disallowed_event_count and not replay.batch_ambiguous_amount:
        b_reasons.append("event_explicitly_disallowed")
    if context_quarantined:
        b_reasons.append("physical_accounting_unsupported_in_segment")
    return {
        "a_input_usable": a_usable, "a_reason_codes": sorted(reasons),
        "b_input_usable": not b_reasons, "b_reason_codes": sorted(b_reasons),
        "c_input_usable": not b_reasons, "c_reason_codes": sorted(b_reasons),
        "quality_same_frame_ambiguous_amount": replay.batch_ambiguous_amount,
        "quality_boundary_expiry_gap": replay.batch_boundary_expiry_gap,
        "quality_disallowed_event_count": replay.batch_disallowed_event_count,
        "quality_physical_accounting_unsupported_segment": training_context_quarantined,
        "quality_causal_ledger_pending_disagreement": settled_disagreement,
    }


def _settled_pending_disagreement(replay: _ReplayState) -> int:
    """連鎖停止中は、既知0の暫定攻撃を未確定扱いせず残量を照合する。"""

    if replay.batch_ambiguous_amount or replay.batch_boundary_expiry_gap:
        return 0
    return settled_pending_disagreement(
        replay.pending,
        {side: replay.causal_exchange.pending(side) for side in SIDES},
        replay.chain_active,
        {
            side: _value_or_none(replay.provisional[side], "generated_amount")
            for side in SIDES
        },
        causal_usable=replay.causal_exchange.usable,
    )


def _event_link_rows(state_id: str, batch: CommittedBatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in batch.events:
        payload = event.get("payload", {})
        allowed = payload.get("model_input_allowed") is not False
        rows.append({
            "schema_version": LEARNING_TABLE_VERSION, "state_id": state_id,
            "event_id": str(event["event_id"]), "event_sequence": int(event["seq"]),
            "event_type": str(event["event_type"]), "side": str(event["side"]),
            "availability_batch_id": batch.batch_id, "model_input_allowed": allowed,
        })
    return rows


def _label_rows(
    states: Sequence[Mapping[str, Any]], assignments: Sequence[_OfficialAssignment],
) -> list[dict[str, Any]]:
    _validate_assignment_ranges(assignments)
    provisional: list[dict[str, Any]] = []
    for state in states:
        assignment = _assignment_for_range(
            assignments, int(state["batch_first_sequence"]),
            int(state["through_sequence"]),
        )
        row = _label_row(state, assignment)
        row["training_usable"] = bool(
            row["evaluation_scope_usable"] and row["c_input_usable"]
            and not state["quality_physical_accounting_unsupported_segment"]
        )
        provisional.append(row)
    counts = Counter(
        row["game_key"] for row in provisional if row["training_usable"]
    )
    for row in provisional:
        key = row["game_key"]
        row["sample_weight"] = 1.0 / counts[key] if row["training_usable"] else None
    return provisional


def _label_row(
    state: Mapping[str, Any], assignment: _OfficialAssignment | None,
) -> dict[str, Any]:
    base = {
        "schema_version": LEARNING_TABLE_VERSION, "state_id": state["state_id"],
        "source_video_id": state["source_video_id"],
        "source_group_id": state["source_group_id"],
        "partition_fold": state["partition_fold"],
        "a_input_usable": state["a_input_usable"], "b_input_usable": state["b_input_usable"],
        "c_input_usable": state["c_input_usable"],
    }
    if assignment is None:
        return base | {
            "game_key": None, "official_game_number": None, "winner_side": None,
            "p1_won": None, "official_assignment_event_id": None,
            "official_winner_event_id": None,
            "win_label_available": False,
            "label_reason_codes": ["official_game_assignment_missing"],
            "evaluation_scope_usable": False,
            "evaluation_scope_reason_codes": ["official_game_assignment_missing"],
        }
    evaluation_usable = bool(state["a_input_usable"])
    return base | {
        "game_key": f"{state['source_video_id']}:game-{assignment.game_number:04d}",
        "official_game_number": assignment.game_number,
        "winner_side": assignment.winner_side, "p1_won": assignment.winner_side == "1P",
        "official_assignment_event_id": assignment.event_id,
        "official_winner_event_id": assignment.winner_event_id,
        "win_label_available": True, "label_reason_codes": [],
        "evaluation_scope_usable": evaluation_usable,
        "evaluation_scope_reason_codes": (
            [] if evaluation_usable else list(state["a_reason_codes"])
        ),
    }


def _validate_assignment_ranges(assignments: Sequence[_OfficialAssignment]) -> None:
    ordered = sorted(assignments, key=lambda item: (item.from_sequence, item.to_sequence))
    for previous, current in zip(ordered, ordered[1:]):
        if current.from_sequence <= previous.to_sequence:
            raise EventLearningTableError("公式勝敗対応の通番範囲が重複しています")


def _assignment_for_range(
    assignments: Sequence[_OfficialAssignment], first_sequence: int,
    last_sequence: int,
) -> _OfficialAssignment | None:
    found = [
        item for item in assignments
        if item.from_sequence <= first_sequence and last_sequence <= item.to_sequence
    ]
    if len(found) > 1:
        raise EventLearningTableError("一状態が複数の公式試合へ対応しています")
    return found[0] if found else None


def _reject_forbidden_state_keys(state: Mapping[str, Any]) -> None:
    forbidden = MODEL_FORBIDDEN_STATE_KEYS & set(state)
    if forbidden:
        raise EventLearningTableError(f"状態表へ正解情報が混入しています: {sorted(forbidden)}")


def _flatten_grid(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or any(not isinstance(row, list) for row in value):
        raise EventLearningTableError(f"{name}が二次元配列ではありません")
    flat = [cell for row in value for cell in row]
    if len(flat) != BOARD_CELL_COUNT or any(not _is_int(cell) for cell in flat):
        raise EventLearningTableError(f"{name}の形または値が不正です")
    return [int(cell) for cell in flat]


def _pair_values(value: Any) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    if not isinstance(value, Mapping):
        raise EventLearningTableError("色ペアがオブジェクトではありません")
    return _optional_int(value.get("first")), _optional_int(value.get("second"))


def _integer(container: Mapping[str, Any], key: str) -> int:
    value = container.get(key)
    if not _is_int(value) or int(value) < 0:
        raise EventLearningTableError(f"{key}が0以上の整数ではありません")
    return int(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not _is_int(value):
        raise EventLearningTableError("任意整数値が不正です")
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise EventLearningTableError("任意真偽値が不正です")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise EventLearningTableError("任意文字列が不正です")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _side(event: Mapping[str, Any]) -> str:
    side = event.get("side")
    if side not in SIDES:
        raise EventLearningTableError("出来事sideがp1/p2ではありません")
    return str(side)


def _value_or_none(values: Mapping[str, int] | None, key: str) -> int | None:
    return None if values is None else values[key]


__all__ = [
    "EventLearningTableError", "LEARNING_TABLE_VERSION", "LearningTables",
    "build_learning_tables", "build_learning_tables_from_run",
    "load_completed_run_batches",
]
