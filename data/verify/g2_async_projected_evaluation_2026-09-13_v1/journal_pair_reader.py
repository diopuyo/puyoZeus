"""本線の既存Witnessから現在updateの原票を読む。世代変化を隠さない。"""
from dataclasses import dataclass, field
import json
from typing import Any
import journal_witness as W

SIDES = ('1P', '2P')


def require(condition: bool, reason: str) -> None:
    """選択済みWitnessの原契約へ委譲し、旧emit型と実stream型の両方を保つ。"""
    context = W.C if hasattr(W, 'C') else W.K.C
    context.require(condition, reason)


@dataclass(frozen=True)
class JournalPairRead:
    frame: int
    rows_json: str
    holds: tuple[str, ...]
    accounting_permission: bool = field(default=False, init=False)
    future_fire_power_supply_authorized: bool = field(default=False, init=False)
    live_hook_verified: bool = field(default=False, init=False)
    physical_identity_verified: bool = field(default=False, init=False)
    quality_gate_clear: bool = field(default=False, init=False)


def validate_row(journal: Any, pipe: Any, row: dict, side: str, frame: int) -> str | None:
    scope = journal.scope(pipe, side, frame, frame / 60)
    require(row['kind'] == 'step' and row['status'] == 'returned'
            and row['exception'] is None and row['side'] == side, 'projected_J_completion')
    names = ('frame_idx', 'time_sec', 'source_id', 'run_id', 'pipe_object_id')
    require(all(row[name] == scope[name] for name in names), 'projected_J_scope')
    require(row['generation_after'] == scope['generation'], 'projected_J_current_generation')
    require(row['software_reset'] == journal.epoch(pipe, side), 'projected_J_software_epoch')
    token = row['token']
    require(type(token) is str and token.startswith('step:'), 'projected_J_token')
    ordinal = token.removeprefix('step:')
    require(ordinal.isascii() and ordinal.isdigit() and str(int(ordinal)) == ordinal,
            'projected_J_ordinal')
    index = int(ordinal)
    require(0 <= index < journal.steps and journal.expected[index] == (frame, side), 'projected_J_issued')
    return f'{side}:generation_changed_within_step' if row['generation'] != row['generation_after'] else None


def read_pair(witness: Any, journal: Any, pipe: Any, frame: int) -> JournalPairRead:
    require(type(witness) is W.Witness and witness.journal is journal, 'projected_J_witness_owner')
    require(not witness.closed and witness.error is None and not journal.closed
            and not journal.errors and journal.active is None, 'projected_J_lifetime')
    require(journal.pipe is pipe and journal.tracker._pipeline is pipe, 'projected_J_pipe')
    require(type(frame) is int and journal.history.frame == frame
            and journal.history.time_sec == frame / 60, 'projected_J_live_clock')
    if set(witness.rows) != set(SIDES):
        return JournalPairRead(frame, '[]', ('both_side_J_not_available',))
    saved = [json.loads(witness.rows[side]) for side in SIDES]
    require(all(row['frame_idx'] <= frame for row in saved), 'projected_J_future_frame')
    if any(row['frame_idx'] < frame for row in saved):
        return JournalPairRead(frame, '[]', ('both_side_J_not_current',))
    rows = witness.pair(frame)
    holds = tuple(reason for side, row in zip(SIDES, rows, strict=True)
                  if (reason := validate_row(journal, pipe, row, side, frame)) is not None)
    ordinals = tuple(int(row['token'].removeprefix('step:')) for row in rows)
    require(ordinals == (journal.steps - 2, journal.steps - 1), 'projected_J_latest_pair')
    return JournalPairRead(frame, json.dumps(rows, sort_keys=True, allow_nan=False), holds)
