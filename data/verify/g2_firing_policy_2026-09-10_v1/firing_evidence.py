"""同actionの原配置ログとorigin/実観測精算票を値で結合する。"""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
import math
from typing import Any
from firing_fixed import require

ORIGIN_KIND, SETTLEMENT_KIND = 'firing_origin/v1', 'firing_settlement/v1'
OBSERVATION_KIND, OBSERVATIONS = 'captured_clear_raw/v1', 2
OBS_KEYS = frozenset(('kind', 'scope', 'source_token', 'action', 'observed_at', 'available_at',
                      'raw_grid', 'effect_window', 'capture_ref'))


def frozen(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((field.name, frozen(getattr(value, field.name))) for field in fields(value)))
    if type(value) is dict:
        require(all(type(k) is str for k in value), 'proof_key')
        return (dict, tuple((key, frozen(value[key])) for key in sorted(value)))
    if type(value) in (tuple, list): return (type(value), tuple(frozen(v) for v in value))
    require(type(value) in (str, int, float, bool, type(None)), 'proof_value')
    require(type(value) is not float or math.isfinite(value), 'proof_nonfinite')
    return type(value), value


def same(left: Any, right: Any, reason: str) -> None:
    require(frozen(left) == frozen(right), reason)


def token(value: Any) -> None:
    require(type(value) is str and bool(value), 'source_token')


def common(p: Any, evidence: Any, state: Any, proof: Any, kind: str, keys: set[str]) -> None:
    s = p.S
    require(type(state) is s.OwnerState and type(proof) is dict and set(proof) == keys, 'proof_shape')
    s.validate_scope(state.scope)
    s.validate_accounting(state)
    require(s.integer(state.action), 'state_action')
    same(evidence.scope, state.scope, 'scope')
    require(proof['kind'] == kind, 'proof_kind')
    same(proof['evidence'], asdict(evidence), 'evidence_fields')
    token(proof['source_token'])
    s.available(state.clock, evidence.available_at)
    require(evidence.available_at.sequence > state.clock.sequence, 'available_not_forward')


def placement(p: Any, logs: list[Any], state: Any, proof: Any, origin: Any) -> None:
    entries = [entry for entry in state.history if entry.kind == 'placement' and entry.action == origin.action]
    require(len(entries) == 1 and entries[0].event_id == proof['placement_event_id'], 'placement_history')
    entry = entries[0]
    records = [row for row in logs if row['purpose'] == 'placement' and row['proof_sha'] == entry.event_id]
    require(len(records) == 1, 'placement_proof_log')
    prior = records[0]['proof']
    require(p.digest(prior) == entry.event_id and prior.get('token') == proof['source_token'], 'placement_token')
    same(prior.get('scope'), asdict(origin.scope), 'placement_scope')
    require(sum(entry.delta) == 2, 'placement_pair')
    p.S.available(entry.available_at, origin.observed_at)


def origin(p: Any, logs: list[Any], evidence: Any, state: Any, proof: Any) -> str:
    s = p.S
    s.validate_origin(evidence)
    common(p, evidence, state, proof, ORIGIN_KIND,
        {'kind', 'source_token', 'placement_event_id', 'evidence'})
    require(evidence.action == state.action and not state.debts, 'origin_action_or_debt')
    require(all(s.origin_key(old) != s.origin_key(evidence) for old in state.origins), 'origin_replay')
    require(state.counter == s.color_counts(evidence.before_grid), 'origin_counter_before')
    placement(p, logs, state, proof, evidence)
    return evidence.origin_id


def registered(p: Any, logs: list[Any], evidence: Any, state: Any, proof: Any) -> None:
    s, origin = p.S, evidence.origin
    require(origin.action == state.action, 'settlement_same_action')
    require(len(state.debts) == 1, 'registered_debt')
    same(state.debts[0].origin, origin, 'registered_origin')
    require(sum(frozen(old) == frozen(origin) for old in state.origins) == 1, 'registered_origin_history')
    records = [row for row in logs if row['purpose'] == 'origin'
        and row['proof']['evidence']['origin_id'] == origin.origin_id]
    require(len(records) == 1, 'registered_origin_proof')
    record = records[0]
    require(p.digest(record['proof']) == record['proof_sha'], 'registered_proof_changed')
    same(record['proof']['evidence'], asdict(origin), 'registered_origin_fields')
    require(record['proof']['source_token'] == proof['source_token'], 'settlement_token')
    require(state.counter == s.color_counts(origin.before_grid), 'settlement_before_counter')


def observation(s: Any, row: Any, origin: Any, proof: Any, now: Any) -> tuple[Any, Any]:
    require(type(row) is dict and set(row) == OBS_KEYS and row['kind'] == OBSERVATION_KIND, 'observation_shape')
    same(row['scope'], asdict(origin.scope), 'observation_scope')
    require(type(row['action']) is int and row['action'] == origin.action, 'observation_action')
    require(row['source_token'] == proof['source_token'] and row['effect_window'] is False, 'clear_token_or_window')
    token(row['capture_ref'])
    s.validate_grid(row['raw_grid'])
    same(row['raw_grid'], origin.predicted_final, 'observed_final_grid')
    observed, available = s.Clock(**row['observed_at']), s.Clock(**row['available_at'])
    s.available(origin.available_at, observed)
    s.available(observed, available)
    s.available(available, now)
    require(observed.frame == available.frame and observed.time_sec == available.time_sec, 'capture_available_frame')
    return observed, available


def settlement(p: Any, logs: list[Any], evidence: Any, state: Any, proof: Any) -> str:
    s = p.S
    require(type(evidence) is s.SettlementEvidence, 'settlement_type')
    s.validate_origin(evidence.origin)
    s.validate_claim(evidence.claim)
    common(p, evidence, state, proof, SETTLEMENT_KIND, {'kind', 'source_token', 'evidence', 'observations'})
    same(evidence.claim, s.counter_claim(state), 'fresh_counter_claim')
    registered(p, logs, evidence, state, proof)
    rows = proof['observations']
    require(type(rows) is tuple and len(rows) == OBSERVATIONS, 'two_observations')
    first, last = [observation(s, row, evidence.origin, proof, evidence.available_at) for row in rows]
    require(first[0].frame < last[0].frame and first[0].time_sec < last[0].time_sec
        and first[0].sequence < last[0].sequence, 'distinct_observation_clocks')
    s.available(first[1], last[0])
    same(last[1], evidence.available_at, 'fresh_settlement_available')
    require(rows[0]['capture_ref'] != rows[1]['capture_ref'], 'distinct_capture_refs')
    return evidence.origin.origin_id
