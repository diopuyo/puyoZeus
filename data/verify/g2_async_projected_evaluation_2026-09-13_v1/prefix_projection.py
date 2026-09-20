"""元Laneの単一PREPOP仮説を条件命題ごと予測する。current/台帳へは反映しない。"""
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any
import identified_origin_candidate as C
import lane_state as N
import projected_view as P


@dataclass(frozen=True)
class ConditionalProjection:
    view: P.ProjectedView
    candidate_json: str
    condition_json: str
    conditional_weights: bool = field(default=True, init=False)
    source_producer_authorized: bool = field(default=False, init=False)
    accounting_connected: bool = field(default=False, init=False)
    quality_gate_clear: bool = field(default=False, init=False)


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def project(parts: Any, ledger: Any, lane: Any, step: dict, origin: dict) -> ConditionalProjection:
    require = parts.mode.B.require
    require(type(lane) is N.Lane and lane.parts is parts, 'projection_lane_owner')
    require(len(lane.families) == 1 and lane.families[0].phase == N.P.PREPOP,
        'projection_single_prepop_required')
    family = lane.families[0]
    require(lane.observations and lane.observations[-1]['frame'] == family.value.frame,
        'projection_family_observation')
    candidate = C.select(parts, ledger, lane.base, lane.families, step, origin)
    before = P.digest(family.value)
    view = P.project(family.value, family.value.scope,
        parts.mode.B.Board.from_dict(origin['before_board']), origin_frame=step['frame_idx'],
        cutoff_frame=step['frame_idx'], origin_token=step['token'], origin_trigger_sec=origin['trigger_sec'])
    condition = dict(meaning='conditional_on_single_prepop_family', prefix=family.prefix,
        phase=family.phase, base_applied_tokens=list(lane.base), applied_at_cutoff=list(ledger.applied),
        absolute_arrival_index=candidate['absolute_arrival_index'], fired_tokens=sorted(lane.fired),
        prior_assumption=lane.assumption, observed_frame=family.value.frame,
        observation_evidence_key=lane.observations[-1]['evidence_key'],
        observation_sha256=hashlib.sha256(encoded(lane.observations[-1]).encode()).hexdigest(),
        source_qualification_replay_required=True, live_owner_check_required=True,
        arrival_ACK_claimed=False, integer_current_changed=False)
    require(before == P.digest(family.value), 'projection_family_mutated')
    return ConditionalProjection(view, encoded(candidate), encoded(condition))
