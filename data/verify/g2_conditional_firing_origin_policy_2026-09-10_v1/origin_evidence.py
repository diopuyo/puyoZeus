"""条件世界の起源を元S型payloadと別来歴へ束縛する。原event資格は付けない。"""
from __future__ import annotations
from dataclasses import asdict
import json
from typing import Any
import origin_fixed as F

KIND='conditional_firing_origin/v1'
PLACEMENT='conditional_firing_placement/v1'
PREFIX='conditional_world:'
KEYS={'kind','source_token','placement_event_id','evidence','placement','previous_anchor','backend',
    'conditional_world','physical_certified','original_event'}
FPS,STRIDE,OP_ORIGIN=60,2,2


def equal(p: Any, left: Any, right: Any, reason: str) -> None:
    F.require(p.encoded(left)==p.encoded(right),reason)


def identity(p: Any, state: Any, placement: Any, final: Any) -> str:
    return p.digest(dict(kind=KIND,scope=asdict(state.scope),action=state.action,
        token=placement['token'],placement=p.digest(placement),before=placement['grid'],final=final))


def placement(p: Any, e: Any, logs: Any, state: Any, value: Any, proof: Any) -> Any:
    e.placement(p,logs,state,proof,value)
    records=[r for r in logs if r['purpose']=='placement' and r['proof_sha']==proof['placement_event_id']]
    source=records[0]['proof']
    e.same(source,proof['placement'],'placement_log_contents')
    F.require(source['kind']==PLACEMENT and source['token']==proof['source_token'],'placement_kind_token')
    equal(p,source['scope'],asdict(state.scope),'placement_scope')
    F.require(source['inferred_grid_is_observed'] is False and source['physical_certified'] is False
        and source['current_permission'] is False,'placement_permissions')
    equal(p,source['grid'],source['inferred_final'],'placement_grid')
    equal(p,source['grid'],value.before_grid,'origin_before')
    F.require(source['available_frame']==value.available_at.frame
        and source['available_time']==value.available_at.time_sec,'same_call_origin')
    F.require(value.observed_at==value.available_at and value.available_at.sequence==
        value.available_at.frame*4+OP_ORIGIN,'origin_operation')
    return source


def prior(p: Any, state: Any, source: Any, saved: Any) -> None:
    value=saved['certificate']; fields=json.loads(value['evidence_json'])
    F.require(value['provisional'] is True and all(value[k] is False for k in ('physical_certified',
        'integer_current_permission','accounting_permission','production_permission')),'certificate_permissions')
    F.require(all(saved[k] is False for k in ('current_permission','accounting_permission','production_permission')),
        'anchor_permissions')
    equal(p,json.loads(saved['owner_scope']),asdict(state.scope),'anchor_owner_scope')
    scope=value['scope']
    F.require(len(scope)==7 and scope[0]=='sha256:'+state.scope.source_sha256 and scope[1]==state.scope.run_id
        and scope[-1]==state.scope.side and scope[2]==state.scope.reset_epoch,'anchor_scope')
    F.require(all(type(scope[k]) is int for k in (2,3,4,5)),'anchor_scope_types')
    F.require(type(value['action']) is int and value['action']+1==state.action,'anchor_action')
    F.require(value['frame']<source['occurred'][0] and value['clock']==value['frame']/FPS,'anchor_clock')
    equal(p,value['inferred_grid'],source['before_grid'],'anchor_before')
    equal(p,json.loads(value['integer_anchor']),None if state.current is None else asdict(state.current),'integer_slot')
    F.require(value['evidence_json']==source['previous_conditional_certificate'],'anchor_original_certificate')
    for key,val in dict(scope=scope,frame=value['frame'],clock=value['clock'],action=value['action'],
        sm=value['sm_grid'],inferred_final=value['inferred_grid'],integer_anchor=value['integer_anchor']).items():
        equal(p,fields[key],val,'anchor_fields:'+key)
    equal(p,fields['conditional_PB'][0],value['hidden'],'anchor_hidden')
    F.require(fields['conditional_not_recognition_confidence'] is True and fields['original_PB_unchanged'] is True,
        'anchor_provisional')
    equal(p,fields['prefix_source'],source['prefix_source'],'anchor_prefix_source')


def observations(p: Any, source: Any, anchor: Any) -> None:
    rows=source['observations']
    F.require(type(rows) in (tuple,list) and len(rows)==2,'two_captures')
    for row in rows:
        F.require(set(row)=={'frame','scope','token','raw','raw_proof'},'capture_fields')
        equal(p,row['scope'],anchor['certificate']['scope'],'capture_scope7')
        F.require(row['token']==source['token'] and type(row['frame']) is int,'capture_token')
        raw=row['raw_proof']; inner=raw['capture']
        F.require(row['frame']==raw['captured_frame']==inner['captured_frame'],'capture_frame')
        for grid in (raw['raw_grid'],inner['raw']['grid'],inner['filtered']['grid'],source['observed_raw']):
            equal(p,grid,row['raw'],'capture_raw')
    F.require(rows[0]['frame']+STRIDE==rows[1]['frame']==source['available_frame'],'capture_adjacency')
    equal(p,[rows[0]['frame'],rows[0]['frame']/FPS],source['occurred'],'first_clock')
    equal(p,[rows[1]['frame'],rows[1]['frame']/FPS],source['clear_last'],'last_clock')
    F.require(type(source['clear_observations']) is int and source['clear_observations']==2,'two_votes')
    equal(p,rows[-1]['raw_proof'],source['raw_capture'],'last_raw_capture')


def geometry(p: Any, source: Any, value: Any, expected_backend: Any) -> None:
    from src import puyo_core_bridge as core
    from src.production_config import GHOST_CHAIN_RULE_ENABLED
    import path_support as P
    equal(p,F.backend(),expected_backend,'backend')
    F.require(GHOST_CHAIN_RULE_ENABLED is True and source['simulation_rule']=='GHOST_CHAIN_RULE_ENABLED=True','rule')
    before,placed,raw=P.grid(source['before_grid']),P.grid(source['grid']),P.grid(source['observed_raw'],hidden_unknown=True)
    prior=core.simulate_chain(P.board(core,before),exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    F.require(prior.chain_count==0,'before_already_firing')
    pair=P.pair(source['pair'])
    choices=tuple(g for g in P.options(core,before,pair) if P.compatible(raw,g))
    F.require(choices==(placed,) and any(c==P.UNKNOWN for c in raw[0]),'unique_conditional_support')
    F.require(tuple(a-b for a,b in zip(p.S.color_counts(placed),p.S.color_counts(before)))==
        tuple(pair.count(c) for c in range(1,6)),'legal_pair')
    result=core.simulate_chain(P.board(core,placed),exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    F.require(result.chain_count>=1,'prediction_chain')
    equal(p,result.final_board.to_dict()['grid'],source['predicted_final'],'prediction_source')
    equal(p,result.final_board.to_dict()['grid'],value.predicted_final,'prediction_origin')
    F.require(sum(value.erased)==result.total_erased,'prediction_erasure')


def origin(p: Any, e: Any, logs: Any, value: Any, state: Any, proof: Any) -> str:
    p.S.validate_origin(value)
    e.common(p,value,state,proof,KIND,KEYS)
    F.require(proof['conditional_world'] is True and proof['physical_certified'] is False
        and proof['original_event'] is False,'conditional_only')
    F.require(value.action==state.action and not state.debts,'action_or_debt')
    F.require(all(p.S.origin_key(old)!=p.S.origin_key(value) for old in state.origins),'origin_replay')
    F.require(state.counter==p.S.color_counts(value.before_grid),'counter_before')
    source=placement(p,e,logs,state,value,proof)
    prior(p,state,source,proof['previous_anchor'])
    observations(p,source,proof['previous_anchor'])
    geometry(p,source,value,proof['backend'])
    expected=identity(p,state,source,value.predicted_final)
    F.require(value.origin_id==expected and value.event_identity==PREFIX+expected,'conditional_identity')
    return value.origin_id
