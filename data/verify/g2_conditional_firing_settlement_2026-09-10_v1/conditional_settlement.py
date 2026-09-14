"""条件付きraw二票の私有精算。確定originや確定盤面へ昇格しない。"""
from __future__ import annotations
from dataclasses import asdict
from types import FunctionType,SimpleNamespace
from typing import Any

KIND='conditional_firing_settlement/v1'
ORIGIN_KIND='conditional_firing_origin/v1'
OBS_KIND='conditional_clear_raw/v1'
FPS,STRIDE,OPERATIONS=60,2,4
ALLOWED_STATES=frozenset(('tsumo_fall','stable'))


def registered(p: Any, e: Any, logs: Any, value: Any, state: Any, proof: Any) -> Any:
    import origin_fixed as F
    e.registered(p,logs,value,state,proof)
    origins=[r['proof'] for r in logs if r['purpose']=='origin'
        and r['proof']['evidence']['origin_id']==value.origin.origin_id]
    F.require(len(origins)==1 and origins[0]['kind']==ORIGIN_KIND,'settlement_origin_kind')
    original=origins[0]
    F.require(value.origin.event_identity.startswith('conditional_world:')
        and original['conditional_world'] is True and original['physical_certified'] is False
        and original['original_event'] is False,'settlement_conditional_origin')
    e.same(original,proof['origin_proof'],'settlement_origin_proof')
    return original


def observation(p: Any, e: Any, row: Any, origin: Any, source: Any) -> None:
    import origin_fixed as F
    import path_support as P
    F.require(type(row) is dict and row['kind']==OBS_KIND,'settlement_observation_kind')
    e.same(row['scope'],asdict(origin.scope),'settlement_observation_scope')
    e.same(row['live_scope'],source['previous_anchor']['certificate']['scope'],'settlement_live_scope')
    F.require(type(row['action']) is int and row['action']==origin.action
        and row['source_token']==source['source_token'] and row['origin_id']==origin.origin_id,'settlement_token_action')
    F.require(row['effect_window'] is False and row['active'] is True and row['actual_chain'] is False
        and row['pending_next'] is False and row['quiet'] is True,'settlement_clear_context')
    F.require(row['state'] in ALLOWED_STATES and row['conditional_world'] is True
        and row['physical_certified'] is False and row['current_permission'] is False,'settlement_not_current')
    observed,available=p.S.Clock(**row['observed_at']),p.S.Clock(**row['available_at'])
    p.S.available(origin.available_at,observed); p.S.available(observed,available)
    F.require(observed.frame>origin.available_at.frame and observed.frame==available.frame
        and observed.time_sec==available.time_sec==observed.frame/FPS
        and observed.sequence==observed.frame*OPERATIONS and available.sequence==observed.sequence+1,'settlement_clock')
    raw=P.grid(row['raw_grid'],hidden_unknown=True)
    F.require(raw[P.HIDDEN_ROWS:]==origin.predicted_final[P.HIDDEN_ROWS:]
        and P.compatible(raw,origin.predicted_final),'settlement_world')
    capture=row['raw_capture']
    F.require(capture['captured_frame']==observed.frame,'settlement_capture_frame')
    e.same(capture['raw_grid'],row['raw_grid'],'settlement_capture_raw')
    inner=capture['capture']
    F.require(inner['captured_frame']==observed.frame,'settlement_original_capture_frame')
    for key in ('raw','filtered'):
        e.same(P.grid(inner[key]['grid'],hidden_unknown=True),raw,'settlement_original_raw')
    F.require(row['capture_ref']==p.digest(dict(capture=capture,scope=row['live_scope'])),'settlement_capture_digest')


def validate(p: Any, e: Any, logs: Any, value: Any, state: Any, proof: Any) -> str:
    import origin_fixed as F
    F.require(type(value) is p.S.SettlementEvidence,'settlement_type')
    p.S.validate_origin(value.origin); p.S.validate_claim(value.claim)
    e.common(p,value,state,proof,KIND,{'kind','source_token','evidence','observations','origin_proof',
        'conditional_world','physical_certified','current_permission'})
    F.require(proof['conditional_world'] is True and proof['physical_certified'] is False
        and proof['current_permission'] is False,'settlement_permissions')
    e.same(value.claim,p.S.counter_claim(state),'settlement_fresh_claim')
    source=registered(p,e,logs,value,state,proof)
    rows=proof['observations']
    F.require(type(rows) is tuple and len(rows)==STRIDE,'settlement_two_captures')
    for row in rows: observation(p,e,row,value.origin,source)
    first,last=rows
    F.require(first['observed_at']['frame']+STRIDE==last['observed_at']['frame']
        and first['capture_ref']!=last['capture_ref'],'settlement_adjacent_distinct')
    e.same(first['raw_grid'],last['raw_grid'],'settlement_same_world_raw')
    e.same(last['available_at'],asdict(value.available_at),'settlement_available')
    p.S.available(p.S.Clock(**first['available_at']),p.S.Clock(**last['observed_at']))
    return value.origin.origin_id


def policy_type(base: Any) -> type:
    """凍結origin policyのorigin検査と旧正常分岐を維持して精算だけ追加する。"""
    import origin_fixed as F
    import settlement_provenance as V
    prior=V.verify(base)
    def settlement(p: Any, logs: Any, value: Any, state: Any, proof: Any) -> str:
        if proof.get('kind')==KIND:
            return validate(p,prior,logs,value,state,proof)
        return prior.settlement(p,logs,value,state,proof)
    facade=SimpleNamespace(**(vars(prior)|dict(settlement=settlement)))
    arm=FunctionType(base.arm.__code__,dict(base.arm.__globals__,E=facade),base.arm.__name__,
        base.arm.__defaults__,base.arm.__closure__)
    arm.__kwdefaults__=base.arm.__kwdefaults__
    result=type('ConditionalSettlementPolicy',(base,),{'arm':arm,'__module__':__name__})
    F.require(result.authorize is base.authorize and result._base_policy is base._base_policy,'settlement_sealed')
    return result


def complete(control: Any, binding: Any, observations: Any) -> Any:
    registration=binding.conditional_firing_registered
    p,state=control.inventory,binding.owner.state
    now=p.S.Clock(**observations[-1]['available_at'])
    identifier=p.digest(dict(origin=asdict(registration['origin']),observations=observations))
    value=p.S.SettlementEvidence(state.scope,registration['origin'],now,p.S.counter_claim(state),identifier)
    proof=dict(kind=KIND,source_token=registration['token'],evidence=asdict(value),observations=observations,
        origin_proof=registration['proof'],conditional_world=True,physical_certified=False,current_permission=False)
    binding.policy.arm('settlement',value,state,proof)
    binding.owner.settle(value,now)
    binding.grid=value.origin.predicted_final
    registration['settled']=True
    registration['settlement_proof']=proof
    assert binding.owner.state.current is state.current and not binding.owner.state.debts
    return dict(stage='conditional_private_settled',frame=now.frame,proof=proof,
        state=asdict(binding.owner.state),current_permission=False,physical_certified=False)
