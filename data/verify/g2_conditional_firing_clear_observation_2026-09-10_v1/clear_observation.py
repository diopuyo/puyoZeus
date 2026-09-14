"""条件付き消去の実raw二票。原SMのSTABLEを捏造せず、writerも呼ばない。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import sys

KIND='conditional_clear_raw/v1'
FPS,STRIDE,MIN_VOTES,OBSERVED_OPERATION,AVAILABLE_OPERATION,SCOPE_FIELDS=60,2,2,0,1,7


def identity(control: Any, binding: Any, view: Any, registration: Any) -> None:
    import path_support as P
    state,origin=binding.owner.state,registration['origin']
    P.require(type(view.scope) is tuple and len(view.scope)==SCOPE_FIELDS
        and all(type(view.scope[k]) is int for k in (2,3,4,5)),'conditional_clear_scope_types')
    P.require(registration['scope']==binding.scope==view.scope,'conditional_clear_scope')
    P.require(type(origin) is control.inventory.S.OriginEvidence and origin.scope==state.scope
        and origin.action==state.action and origin in state.origins,'conditional_clear_origin')
    P.require(origin.event_identity.startswith('conditional_world:') and len(state.debts)==1
        and state.debts[0].origin==origin and origin.origin_id not in state.consumed_ids,'conditional_clear_debt')
    P.require(state.counter==control.inventory.S.color_counts(origin.before_grid)
        and binding.grid==origin.before_grid,'conditional_clear_inventory')
    P.require(registration['token'] in binding.consumed_tokens and binding.next_token is None
        and not view.queue and not view.refs and not view.tokens,'conditional_clear_pending')
    P.require(type(view.frame) is int and view.frame>origin.available_at.frame
        and type(view.clock) is float and view.clock==view.frame/FPS,'conditional_clear_clock')


def capture(control: Any, binding: Any, view: Any, signals: Any, sm: Any, pipe: Any) -> Any:
    import path_support as P
    registration=binding.conditional_firing_registered
    identity(control,binding,view,registration)
    if (signals.is_match_active is not True or signals.effect_gate_window_active is not False
        or signals.chain_event is not None or view.quiet is not True or view.added): return None
    live=control.provider.link.current(view)
    P.require(live['pipe'] is pipe and getattr(pipe,'_sm_'+view.scope[-1].lower()) is sm,'conditional_clear_live')
    P.require(view.scope[3]==id(pipe) and view.scope[4]==id(sm),'conditional_clear_object_scope')
    if live['accepted']!=view.next_pair or live['dnext']!=view.dnext_pair: return None
    P.require(getattr(pipe,'_active_chain_'+view.scope[-1].lower()) is None,'conditional_clear_actual_chain')
    raw,proof=control.provider.raw(pipe,view.scope[-1],view)
    P.require(proof['captured_frame']==view.frame and proof['raw_grid']==raw,'conditional_clear_capture')
    if raw!=control._parts.T.board_key(signals.cnn_board): return None
    raw=P.grid(raw,hidden_unknown=True)
    final=registration['origin'].predicted_final
    if not P.compatible(raw,final) or raw[P.HIDDEN_ROWS:]!=final[P.HIDDEN_ROWS:]: return None
    P.require(raw!=registration['origin'].before_grid,'conditional_clear_not_placed')
    clock=control._parts.C.H.clock
    p=control.inventory
    observed=clock(p,view.frame,view.clock,OBSERVED_OPERATION)
    available=clock(p,view.frame,view.clock,AVAILABLE_OPERATION)
    value=dict(kind=KIND,scope=asdict(binding.owner.state.scope),live_scope=view.scope,
        source_token=registration['token'],origin_id=registration['origin'].origin_id,action=binding.owner.state.action,
        observed_at=asdict(observed),available_at=asdict(available),raw_grid=raw,raw_capture=proof,
        capture_ref=p.digest(dict(capture=proof,scope=view.scope)),state=sm.context.state.value,
        effect_window=False,active=True,actual_chain=False,pending_next=False,
        observed_next=view.next_pair,observed_dnext=view.dnext_pair,quiet=True,
        physical_certified=False,current_permission=False,conditional_world=True)
    return value


def observe(control: Any, binding: Any, view: Any, signals: Any, sm: Any, pipe: Any, rows: list[Any]) -> Any:
    value=capture(control,binding,view,signals,sm,pipe)
    if value is None:
        binding.conditional_clear_observations=()
        return None
    previous=getattr(binding,'conditional_clear_observations',())
    if previous and (previous[-1]['observed_at']['frame']+STRIDE!=view.frame
        or previous[-1]['source_token']!=value['source_token'] or previous[-1]['origin_id']!=value['origin_id']
        or previous[-1]['live_scope']!=value['live_scope'] or previous[-1]['raw_grid']!=value['raw_grid']): previous=()
    saved=tuple((*previous,value)[-MIN_VOTES:])
    binding.conditional_clear_observations=saved
    rows.append(dict(stage='conditional_final_raw_observed',frame=view.frame,count=len(saved),
        observation=value,writer_called=False,physical_certified=False))
    if len(saved)!=MIN_VOTES: return None
    assert saved[0]['capture_ref']!=saved[1]['capture_ref']
    return saved


def install(stack: Any, factory: Any, patch: Any, rows: list[Any]) -> None:
    control=factory.controller
    cls=type(control)
    original=cls.observed
    def observed(self: Any, sm: Any, signals: Any, view: Any, pipe: Any, side: str) -> Any:
        caller=sys._getframe(1)
        assert caller.f_code is self._parts.C.Controller.update.__code__
        assert caller.f_locals['signals'] is signals and caller.f_locals['pipe'] is pipe
        raw=original(self,sm,signals,view,pipe,side)
        binding=self.history.get(side)
        if binding is not None and getattr(binding,'conditional_firing_registered',None) is not None:
            observe(self,binding,view,signals,sm,pipe,rows)
        return raw
    patch(stack,cls,'observed',observed)
