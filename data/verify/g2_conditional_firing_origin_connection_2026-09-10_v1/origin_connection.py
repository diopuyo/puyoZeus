"""一回private originを接続し、未精算の確定候補公開は保留する。"""
from __future__ import annotations
from dataclasses import asdict
from types import FunctionType
from typing import Any
import inspect
import commit as C
import origin_policy_v2 as P
import origin_registration as R


def pending(control: Any, call: Any, pipe: Any) -> Any:
    binding=call['binding']; saved=binding.conditional_firing_registered
    state,origin=binding.owner.state,saved['origin']
    assert call['consumed'] is True and call['view'].scope==saved['scope']==binding.scope
    assert call['view'].frame==origin.available_at.frame and state.clock==origin.available_at
    assert len(state.debts)==1 and state.debts[0].origin==origin and state.origins==(origin,)
    assert state.counter==control.inventory.S.color_counts(origin.before_grid)
    assert saved['token'] in binding.consumed_tokens and binding.next_token is None
    assert not pipe._pending_tsumo_1p and pipe._active_chain_1p is None
    assert binding.grid==origin.before_grid and origin.event_identity.startswith('conditional_world:')
    control.unchanged(call,pipe,'1P')
    return saved


def withhold(stack: Any, factory: Any, pipe: Any, patch: Any, rows: list[Any]) -> None:
    import hidden_bundle as B
    module=B.CURRENT.C.C
    original=module.make
    def make(control: Any, call: Any, result: Any, provisional: Any) -> Any:
        if getattr(call['binding'],'conditional_firing_registered',None) is None:
            return original(control,call,result,provisional)
        saved=pending(control,call,pipe)
        rows.append(dict(stage='conditional_origin_current_withheld',frame=call['view'].frame,
            origin_id=saved['origin'].origin_id,current_permission=False,physical_certified=False))
        return None
    patch(stack,module,'make',make)


def install(stack: Any, factory: Any, pipe: Any, patch: Any, rows: list[Any]) -> None:
    control=factory.controller
    assert not control.history
    accounting=inspect.getclosurevars(control.inventory.BoundPolicy.__init__).nonlocals['p']
    patch(stack,accounting,'BoundPolicy',P.policy_type(accounting))
    def consumed(control: Any, call: Any, caller: Any, rows: list[Any]) -> None:
        C.consumed(control,call,caller,rows)
        binding=call['binding']
        origin,proof=R.register(control,binding,call)
        binding.conditional_firing_registered=dict(origin=origin,proof=proof,
            scope=binding.scope,token=proof['source_token'])
        rows.append(dict(stage='conditional_origin_registered',frame=call['view'].frame,
            origin=asdict(origin),proof=proof,state=asdict(binding.owner.state),
            original_event=False,physical_certified=False,current_permission=False))
    FunctionType(C.install.__code__,dict(vars(C),consumed=consumed))(stack,factory,pipe,patch,rows)
    withhold(stack,factory,pipe,patch,rows)
