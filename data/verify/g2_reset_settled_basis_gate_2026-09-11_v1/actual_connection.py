"""原Recoveryの同call入力を読み、別確率基準の候補を記録する。整数復帰は変更しない。"""
from __future__ import annotations
from dataclasses import asdict
import json
from types import MethodType
from typing import Any
import gate_v2 as G


def grid(value: Any) -> Any:
    return None if value is None else tuple(tuple(row) for row in value['grid'])


def probabilities(channels: Any) -> tuple[Any,Any]:
    pb=channels['probability']
    cells=None if pb is None else pb.get('cells')
    if cells is None: return None,None
    dist=lambda r,c:tuple((k,float(p)) for k,p in cells[r][c] if p>0)
    return (tuple(dist(r,c) for r in range(1,13) for c in range(6)),
            tuple(dist(0,c) for c in range(6)))


class Observer:
    def __init__(self,recovery: Any,reset_frame: int,deadline: int) -> None:
        self.recovery=recovery
        scope=recovery.evidence.scope(recovery.factory,recovery.pipe)
        self.gate=G.SettledBasisGate(scope,reset_frame,deadline)
        self.error: str|None=None

    def capture(self,item: Any,result: Any,error: Any) -> None:
        r=self.recovery
        saved=r.waits.get(id(item['frame']))
        if saved is None: return
        frame=item['frame']
        G.V1.require(error is None and saved['item'] is item and saved['caller'] is frame,'actual_saved_call')
        G.V1.require(frame.f_code in r.journal.codes and r.journal.active is None and item['pipe'] is r.pipe,'actual_J_identity')
        local=frame.f_locals
        scope=r.evidence.scope(r.factory,r.pipe)
        G.V1.require(scope==saved['scope'] and item['epoch']==scope[2],'actual_scope_epoch')
        view=saved['view']
        fallback=type('View',(),dict(frame=saved['frame'],clock=saved['clock']))()
        raw,_=r.provider.raw(r.pipe,'1P',view or fallback)
        typed=r.state['postcommit_current_receiver'].rec.side_value(result)
        visible,hidden=probabilities(typed)
        signals=local['signals']
        grace=r.pipe._landing_grace_1p
        present=typed['confirmed'] is not None
        obs=G.V1.CallObservation(item['token'],saved['frame'],scope,scope[2],scope[5],typed['state_value'],
            None,signals.is_match_active,signals.effect_gate_window_active,
            True if view is None else not r.provider.no_origin(r.pipe,'1P',view),
            grace is None or saved['clock']>=grace[2],G.V1.ACTUAL_CALL_STAGE,present,
            tuple(tuple(row) for row in raw),grid(typed['cnn']),
            None if local['sm'].context.confirmed_board is None else tuple(map(tuple,local['sm'].context.confirmed_board._grid.tolist())),
            grid(typed['confirmed']),visible,hidden)
        self.gate.observe(obs)


def install(stack: Any,recovery: Any,state: dict[str,Any],reset_frame: int,deadline: int) -> Observer:
    G.V1.require('settled_basis_observer' not in state,'duplicate_actual_basis')
    value=Observer(recovery,reset_frame,deadline)
    state['settled_basis_observer']=value
    original=recovery.complete
    existed='complete' in vars(recovery)
    stored=vars(recovery).get('complete')
    def complete(self: Any,item: Any,result: Any,error: Any) -> Any:
        failure=None
        try: value.capture(item,result,error)
        except BaseException as caught: failure=caught; value.error=repr(caught)
        returned=original(item,result,error)
        if failure is not None: raise failure
        return returned
    def close() -> None:
        setattr(recovery,'complete',stored) if existed else vars(recovery).pop('complete',None)
        candidate=value.gate.candidate
        result=dict(error=value.error,rows=value.gate.rows,
            candidate=None if candidate is None else asdict(candidate),
            restored=('complete' in vars(recovery))==existed,actual_factory_observer=True,
            original_integer_recovery_unchanged=True,quality_gate_clear=False)
        with (state['output']/'SETTLED_BASIS_OBSERVER.json').open('x',encoding='utf-8') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    stack.callback(close)
    recovery.complete=MethodType(complete,recovery)
    return value
