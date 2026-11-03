"""Yの早い発火から、2テープの評価済み単位を先に返す独立試作。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import zip_longest
from time import perf_counter
from typing import Any,Callable
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import Budget,SearchBudgetExpired
from src.phase_j._live_ex_frozen.src.prefire_search_v411 import BoardMismatch
from src.phase_j._live_ex_frozen.src.prefire_second_v411 import validate
from src.phase_j._live_ex_frozen.src.prefire_second_v412 import align,archive,aggregate,finish,z_candidates,y_candidates
from src.phase_j._live_ex_frozen.src.prefire_second_v410 import evaluate_branches
from src.phase_j._live_ex_frozen.src.prefire_tsumo_v411 import sample_tape
from src.phase_j._live_ex_frozen.src.prefire_zero_v415 import connected
from src.phase_j._live_ex_frozen.src.prefire_paths_v415 import steps

MS=1000.
B3_RESERVE_SEC=.020


def units(native: Any, ctx: Any, tapes: list, budget: Budget, evidence: dict) -> Any:
    if ctx.triggers[ctx.receiver] is not None:
        yield [y_candidates(native,ctx,budget,t)[0] for t in tapes]; return
    zero=connected(native,ctx,evidence)
    if zero is not None:
        yield [{zero.policy:deepcopy(zero)} for _ in tapes]; return
    streams=[steps(native,ctx,t[ctx.receiver],budget) for t in tapes]
    latest=[{},{}]
    for updates in zip_longest(*streams):
        latest=[u if u is not None else old for u,old in zip(updates,latest)]
        yield latest


def evaluate_pair(ctx: Any, ys: list, zs: list, ym: str, zm: str,
                  evaluate: Callable, start: float, deadline: float, records: list) -> dict | None:
    if len(ys)!=2 or len(zs)!=2:
        raise ValueError('2テープ一組だけを評価する')
    if perf_counter()>=deadline:
        raise SearchBudgetExpired('B3開始前の期限')
    unit=[finish(ctx,yp[ym],zp[zm]) for yp,zp in zip(ys,zs)]
    if any(w.invalid for w,_ in unit):
        return None
    values=evaluate_branches([w for w,_ in unit],evaluate)
    elapsed=perf_counter()-start
    if perf_counter()>deadline:
        raise SearchBudgetExpired('B3完了時の期限超過')
    for sample,((world,r),p) in enumerate(zip(unit,values)):
        r.update(sample=sample,y_policy=ym,z_policy=zm,p1=p,
            b3_input=dict(boards=world.audit()['boards'],pending=world.audit()['pending']),
            b3_output=None if any(world.dead) else p)
        records.append(r)
    return dict(elapsed_ms=elapsed*MS,**aggregate(records,ctx.receiver))


def process_y(native: Any, ctx: Any, tapes: list, budget: Budget, pools: list,
              ym: str, evaluate: Callable, start: float, output: dict) -> None:
    zpools=[]
    for index,(t,p) in enumerate(zip(tapes,pools)):
        now=perf_counter(); remaining=budget.deadline-B3_RESERVE_SEC-now
        limited=replace(budget,deadline=now+max(0.,remaining)/(len(tapes)-index))
        zpools.append(z_candidates(native,ctx,limited,t,p[ym]))
    output['candidates'].extend(archive([p for p,_ in zpools],'Z',ym))
    output['cutoff'] |= not all(done for _,done in zpools)
    zs,modes=align([p for p,_ in zpools])
    for zm in modes:
        checkpoint=evaluate_pair(ctx,pools,zs,ym,zm,evaluate,start,budget.deadline,output['branches'])
        if checkpoint:
            output['checkpoints'].append(checkpoint)


def solve(native: Any, ctx: Any, budget: Budget, evaluate: Callable,
          knowledge: dict, evidence: dict) -> dict:
    start=perf_counter(); budget=replace(budget,deadline=min(budget.deadline,start+budget.milliseconds/MS))
    if knowledge['evidence_max']>ctx.input_sec:
        raise ValueError('未来のツモ')
    out=dict(active=False,v_post=None,branches=[],candidates=[],checkpoints=[],cutoff=False,
        knowledge=knowledge,zero_evidence=evidence,receiver=ctx.receiver,attacker=ctx.attacker)
    tapes=[[sample_tape(knowledge['sides'][s],ctx.colors,i,s) for s in (0,1)] for i in range(2)]
    completed=set(); pools=[{},{}]
    try:
        validate(ctx)
        for pools in units(native,ctx,tapes,budget,evidence):
            aligned,modes=align(pools)
            order=sorted((m for m in modes if m not in completed),key=lambda m:
                (min((p[m].fire for p in aligned if p[m].fire is not None),default=float('inf')),m))
            for ym in order:
                budget.check(); process_y(native,ctx,tapes,budget,aligned,ym,evaluate,start,out)
                completed.add(ym)
    except SearchBudgetExpired:
        out['cutoff']=True
    except BoardMismatch as exc:
        out['reason']=str(exc)
    out['candidates']=archive(pools,'Y')+out['candidates']
    if out['branches']:
        out.update(aggregate(out['branches'],ctx.receiver),active=True)
    out.update(complete=not out['cutoff'],reason=out.get('reason',None if out['active'] else 'no_complete_unit'),
        evidence_max=max(ctx.evidence_max,knowledge['evidence_max'],evidence.get('evidence_sec',ctx.input_sec)),
        actual_reply=dict(fired=ctx.triggers[ctx.receiver] is not None,trigger_sec=ctx.triggers[ctx.receiver]))
    return out
