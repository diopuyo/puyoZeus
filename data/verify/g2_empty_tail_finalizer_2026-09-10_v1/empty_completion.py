"""元消費の成功直後の私有状態を保存し、後続の状態を巻き戻さない。"""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import CodeType
from typing import Any

KIND='hidden_empty_tail_next_history/v1'
RECEIPT_KIND='empty_tail_original_commit_return/v1'


def source(binding: Any, record: Any) -> Any:
    return dict(scope=record.scope,frame=record.frame,clock=record.clock,grid=record.grid,
        state=asdict(record.state),proof=record.proof,digest=record.digest,
        source_content=record.source_content,votes_content=record.votes_content,
        policy_content=record.policy_content,head=record.head,token=record.token,
        queue_ref_id=id(record.queue),binding_ref_id=id(binding),basis_ref_id=id(record),
        started=record.started,started_state=asdict(record.started_state),current_permission=False,
        probability_permission=False,evaluation_permission=False,physical_certified=False)


def active_call(factory: Any, call: Any, caller: Any, proof: Any) -> Any:
    control=factory.controller
    rec=control.provider.journal
    assert factory.provider is control.provider and control.calls[id(caller)] is call
    assert call['frame'] is caller and caller.f_code in rec.codes and rec.active['frame'] is caller
    before=[r for r in rec.active['events'] if r['stage']=='fifo_before']
    after=[r for r in rec.active['events'] if r['stage']=='fifo_after']
    assert len(before)==len(after)==1 and not rec.errors
    assert before[0]['fifo_occurrence_tokens']==[proof['token']]
    assert after[0]['enqueue_occurrence_token']==proof['token']
    assert tuple(after[0]['committed'])==tuple(proof['pair'])
    assert after[0]['accounting']['pending_tsumo']==[]
    for key in ('tsumo_count','first_move_sec','landing_pending','last_consumed_color'):
        assert before[0]['accounting'][key]==after[0]['accounting'][key]
    return dict(token=rec.active['token'],events=[before[0],after[0]])


def capture(factory: Any, basis: Any, placement: Any, C: Any, call: Any, caller: Any) -> None:
    control,binding,view=factory.controller,call['binding'],call['view']
    record=basis.validate(control,binding,binding.empty_tail_basis)
    proof=placement.committed(basis,control,binding)
    assert call['consumed'] is True and not view.queue and binding.next_token is None
    assert call['prepared']['private_basis'] is record and view.queue is record.queue
    active=active_call(factory,call,caller,proof)
    state,policy=binding.owner.state,basis.logs(binding)
    payload=dict(kind=RECEIPT_KIND,frame=view.frame,scope=view.scope,basis=source(binding,record),
        state=asdict(state),proof=proof,policy_prefix=policy,active_journal=active,
        consumed=True,current_permission=False,physical_certified=False)
    text=C.encoded(payload)
    receipt=C.Receipt(view.frame,text,sha256(text.encode()).hexdigest())
    key=(view.frame,view.scope[-1])
    assert key not in control.empty_completion_refs
    control.empty_completion_rows.append(receipt)
    control.empty_completion_refs[key]=(factory,binding,record,state,proof,policy,call,caller)


def producer_code(reuse: Any, placement: Any) -> None:
    """既検証の限定AST派生と実consumed本体を一回照合する。"""
    assert reuse is not None
    path=Path(placement.__file__).resolve()
    tree,_=reuse.transformed(path.name,path.read_bytes())
    code=compile(tree,str(path),'exec',dont_inherit=True)
    expected=next(c for c in code.co_consts if isinstance(c,CodeType) and c.co_name=='consumed')
    assert placement.consumed.__globals__ is vars(placement) and placement.consumed.__code__==expected


def install(stack: Any, factory: Any, basis: Any, placement: Any, patch: Any, C: Any,
            write: Any, reuse: Any=None) -> None:
    control=factory.controller
    assert not hasattr(control,'empty_completion_rows')
    assert placement.KIND==KIND and placement.empty_tail_derivation['renamed']
    producer_code(reuse,placement)
    control.empty_completion_rows,control.empty_completion_refs=[],{}
    def save() -> None:
        write(control.provider.journal.output/'EMPTY_COMPLETION.json',
            dict(rows=[asdict(r) for r in control.empty_completion_rows],physical_certified=False))
    stack.callback(save)
    original=placement.consumed
    assert original.__globals__ is vars(placement)
    def consumed(actual: Any, who: Any, call: Any, caller: Any, rows: Any) -> Any:
        result=original(actual,who,call,caller,rows)
        if who is control:
            assert actual is basis
            capture(factory,basis,placement,C,call,caller)
        return result
    patch(stack,placement,'consumed',consumed)


def related(C: Any, control: Any, binding: Any, record: Any) -> None:
    events=[r for r in control.empty_tail_basis_events if r['binding_id']==id(binding)
        and (r.get('digest')==record.digest or r.get('old_tail_digest')==record.digest)]
    assert len(events)==2
    first,last=events
    assert first['kind']=='empty_tail_captured/v1' and first['frame']==record.frame
    assert first['head'] is first['token'] is None
    assert C.encoded(first['state'])==C.encoded(asdict(record.state))
    assert C.encoded(first['proof'])==C.encoded(record.proof)
    assert first['source_content']==record.source_content and first['votes_content']==record.votes_content
    assert last['kind']=='empty_tail_new_head_started/v1' and last['frame']==record.started[0]
    assert last['clock']==record.started[1] and last['token']==record.token
    assert C.encoded(last['head'])==C.encoded(record.head)
    assert C.encoded(last['state'])==C.encoded(asdict(record.started_state))
    assert last['bound_content']==record.bound_content


def verify(factory: Any, C: Any, journal: Any, history: Any, step: Any) -> Any:
    control=factory.controller
    assert factory.provider is control.provider and not control.calls and not control.tickets
    prepared={(r['scope']['frame_idx'],r['scope']['side']):r for r in history
        if r['prepared'] and r['prepared']['kind']==KIND}
    found=set()
    for receipt in control.empty_completion_rows:
        assert type(receipt) is C.Receipt and sha256(receipt.payload_json.encode()).hexdigest()==receipt.sha256
        value=json.loads(receipt.payload_json)
        key=(receipt.frame,value['scope'][-1])
        assert key not in found and key in prepared and value['kind']==RECEIPT_KIND
        owned,binding,record,state,proof,policy,call,caller=control.empty_completion_refs[key]
        assert type(value['frame']) is type(receipt.frame) is int and value['frame']==receipt.frame
        assert C.encoded(value['scope'])==C.encoded(record.scope)
        # 原Controller.finishは正常消費後にもframe参照をNoneへ片付ける。
        assert owned is factory and call['frame'] is None and call['consumed'] is True
        assert C.encoded(value['basis'])==C.encoded(source(binding,record))
        assert C.encoded(value['proof'])==C.encoded(proof)==C.encoded(prepared[key]['prepared'])
        assert C.encoded(value['state'])==C.encoded(asdict(state))==C.encoded(prepared[key]['decision']['history_state'])
        assert C.encoded(value['policy_prefix'])==C.encoded(policy[:len(value['policy_prefix'])])
        current_policy=getattr(binding.policy,'accounting',binding.policy).proofs
        assert C.encoded(value['policy_prefix'])==C.encoded(current_policy[:len(value['policy_prefix'])])
        current=binding.owner.state
        assert control.history[key[-1]] is binding and current.scope==state.scope
        assert current.history[:len(state.history)]==state.history
        control.inventory.S.validate_accounting(state)
        assert state.counter==control.inventory.S.color_counts(tuple(map(tuple,proof['grid'])))
        assert value['consumed'] is True and value['current_permission'] is value['physical_certified'] is False
        related(C,control,binding,record)
        for frame in (record.frame,record.started[0],receipt.frame): step(journal,frame,record.scope)
        actual=step(journal,receipt.frame,record.scope)
        assert C.encoded(value['active_journal'].get('token'))==C.encoded(actual['token'])
        events=[r for r in actual['events'] if r['stage'] in ('fifo_before','fifo_after')]
        assert C.encoded(events)==C.encoded(value['active_journal']['events'])
        checks=[r for r in control.empty_tail_prepop_checks if r['frame']==receipt.frame and r['token']==record.token]
        assert len(checks)==1 and checks[0]['event_id']==control.inventory.digest(proof)
        found.add(key)
    assert found==set(prepared) and len(control.empty_tail_basis_events)==len(found)*2
    return dict(empty_commit_samecall_verified=True,commits=len(found),later_state_reinterpreted=False,
        physical_certified=False,quality_gate_clear=False)
