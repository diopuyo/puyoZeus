"""原発火codeだけをread-only profileで採録する。別codeのlocalsは読まない。"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import sys
from typing import Any
import fire_inputs as I


def capture(frame: Any, factory: Any) -> Any:
    values=frame.f_locals
    binding=factory.controller.history['1P']
    pipe=frame.f_back.f_locals['self'] if 'pipe' not in values else values['pipe']
    key=frame.f_globals['key']
    queue,owner=values['queue'],values['owner']
    caller=values['caller']
    assert owner['queue'] is queue and len(queue)==len(owner['tokens'])==len(owner['refs'])==1
    assert owner['refs'][0] is queue[0] is binding.candidate.pair
    assert owner['tokens'][0]==binding.next_token and binding.next_token not in binding.consumed_tokens
    anchor=binding.hidden_anchor
    assert I.H.B.LIFE.L.L.compatible(binding.owner.state,binding)
    identity=dict(scope=binding.scope,binding_id=id(binding),pipe_id=id(pipe),
        sm_id=id(pipe._sm_1p),queue_id=id(queue),head_id=id(queue[0]),
        tokens=list(owner['tokens']),owner_epoch=owner['epoch'],next_token=binding.next_token,
        next_started=binding.next_started,candidate_pair=list(binding.candidate.pair),
        anchor=asdict(anchor),anchor_id=id(anchor),candidate_id=id(binding.candidate),
        clear_grid=binding.clear_grid,clear_first=binding.clear_first,clear_last=binding.clear_last,
        clear_count=binding.clear_count,frame=values['clock'],clock=caller.f_locals['time_sec'],
        caller_code_sha256=hashlib.sha256(caller.f_code.co_code).hexdigest(),
        caller_file=caller.f_code.co_filename,caller_name=caller.f_code.co_name,
        observed=key(values['observed']),original_J_head_identity_checked=True,
        original_anchor_correspondence_checked=True,not_after_original_J_cleanup=True)
    return dict(line=frame.f_lineno,current=binding.current,inferred=binding.grid,
        before=key(values.get('before')),raw=key(values.get('raw')),
        owner=asdict(binding.owner.state),queue=[list(v) for v in pipe._pending_tsumo_1p],
        ticket_present=getattr(binding,'firing_ticket',None) is not None,
        native_counter=dict(pipe._tsumo_count_1p),permission=False,identity=identity)


def install(stack: Any, pipe: Any, factory: Any, rows: list[Any]) -> None:
    import front_probe as F
    import firing_ticket as T
    code=F.capture.__code__
    codes={code:'front_capture',T.qualified.__code__:'v5_qualified',
        pipe._apply_chain_formula_early_fire.__func__.__code__:'handoff_formula'}
    previous=sys.getprofile()
    def profile(frame: Any, event: str, returned: Any) -> None:
        if previous is not None: previous(frame,event,returned)
        if frame.f_code not in codes or event not in ('call','return'): return
        row=dict(stage=codes[frame.f_code],event=event,file=frame.f_code.co_filename,line=frame.f_lineno)
        if frame.f_code is code and event=='return': row['actual']=capture(frame,factory)
        rows.append(row)
    sys.setprofile(profile)
    stack.callback(sys.setprofile,previous)
