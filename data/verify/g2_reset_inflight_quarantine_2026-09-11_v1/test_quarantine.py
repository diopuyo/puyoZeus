"""原連続着地分岐のCPU接続。J/初期SMは人工fixture、実factory合格ではない。"""
from contextlib import ExitStack
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
import pytest
from quarantine import Guard,install

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'g2_midmatch_side_resync_candidate_2026-09-11_v1'))
import check_candidate as Q
SOURCE=ROOT.parents[2]/'.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/recognition_pipeline.py'
SHA='6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02'


def setup() -> tuple[dict,N,list]:
    pipe,_,_=Q.midmatch_pipe()
    saved=json.loads((ROOT.parent/'video38_history_publication_probe_live_2026-09-11_v12/LIVE_EMPTY_RESET.json').read_bytes())
    waits={r['frame']:r for r in saved['recovery'] if r['kind']=='reset_baseline_wait'}
    prev=Q.Board.from_dict({'grid':waits[35164]['raw']})
    raw=Q.Board.from_dict({'grid':waits[35172]['raw']})
    pipe._sm_1p.context.confirmed_board=raw.copy()
    scope=tuple(waits[35172]['scope'])
    old=(*scope[:2],scope[2]-1,*scope[3:5],scope[5]-1,scope[6])
    journal=N(active=None,codes=set(),errors=[])
    recovery=N(pipe=pipe,pending=dict(epoch=scope[2],old_scope=old,used=False),journal=journal,
               evidence=N(scope=lambda *_:scope),factory=object(),error=None)
    ns=dict(sys.modules[Q.repro.RecognitionPipeline.__module__].__dict__)
    ns.update(sys=sys,self=pipe,side='1P',frame_idx=35172,time_sec=35172/60,prev_state=Q.BoardState.TSUMO_FALL,
              ctx=pipe._sm_1p.context,prev_confirmed=prev,cnn_board=raw,prev_next_queue=[(4,5)],
              _skip_infer_by_ojama_guard=False,frame_bgr=None,side_prob_board=None,next_pair=(4,5))
    return ns,recovery,waits[35172]['raw']


def block() -> object:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    tree=ast.parse(SOURCE.read_bytes())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.If) and n.lineno==7426]
    assert len(nodes)==1
    entry=ast.parse('REGISTER(sys._getframe())').body
    return compile(ast.fix_missing_locations(ast.Module(body=entry+nodes,type_ignores=[])),str(SOURCE)+'#7426','exec')


def test_original_landing_block_preserves_raw_uncertainty() -> None:
    assert not Q.CV2_STUBBED
    ns,recovery,raw=setup()
    code=block()
    recovery.journal.codes.add(code)
    ns['REGISTER']=lambda frame:setattr(recovery.journal,'active',dict(frame=frame,pipe=recovery.pipe,token='fixture-J'))
    old=ns['ctx'].confirmed_board
    counter=dict(recovery.pipe._tsumo_count_1p)
    other=recovery.pipe._sm_2p.context
    original=ns['infer_placement']
    with ExitStack() as stack:
        guard=install(stack,recovery,ns)
        exec(code,ns)
        assert ns['inferred_landing'] is None and ns['side_prob_board'] is None
        assert ns['ctx'].confirmed_board is old and old.to_dict()['grid']==raw
        assert len(guard.rows)==1 and guard.rows[0]['passed_pair'] is None
        assert guard.rows[0]['original_pair']==(4,5)
    assert ns['infer_placement'] is original and dict(recovery.pipe._tsumo_count_1p)==counter
    assert recovery.pipe._sm_2p.context is other


@pytest.mark.parametrize('case',('wrong_code','wrong_pipe','used','new_fifo','prior_error'))
def test_bad_waiting_context_refused(case: str) -> None:
    ns,recovery,_=setup()
    code=block()
    if case!='wrong_code': recovery.journal.codes.add(code)
    ns['REGISTER']=lambda frame:setattr(recovery.journal,'active',dict(frame=frame,pipe=recovery.pipe,token='fixture-J'))
    if case=='wrong_pipe': ns['self']=Q.repro.build()
    elif case=='used': recovery.pending['used']=True
    elif case=='new_fifo': recovery.pipe._pending_tsumo_1p.append((4,5))
    elif case=='prior_error': recovery.error='prior'
    with ExitStack() as stack:
        install(stack,recovery,ns)
        with pytest.raises(RuntimeError,match='reset_inflight_quarantine'): exec(code,ns)


def test_nonwaiting_original_call_is_preserved() -> None:
    calls=[]
    guard=Guard(N(pending=None),lambda *a,**kw:calls.append((a,kw)))
    guard.infer('before','after',(4,5),marker=True)
    assert calls==[(('before','after',(4,5)),{'marker':True})] and guard.rows==[]
