"""原65の後の人工BB発火を原collectorへ通す。期待拒否もexit1で保存する。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict
from types import FunctionType,SimpleNamespace as N
from typing import Any
import sys
import traceback
import fire_inputs as I
import fire_score as F
import front_trace as T

H,R,K=I.H,I.H.R,I.H.B.K
ROOT=I.ROOT


def loop(args: Any, stack: Any, data: Any) -> None:
    q,m,state,real,fixture,binding,factory=args
    pipe,rec=real[:2]
    K.write(state['output']/'ARTIFICIAL_INITIAL.json',I.initial_fixture(pipe,factory))
    collector=m.loop.collector_loop()
    collector.RecognitionPipeline=type(pipe)
    pixels=binding['provider']
    cap=m.smoke.SyntheticCapture(real,fixture,collector.cv2,pixels.o.np)
    clock={'frame':I.FRAMES[0]}
    data['supplied']=I.install(q,stack,pipe,pixels,factory.types.parts.O,clock)
    H.C.install(stack,factory,pipe,state,data['firing'])
    import firing_handoff_fixed as A
    data['qualification']={}
    stack.enter_context(A.V5.installed(pipe,data['qualification']))
    A.verify(pipe)
    T.install(stack,pipe,factory,data['front'])
    sink=m.recording.install(stack,state,collector,fixture.subject.base)
    raw=m.raw.install(stack,collector,pixels,rec.emit)
    for frame in I.FRAMES:
        clock['frame'],cap.position=frame,frame
        if frame==I.NEXT: F.install(stack,pipe,clock)
        F.paint(cap,frame)
        try:
            collector.collect_lean(cap,pipe,frame,1,I.STRIDE,I.FPS)
            owner=factory.controller.history.get('1P')
            data['trace'].append(dict(frame=frame,state=pipe._sm_1p.context.state.value,
                owner=None if owner is None else asdict(owner.owner.state)))
            if frame==I.OLD.FRAMES[-1]:
                data['prefix65']=H.B.HR.verify(state,factory,pipe,factory.controller.hidden_history_rows,data['trace'])
        except BaseException as error:
            data['primary']=dict(frame=frame,type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
            raise
    data['unexpected_completion']=True


def drive(*args: Any) -> Any:
    state,pipe,factory=args[2],args[3][0],args[6]
    profile=sys.getprofile()
    data=dict(trace=[],front=[],firing=[],primary=None,fixture=I.generated(),physical_certified=False)
    try:
        with ExitStack() as stack: loop(args,stack,data)
        raise AssertionError('expected_hidden_firing_entry_rejection_not_seen')
    finally:
        data['profile_restored']=sys.getprofile() is profile
        data['score_refs_restored']=all(getattr(pipe,n) is None for n in
            ('_score_ocr','_score_tracker_1p','_score_tracker_2p'))
        if 'combined_restored' in state: state['combined_restored']()
        data['common_refs_restored']=True
        H.C.saved(state,factory)
        K.write(state['output']/'ENTRY_DIAGNOSTIC.json',data)


def load(alias: str, path: Any, stack: Any) -> Any:
    module=K.load(alias,path,stack)
    if alias=='_hidden_prefix_full_parent':
        module.palette_fixture=N(configured=F.configured)
        module.execute=FunctionType(H.B.CURRENT.palette_fixture.execute.__code__,
            dict(vars(module)))
    return module


def guards() -> Any:
    return H.C.guards()|{str(p):K.sha(p) for p in
        (I.SAVED,I.VERIFY/'g2_chain_firing_continuous_cpu_2026-09-10_v1/artificial_inputs.py',
         I.VERIFY/'g2_chain_firing_continuous_cpu_2026-09-10_v1/constructor_input.py')}


def main() -> int:
    helper=N(**(vars(K)|dict(guards=guards,load=load)))
    return FunctionType(R.main.__code__,dict(vars(R),ROOT=ROOT,K=helper,I=I,drive=drive))()


if __name__=='__main__': raise SystemExit(main())
