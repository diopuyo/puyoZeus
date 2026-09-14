"""原47updateで先頭履歴だけの消費を検収。後続current完成とは別の中間工程。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import FunctionType
from typing import Any
import ast
import os
import shutil
import sys
import time
import traceback
import prefix_inputs as I
import prefix_connection as C

ROOT, VERIFY = Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent
PARENT = VERIFY/'g2_reset_recovery_candidate_2026-09-10_v1'
RUNTIME = VERIFY/'g2_history_publication_probe_runtime_2026-09-10_v13'
DEFERRED = VERIFY/'g2_deferred_next_handoff_2026-09-10_v1'
sys.path[:0] = [str(RUNTIME),str(PARENT)]
import common as K


def verify(state: Any, factory: Any, pipe: Any, records: Any, trace: Any) -> dict[str,Any]:
    binding = factory.controller.history['1P']
    private,base = binding.owner.state,I.saved()[0]
    assert len(trace)==len(I.FRAMES) and len(records)==1, 'prefix_not_once'
    assert sum(private.counter)==63 and len(private.history)==1, 'prefix_inventory'
    assert [list(pair) for pair in pipe._pending_tsumo_1p]==[list(I.BB)], 'prefix_FIFO'
    assert private.current is None and binding.current==tuple(map(tuple,base)), 'prefix_current'
    assert factory.controller._parts.T.board_key(pipe._sm_1p.context.confirmed_board)==binding.current
    assert not pipe._tsumo_count_1p and not pipe._tsumo_count_2p, 'prefix_native_counter'
    receiver,consumer = state['postcommit_current_receiver'],state['postcommit_publication_consumer']
    assert receiver.issued==receiver.released==0 and not receiver.errors, 'prefix_publication'
    assert len(consumer.rows)==len(I.FRAMES) and not consumer.errors, 'prefix_outer'
    assert all(not row['changed_sides'] and row['full_before']==row['full_after'] for row in consumer.rows)
    assert factory.provider.journal.steps==len(I.FRAMES)*2 and not factory.provider.journal.errors
    assert not factory.controller.sticky_error and not factory.provider.handoff_proofs
    return dict(completed_updates=len(trace),prefix_frame=records[0]['frame'],prefix_counter=63,
        retained_current_count=61,remaining_FIFO=[list(I.BB)],original_J_pop_once=True,
        new_current_permission=False,issued=0,released=0,second_hand_connected=False,
        integer_current_recovered=False,physical_certified=False,quality_gate_clear=False)


def install(stack: Any, factory: Any, records: Any) -> None:
    previous = list(sys.path)
    stack.callback(lambda:sys.path.__setitem__(slice(None),previous))
    sys.path.insert(0,str(DEFERRED))
    base = K.load('_hidden_prefix_original_deferred',DEFERRED/'connection.py',stack)
    identity = K.load('_hidden_prefix_identity',DEFERRED/'connection_v2.py',stack)
    held = K.load('_hidden_prefix_held_v2',DEFERRED/'held_v2.py',stack)
    base.patch(stack,base.D,'check',held.check)
    stack.enter_context(identity.installed(base,factory.controller))
    C.install(stack,factory,base,records)


def drive(q: Any, m: Any, state: Any, real: Any, fixture: Any, binding: Any, factory: Any) -> Any:
    pipe,rec = real[:2]
    trace,records,result = [],[],None
    cls = type(factory.controller)
    refs = cls.prepared,cls.call,cls.consumed_history,factory.provider.after_history_consume
    try:
        with ExitStack() as stack:
            initial = I.initial_fixture(pipe,factory)
            K.write(state['output']/'ARTIFICIAL_INITIAL.json',initial)
            collector = m.loop.collector_loop()
            collector.RecognitionPipeline = type(pipe)
            pixels = binding['provider']
            cap = m.smoke.SyntheticCapture(real,fixture,collector.cv2,pixels.o.np)
            clock = {'frame':I.FRAMES[0]}
            supplied = I.install(q,stack,pipe,pixels,factory.types.parts.O,clock)
            install(stack,factory,records)
            sink = m.recording.install(stack,state,collector,fixture.subject.base)
            raw = m.raw.install(stack,collector,pixels,rec.emit)
            for frame in I.FRAMES:
                clock['frame'],cap.position = frame,frame
                collector.collect_lean(cap,pipe,frame,1,I.STRIDE,I.FPS)
                if frame<I.FIRST:
                    assert factory.controller._parts.T.board_key(pipe._sm_1p.context.confirmed_board)==tuple(map(tuple,initial['initial_confirmed'])), 'initial_fixture_changed'
                owner = factory.controller.history.get('1P')
                trace.append(dict(frame=frame,state=pipe._sm_1p.context.state.value,
                    owner=None if owner is None else asdict(owner.owner.state)))
            result = verify(state,factory,pipe,records,trace) | dict(supplied=supplied)
        assert refs==(cls.prepared,cls.call,cls.consumed_history,factory.provider.after_history_consume)
        assert sink.closed and not sink.errors and raw.closed and raw.error is None
        return result | dict(references_restored=True)
    finally:
        K.write(state['output']/'PREFIX_HISTORY.json',fixture.subject.base.json_value(dict(rows=records,
            trace=trace,restored=refs==(cls.prepared,cls.call,cls.consumed_history,factory.provider.after_history_consume))))


def main() -> int:
    output = ROOT/sys.argv[1]
    assert output.parent==ROOT and output.name.startswith('prefix_cpu_v')
    output.mkdir(exist_ok=False)
    paths = [*ROOT.glob('*.py'),ROOT/'PLAN.md',I.LIVE/'directional_history.jsonl',PARENT/'run_cpu.py']
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for path in paths:
        if path.parent==ROOT: shutil.copy2(path,snapshot/path.name)
    guards = K.guards() | {str(p):K.sha(p) for p in paths}
    started,code,result = time.perf_counter(),0,{}
    try:
        for path in ROOT.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_bytes())):
                if isinstance(node,ast.FunctionDef):
                    assert node.returns is not None and node.end_lineno-node.lineno+1<=50
        import cv2
        import torch
        cv2.setNumThreads(2)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        with ExitStack() as stack:
            parent = K.load('_hidden_prefix_full_parent',PARENT/'run_cpu.py',stack)
            execute = FunctionType(parent.execute.__code__,dict(vars(parent),I=I,drive=drive))
            result = execute(output)
        assert not torch.cuda.is_initialized()
    except BaseException:
        code,result = 1,dict(error=traceback.format_exc())
    equal = all(K.sha(Path(p))==h for p,h in guards.items())
    result.update(exit_code=code,pid=os.getpid(),seconds=time.perf_counter()-started,
        guards=guards,guards_unchanged=equal,gpu=False,quality_gate_clear=False)
    K.write(output/'RESULT.json',result)
    K.write(output/'INDEX.json',{str(p.relative_to(output)):K.sha(p) for p in output.rglob('*') if p.is_file()})
    print({k:v for k,v in result.items() if k not in ('guards','supplied')},flush=True)
    return code or int(not equal)


if __name__ == '__main__':
    raise SystemExit(main())
