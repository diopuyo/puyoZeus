"""A38実完了prefixを新consumerへ接続し、人工未来tailで保存契約を検査。原exit1は保持。"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import second_inactive_saved as NEW
import terminal_boundary as C

ROOT = Path(__file__).resolve().parent
VERSION = sys.argv[1] if len(sys.argv)>1 else 'v38'
if VERSION not in ('v38','v39'):
    raise ValueError('invalid CPU runtime')
sys.path.insert(0, str(ROOT.parent/('g2_second_prefix_runtime_2026-09-14_' + VERSION)))
import postrun as P


def decode(module: object, value: dict) -> object:
    scope = tuple(value['scope'])
    arrivals = tuple(module.Arrival(scope,a['token'],tuple(a['pair']),a['frame'],a['call_token']) for a in value['arrivals'])
    acks = tuple(module.Ack(a['token'],a['frame'],a['call_token']) for a in value['acknowledgements'])
    return module.Ledger(scope,value['start'],value['deadline'],value['clock'],arrivals,tuple(value['applied']),acks)


def synthetic_tail(module: object, ledger: object, source: dict, steps: dict, contexts: dict) -> tuple:
    """未観測未来は人工トークンで明示。原A38ファイルへは一切書かない。"""
    packets, scope = [], ledger.scope
    unacked = ledger.arrivals[len(ledger.acknowledgements):]
    prior = steps[ledger.clock]
    index = max(row['row_index'] for row in steps.values())+1
    for frame in range(ledger.clock+2,ledger.deadline+2,2):
        row = dict(source_id=scope[0],run_id=scope[1],software_reset=scope[2],pipe_object_id=scope[3],
            generation=deepcopy(prior['generation_after']),side='2P',frame_idx=frame,time_sec=frame/60,
            kind='enqueue',status='returned',active=False,returned_none=True,row_index=index,
            token=f'synthetic-future-enqueue:{frame}',added_occurrence_tokens=[],fifo_occurrence_tokens=[],
            before={'pending_tsumo':[]},after={'pending_tsumo':[],'tsumo_count':{}},
            discarded_tokens=[a.token for a in unacked],update_begin_accounting={
                'pending_tsumo':[list(a.pair) for a in unacked] if not packets else []},
            synthetic_future_fixture=True)
        source[frame] = row
        step = deepcopy(row) | dict(kind='step',row_index=index+1,token=f'synthetic-future-step:{frame}',
            exception=None,events=[],generation_after=row['generation'],code_sha256=prior['code_sha256'])
        steps[frame] = step
        receipt = C.prepare(module,ledger,row,C.end_from_context(ledger,contexts[ledger.clock])) if not packets else None
        packets.append(dict(source_fields=row,completed_call_token=step['token'],generation_after=step['generation_after'],
            terminal_receipt=None if receipt is None else json.loads(json.dumps(receipt)),quality_gate_clear=False))
        index += 2
    status = dict(closed=True,error=None,original_body=None,rows=len(packets),last_frame=ledger.deadline,
        receipt=packets[0]['terminal_receipt'],synthetic_future_fixture=True)
    return packets,status


def execute(parts: object, owner: object, stack: object, output: Path) -> dict:
    packet = P.read(output/'BELIEF_M1_SESSION.json')['modes'][0]
    call = packet['initial']['source_call_token']
    name = sha256(call.encode()).hexdigest()[:16]
    terminal = P.lines(output/('SECOND_TERMINAL_'+name+'.jsonl'))
    ledger = decode(parts.mode.L,terminal[-1]['ledger'])
    journal = P.lines(output/'atomic_journal.jsonl')
    source = P.source_rows(output,journal)  # 原tap実保存・閉鎖検査。人工tail追加前に行う。
    start = packet['initial']['state']['frame']
    steps = {row['frame_idx']:row for row in journal if row['kind']=='step' and row['side']=='2P' and row['frame_idx']>=start}
    contexts = {row['frame_idx']:row for row in P.lines(output/'provisional_context.jsonl') if row['frame_idx']>=start}
    packets,status = synthetic_tail(parts.mode.L,ledger,source,steps,contexts)
    if VERSION == 'v39':
        future = P.ROOT.parent/P.T.OUTPUT_NAME
        lines,read = P.lines,P.read
        replace = P.T.A.A.A.A.V4.replace_owned
        replace(stack,P,'lines',lambda path:packets if path == future/'SECOND_INACTIVE_END.jsonl' else lines(path))
        replace(stack,P,'read',lambda path:status if path == future/'SECOND_INACTIVE_END_STATUS.json' else read(path))
    consumer, original, saved, services = P.modules(stack,owner,parts)
    old = saved.Replay(parts.mode.C.T,parts.binding.S,parts.mode.CORE.V1.N,original,packet,steps,contexts)
    cls = consumer.Replay if VERSION == 'v39' else NEW.derive(consumer,packets,status)
    replay = cls(parts,services,old,source,tuple(P.lines(output/('SECOND_WARNING_'+name+'.jsonl'))),terminal)
    result = replay.run(P.lines(output/('SECOND_PREFIX_'+call.replace(':','_')+'.jsonl')),ledger.deadline)
    assert replay.ledger == ledger and result['applied'] == result['arrivals'] == 11
    assert result['acknowledged'] == 10 and len(result['unacknowledged_terminal_tokens']) == 1
    return result


def main() -> None:
    output = ROOT.parent/'video38_second_prefix_candidate_v38'
    assert P.read(output/'ENTRY_RESULT.json')['exit_code'] == 1
    started,error,result = time.perf_counter(),None,None
    try:
        with ExitStack() as stack:
            P.T.A.configured(stack)
            sys.path.insert(0,str(ROOT.parent/'g2_settled_basis_actual_mismatch_2026-09-12_v1'))
            import probe_native_merge
            owner = sys.modules[P.T.A.A.A.A.V4.OWNED_ALIAS]
            parts = owner.dependencies().modules()
            result = execute(parts,owner,stack,output)
    except BaseException as caught:
        error = repr(caught)
    value = dict(error=error,result=result,seconds=time.perf_counter()-started,
        original_exit_code=1,synthetic_future_after_36884=True,actual_video_tail=False,quality_gate_clear=False,
        runtime_version=VERSION,actual_new_runtime_postrun_modules=VERSION=='v39')
    name = 'SECOND_FULL_SAVED_CPU_v1.json' if VERSION == 'v38' else 'SECOND_FULL_SAVED_A39_CPU_v1.json'
    with (ROOT/name).open('x') as stream:
        json.dump(value,stream,indent=2)
    print(json.dumps(value))
    if error is not None:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
