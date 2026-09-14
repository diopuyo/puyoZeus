"""A40終了原票と保存結果を確認し、欠測・破損をそのままDへ記録する。"""
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as N

ROOT = Path(__file__).resolve().parent
DEST = Path('/mnt/d/puyo_analyzer/verify/g2_a40_recovery_2026-09-14_v1')
sys.path.insert(0, str(ROOT.parent/'g2_second_prefix_runtime_2026-09-14_v40'))
import target_entry as T
sys.path.insert(0, str(ROOT.parent/'g2_a38_postrun_2026-09-14_v1'))
import first_terminal_saved as V


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def lines(path: Path) -> list:
    with path.open() as stream: return [json.loads(line) for line in stream]


def first(output: Path, parts: object) -> dict:
    state = read(output/'FIRST_TERMINAL_STATUS.json')
    saved, L = state['receipt']['last_live_ledger'], parts.mode.L
    ledger = L.Ledger(tuple(saved['scope']), saved['start'], saved['deadline'], saved['clock'],
        tuple(L.Arrival(tuple(r['scope']),r['token'],tuple(r['pair']),r['frame'],r['call_token']) for r in saved['arrivals']),
        tuple(saved['applied']), tuple(L.Ack(r['token'],r['frame'],r['call_token']) for r in saved['acknowledgements']))
    packets = lines(output/'FIRST_TERMINAL.jsonl')
    contexts = {r['frame_idx']:r for r in lines(output/'provisional_context.jsonl')}
    original = N(L=parts.mode.L, R=parts.arrival_saved.V.R)
    result = V.verify_records(original,ledger,lines(output/'atomic_journal.jsonl'),packets,contexts)
    assert state['closed'] and not state['pending'] and state['error'] is state['original_body'] is None
    assert state['rows']==len(packets)==8 and state['last_frame']==result['terminal_end']==36900
    assert state['receipt']==packets[0]['terminal_receipt']
    return result | dict(arrivals=len(ledger.arrivals), applied=len(ledger.applied),
        acknowledged=len(ledger.acknowledgements), status_verified=True)


def outlets(output: Path) -> dict:
    rows = [r for r in lines(output/'PROJECTED_ORIGIN_CAPTURE.jsonl') if r.get('frame',-1)>=36886]
    projected = [r['projected_input'] for r in rows if 'projected_input' in r]
    assert len(projected)==8 and all(r['status']=='HOLD' and r['reason']=='match_ended_scope_frozen' for r in projected)
    close = read(output/'PROJECTED_ORIGIN_CAPTURE_CLOSE.json')
    assert close['last_frame']==36900 and close['capture_closed'] and close['stream_closed']
    assert close['original_error'] is None and close['cleanup_errors']==[]
    session = read(output/'BELIEF_M1_SESSION.json')
    assert session['schedule']['last']==36900 and session['schedule']['pending'] is None
    assert session['error'] is session['session_error'] is None and session['restored']
    return dict(projected_terminal_HOLD=len(projected), projected_close=close,
        legacy_schedule=session['schedule'], legacy_saved=len(session['saved']))


def main() -> None:
    output = ROOT.parent/T.OUTPUT_NAME
    started, results = time.perf_counter(), {}
    T.approved(output)
    with ExitStack() as stack:
        T.A.configured(stack)
        sys.path.insert(0,str(ROOT.parent/'g2_settled_basis_actual_mismatch_2026-09-12_v1'))
        import probe_native_merge
        parts = sys.modules[T.A.A.A.A.V4.OWNED_ALIAS].dependencies().modules()
        for name, fn in (('first_terminal',lambda:first(output,parts)),):
            try: results[name] = dict(result=fn(),error=None)
            except BaseException as error: results[name] = dict(error=repr(error))
    results.update(original_exit_code=read(output/'ENTRY_RESULT.json')['exit_code'],
        quality_gate_clear=False, whole_run_success=False, seconds=time.perf_counter()-started)
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/'FIRST_TERMINAL_RECORD_REVIEW_v2.json').open('x') as stream: json.dump(results,stream,indent=2)
    print(json.dumps(results))


if __name__=='__main__': main()
