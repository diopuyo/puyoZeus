"""実v11の既検査済み保存票で不整合信号時点を診断。実呼出の権限は復元しない。"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import desync_signal as S

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent/'g2_empty_tail_v11_next_audit_2026-09-10_v1'
POST = ROOT.parent/'g2_empty_tail_v11_postrun_2026-09-10_v1'
LAST_COMMIT = 35068
INPUTS = (AUDIT/'cpu_v1/CASES.json',AUDIT/'accounting_v1/RESULT.json',
    AUDIT/'saved_v1/RESULT.json',POST/'saved_v1/RESULT.json')


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def inputs() -> Any:
    cases,accounts,decisions,enqueues = map(read,INPUTS)
    checked = next(r['rows'] for r in cases if r['case']=='saved57_original_functions')
    return (checked,{r['frame_idx']:r for r in accounts['rows']},
        {r['frame']:r for r in decisions['rows']},{r['frame_idx']:r for r in enqueues['enqueues']})


def fact(row: Any, account: Any, saved: Any, enqueue: Any) -> S.Fact | None:
    assert row['frame']==account['frame_idx']==saved['frame']==enqueue['frame_idx']
    assert row['reason']==saved['reason'] and not saved['native_appended']
    if (row['reason']!='await_dnext_successor' or not row['quiet'] or saved['quiet_frames'] is None
        or account['after']['pending_tsumo']): return None
    candidate = saved['held_candidate']
    payload = account['history_after']['payload']
    scope = (enqueue['source_id'],enqueue['run_id'],enqueue['pipe_object_id'],enqueue['side'],
        enqueue['software_reset'],payload[4][0])
    event = tuple(candidate[k] for k in ('sequence_number','first_support_frame','available_frame','reference'))
    dnext = account['raw']['main']['returned']['1P']['dnext']
    return S.Fact(row['frame'],scope,LAST_COMMIT,tuple(row['baseline_dnext']),tuple(row['pair']),
        tuple(dnext),event,tuple(saved['quiet_frames']),row['quiet'])


def evaluate() -> Any:
    rows,accounts,decisions,enqueues = inputs()
    assert decisions[LAST_COMMIT]['reason']=='occurrence_committed'
    matcher,signals,eligible = S.Matcher(),[],[]
    for row in rows:
        frame = row['frame']
        item = fact(row,accounts[frame],decisions[frame],enqueues[frame])
        if item is not None: eligible.append(frame)
        signal = matcher.push(item)
        if signal is not None: signals.append([asdict(v) for v in signal])
    return dict(saved_updates=len(rows),eligible_frames=eligible,signals=signals,
        source='original_v11_saved_only',actual_invocation_restored=False,
        reset_permission=False,missing_count='UNCERTIFIED',quality_gate_clear=False)


def main() -> int:
    output = ROOT/sys.argv[1]
    output.mkdir(exist_ok=False)
    paths = list(INPUTS)+[Path(__file__),Path(S.__file__)]
    before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    started = time.monotonic()
    result = evaluate()
    result.update(pid=os.getpid(),seconds=time.monotonic()-started,
        inputs_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()))
    (output/'INPUTS.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    (output/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(pid=result['pid'],seconds=result['seconds'],
        signal_frames=[[r['frame'] for r in s] for s in result['signals']],inputs_unchanged=result['inputs_unchanged'])))
    return 0 if result['inputs_unchanged'] else 1


if __name__=='__main__':
    raise SystemExit(main())
