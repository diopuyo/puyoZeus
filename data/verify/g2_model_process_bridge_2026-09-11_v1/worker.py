"""独立Python環境で原入力変換・固定三seed評価を実行する。生Registry資格は発行しない。"""
from __future__ import annotations
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
PUB=ROOT.parent/'g2_belief_live_publication_2026-09-11_v1'
MATH=ROOT.parent/'g2_probabilistic_scope_candidate_2026-09-11_v1'
sys.path[:0]=[str(PUB),str(MATH),str(PROJECT)]
import trained_sidecar as T

SCHEMA='belief-model-process-request/v1'
KEYS=frozenset(('schema','frame','tokens','context_digest','row','states','observed','seed','sample_count'))
MAX_BYTES=32*1024*1024
MAX_SAMPLES=4096


def encoded(value: Any) -> bytes:
    return json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()


def bound(request: Any) -> Any:
    L=T.T.L
    L.C.require(type(request) is dict and set(request)==KEYS and request['schema']==SCHEMA,'wire_schema')
    frame=request['frame']
    L.C.require(type(frame) is int and frame>=0,'wire_frame')
    L.C.require(type(request['seed']) is int and request['seed']>=0,'wire_seed')
    L.C.require(type(request['sample_count']) is int and 0<request['sample_count']<=MAX_SAMPLES,'wire_samples')
    row=request['row']
    contract=T.T.backend().B
    registration=dict(source_id=row['source_id'],run_id=row['run_id'],time_base_numerator=1,
        time_base_denominator=60,ledger_connection='NOT_CONNECTED')
    contract._identity(row,registration)
    contract._update(row)
    contract._generation(row)
    L.C.require(row['frame_idx']==frame and contract.digest(row)==request['context_digest'],'wire_context')
    tokens=request['tokens']
    L.C.require(type(tokens) is list and len(tokens)==2 and len(set(tokens))==2
        and all(type(t) is str and t.startswith('step:') and t[5:].isdigit() for t in tokens),'wire_tokens')
    L.C.require(type(request['states']) is list and len(request['states'])==2,'wire_states')
    values=tuple(T.S.S.decode(value) for value in request['states'])
    L.C.require(all(v.scope[:2]==(row['source_id'],row['run_id']) for v in values),'wire_scope')
    observed=request['observed']
    L.C.require(type(observed) is list and len(observed)==2,'wire_observed')
    L.C.require(all(grid==row['sides'][side]['before_hold']['confirmed']['grid']
        for grid,side in zip(observed,('1P','2P'),strict=True)),'wire_observed_context')
    boards=tuple(L.B.Board.from_dict({'grid':grid}) for grid in observed)
    L.J.inputs(values,frame,('STABLE','STABLE'),boards)
    return L.Bound(values,boards,contract._inputs(row),frame,tuple(tokens),request['context_digest'])


def evaluate(request: Any) -> dict[str,Any]:
    value=bound(request)
    loader=T.T.backend().load_loader()
    members=loader.load_members(value.values[0].scope[0],'cpu')
    current,result,details=T.T.evaluate(lambda:value,members,
        sample_count=request['sample_count'],seed=request['seed'])
    manifest=json.dumps(details['members'],sort_keys=True,separators=(',',':')).encode()
    packet=T.S.packet(current,result,hashlib.sha256(manifest).hexdigest())
    packet.update(model_artifact_kind='ensemble_manifest',trained_details=details)
    return dict(request_sha256=hashlib.sha256(encoded(request)).hexdigest(),packet=packet,
        live_registry_authorized=False,quality_gate_clear=False)


def main() -> None:
    raw=sys.stdin.buffer.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES: raise ValueError('wire_request_size')
    request=json.loads(raw)
    with redirect_stdout(sys.stderr):
        result=evaluate(request)
    sys.stdout.buffer.write(encoded(result)+b'\n')


if __name__=='__main__': main()
