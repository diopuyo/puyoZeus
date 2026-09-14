"""認証済みcaptureを別環境へ渡し、同一性再確認後にだけ専用票を保存する。"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any,Callable

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
MODEL_MANIFEST='0b15913525c4c535f210954f24997b0377539be6e404a29785a6cb476de85399'
MAX_BYTES=32*1024*1024
TIMEOUT_SECONDS=180
DEFAULT_SAMPLES=256
FALSE_FIELDS=('supported','accounting_permission','integer_current_permission','training_permission','quality_gate_clear')


@dataclass(frozen=True)
class Configuration:
    serializer: Any
    grid: Callable[...,Any]
    input_type: type


def require(condition: bool,reason: str) -> None:
    if not condition: raise ValueError(reason)


def encoded(value: Any) -> bytes:
    return json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()


def request(value: Any,config: Configuration,count: int,seed: int) -> dict[str,Any]:
    require(type(value.inputs) is config.input_type,'client_input_carrier_type')
    return dict(schema='belief-model-process-request/v1',frame=value.frame,tokens=list(value.tokens),
        context_digest=value.digest,row=json.loads(value.inputs.source_json),
        states=[config.serializer.encode(v) for v in value.values],
        observed=[[list(r) for r in config.grid(v)] for v in value.observed],seed=seed,sample_count=count)


def exchange(value: dict[str,Any]) -> dict[str,Any]:
    raw=encoded(value)
    require(len(raw)<=MAX_BYTES,'client_request_size')
    environment=dict(os.environ,PYTHONPATH=str(PROJECT),CUDA_VISIBLE_DEVICES='-1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    child=subprocess.run([sys.executable,str(ROOT/'worker.py')],input=raw,capture_output=True,
        cwd=PROJECT,env=environment,timeout=TIMEOUT_SECONDS)
    require(child.returncode==0,'model_process_failed:'+child.stderr.decode(errors='replace')[-2000:])
    require(len(child.stdout)<=MAX_BYTES,'client_response_size')
    return json.loads(child.stdout)


def validated(reply: Any,req: Any) -> dict[str,Any]:
    require(type(reply) is dict and set(reply)=={'request_sha256','packet','live_registry_authorized','quality_gate_clear'},'client_reply_schema')
    require(reply['request_sha256']==hashlib.sha256(encoded(req)).hexdigest(),'client_request_digest')
    require(reply['live_registry_authorized'] is False and reply['quality_gate_clear'] is False,'client_child_authority')
    packet=reply['packet']
    require(packet['schema']=='belief-joint-evaluation-sidecar/v1','client_packet_schema')
    require(type(packet['frame']) is int and packet['frame']==req['frame'],'client_packet_frame')
    require(packet['context_digest']==req['context_digest'] and packet['journal_tokens']==req['tokens'],'client_context_tokens')
    require(encoded(packet['states'])==encoded(req['states']),'client_states_changed')
    require(all(packet[k] is False for k in FALSE_FIELDS) and packet['provisional'] is True
        and packet['ledger_connection']=='NOT_CONNECTED','client_permissions')
    require(packet['model_artifact_kind']=='ensemble_manifest','client_model_kind')
    manifest=json.dumps(packet['trained_details']['members'],sort_keys=True,separators=(',',':')).encode()
    require(hashlib.sha256(manifest).hexdigest()==packet['model_artifact_sha256']==MODEL_MANIFEST,'client_model_manifest')
    evaluation=packet['evaluation']
    for result in (evaluation,packet['trained_details']['raw']):
        require(all(result[k] is False for k in ('integer_current_permission','accounting_permission',
            'quality_gate_clear','source_producer_connected')) and result['provisional'] is True,'client_result_permissions')
    require(packet['trained_details']['supported'] is False and packet['trained_details']['trained_weights'] is True,'client_trained_status')
    require(evaluation['frame']==req['frame'] and evaluation['seed']==req['seed'],'client_evaluation_scope')
    counts=[len(s['hidden_worlds']) for s in req['states']]
    require(evaluation['world_counts']==counts and evaluation['used_samples']==(1 if counts==[1,1] else req['sample_count']),'client_sample_scope')
    require(type(evaluation['probability_p1']) in (int,float) and 0<=evaluation['probability_p1']<=1,'client_probability')
    encoded(packet)
    return packet


def run(capture: Callable[[],Any],members: Configuration,path: Path,*,
        sample_count: int=DEFAULT_SAMPLES,seed: int=0) -> dict[str,Any]:
    before=capture()
    req=request(before,members,sample_count,seed)
    packet=validated(exchange(req),req)
    fresh=capture()
    require(fresh.digest==before.digest and fresh.tokens==before.tokens
        and all(a is b for a,b in zip(fresh.values,before.values,strict=True)),'client_live_context_changed')
    require(encoded(request(fresh,members,sample_count,seed))==encoded(req),'client_live_inputs_changed')
    raw=encoded(packet)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
    require(path.read_bytes()==raw,'client_saved_bytes')
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),frame=before.frame,
        model_manifest_sha256=MODEL_MANIFEST,probability_p1=packet['evaluation']['probability_p1'],
        actual_video=False,quality_gate_clear=False)
