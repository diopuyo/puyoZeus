"""既存SessionのJ/Registry認証を残し、専用保存の出口だけ新親へ接続する。"""
from __future__ import annotations
from collections import ChainMap
import hashlib
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any
import parent_client_v2 as CLIENT
import parent_publication as PUBLICATION

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_belief_publication_runtime_2026-09-11_v1'))
import run_v6 as V6
import collector_session as BOUNDARY


def create(stack: Any, context: dict) -> Any:
    state = context['state']
    capture = state['joint_producer_capture']
    identity = (capture.identity['source_id'], capture.identity['run_id'], str(state['output']) + ':joint-ledger-v2')
    CLIENT.T.P.require(stack is state['joint_capture_stack'], 'joint_session_stack')
    client = stack.enter_context(CLIENT.Client(state['output'] / 'JOINT_EVENTS.jsonl', identity))
    state['joint_parent_client'] = client
    original_load = sys.modules['inflight_loader'].load
    legacy = original_load('_g2_joint_legacy_client', V6.V4.BRIDGE / 'client.py')
    def save(current: Any, config: Any, path: Path, *, sample_count: int = 256, seed: int = 0) -> dict:
        joint = sys.modules['_g2_pub_runtime_joint']
        packet = PUBLICATION.publish(client, current, capture.snapshot, config, legacy.request,
                                     joint, path, count=sample_count, seed=seed)
        client.runtime_stage = 'candidate_saved'
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    frame=packet['result']['frame'], quality_gate_clear=False,
                    probability_p1=packet['result']['details']['calibrated']['probability_p1'],
                    actual_video=False, session_completion_required=packet['session_completion_required'])
    saver = N(Configuration=legacy.Configuration, run=save)
    def load(alias: str, path: Any, injection: Any = None) -> Any:
        if Path(path).resolve() == V6.V4.BRIDGE / 'client.py': return saver
        return original_load(alias, path, injection)
    proxy = N(modules=ChainMap(dict(inflight_loader=N(load=load)), sys.modules))
    create_original = FunctionType(V6.create.__code__, dict(vars(V6), sys=proxy))
    return create_original(stack, context)


def start(post: Any, context: dict) -> None:
    root, state = context['stack'], context['state']
    CLIENT.T.P.require(root is not post and root is state['joint_capture_stack'], 'joint_root_owner')
    CLIENT.T.P.require(not state['joint_producer_capture'].closed, 'joint_capture_closed')
    state['belief_collector_boundary'] = BOUNDARY.Boundary(root, context, create)
