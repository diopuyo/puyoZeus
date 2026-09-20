"""既存の原発行票fixtureを値へ変換し、別processの三seed評価まで検査する。"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import subprocess
import sys
from typing import Any
import numpy as np
import pytest
import worker as W
from test_live_binding import saved,live
from test_live_binding_v2 import connected


@pytest.fixture
def request_value(connected: Any) -> Any:
    value=connected.capture()
    return dict(schema=W.SCHEMA,frame=value.frame,tokens=list(value.tokens),
        context_digest=value.digest,row=deepcopy(connected.base.rec.rows[-1]),
        states=[W.T.S.S.encode(v) for v in value.values],
        observed=[[list(r) for r in W.T.T.L.B.grid(v)] for v in value.observed],
        seed=17,sample_count=32)


def test_worker_reconstructs_original_inputs(request_value: Any,connected: Any) -> None:
    received=W.bound(request_value)
    expected=connected.capture()
    assert received.values==expected.values
    assert received.tokens==expected.tokens and received.digest==expected.digest
    for name in ('boards','queues','ledger_values','ledger_availability'):
        np.testing.assert_array_equal(getattr(received.inputs,name),getattr(expected.inputs,name))


@pytest.mark.parametrize('case',range(7))
def test_corrupt_request_rejected_before_weights(request_value: Any,case: int,monkeypatch: Any) -> None:
    if case==0: request_value['frame']+=2
    if case==1: request_value['context_digest']='0'*64
    if case==2: request_value['states'][0]['scope'][1]='other-run'
    if case==3: request_value['observed'][0][-1][0]=99
    if case==4: request_value['tokens'][1]=request_value['tokens'][0]
    if case==5: request_value['sample_count']=True
    if case==6:
        request_value['row']['ledger']['connection']='CONNECTED'
        request_value['context_digest']=W.T.T.backend().B.digest(request_value['row'])
    loader=W.T.T.backend().load_loader()
    def forbidden(*args: Any,**kwargs: Any) -> Any: raise AssertionError('weights_loaded_for_bad_request')
    monkeypatch.setattr(loader,'load_members',forbidden)
    with pytest.raises(ValueError): W.evaluate(request_value)


def test_separate_process_matches_saved_trained_result(request_value: Any) -> None:
    child=subprocess.run([sys.executable,str(W.ROOT/'worker.py')],input=W.encoded(request_value),
        capture_output=True,cwd=W.PROJECT,timeout=120)
    assert child.returncode==0,child.stderr.decode()
    reply=json.loads(child.stdout)
    assert reply['request_sha256']==hashlib.sha256(W.encoded(request_value)).hexdigest()
    assert reply['live_registry_authorized'] is False
    packet=reply['packet']
    expected=json.loads((W.PUB/'TRAINED_SIDECAR_v1.json').read_bytes())
    assert packet['frame']==expected['frame']==request_value['frame']
    assert packet['states']==request_value['states']
    assert packet['journal_tokens']==request_value['tokens']
    assert packet['model_artifact_sha256']==expected['model_artifact_sha256']
    for name in ('seed_raw','seed_calibrated'):
        for seed,values in packet['trained_details'][name].items():
            np.testing.assert_allclose(values,expected['trained_details'][name][seed],rtol=0,atol=1e-8)
    assert packet['evaluation']==expected['evaluation']
    assert packet['quality_gate_clear'] is False and packet['accounting_permission'] is False
