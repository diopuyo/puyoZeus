"""保存直前の生参照再検査、改変応答拒否、元モデル子processまでの連続対照。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import client as C
import pure_contract as P
import worker as W
from test_live_binding import saved,live
from test_live_binding_v2 import connected


@pytest.fixture
def endpoint(connected: Any) -> Any:
    def capture() -> Any:
        return replace(connected.capture(),inputs=P._inputs(connected.base.rec.rows[-1]))
    config=C.Configuration(W.T.S.S,W.T.T.L.B.grid,P.RowInputs)
    def reply(req: Any) -> Any:
        packet=json.loads((W.PUB/'TRAINED_SIDECAR_v1.json').read_bytes())
        packet.update(states=req['states'],context_digest=req['context_digest'],journal_tokens=req['tokens'])
        return dict(request_sha256=hashlib.sha256(C.encoded(req)).hexdigest(),packet=packet,
            live_registry_authorized=False,quality_gate_clear=False)
    return N(capture=capture,config=config,reply=reply)


@pytest.mark.parametrize('case',range(8))
def test_bad_response_never_saved(endpoint: Any,tmp_path: Path,monkeypatch: Any,case: int) -> None:
    def exchange(req: Any) -> Any:
        reply=endpoint.reply(req)
        if case==0: reply['request_sha256']='0'*64
        if case==1: reply['packet']['frame']+=2
        if case==2: reply['packet']['journal_tokens']=['step:0','step:1']
        if case==3: reply['packet']['states']=deepcopy(req['states']); reply['packet']['states'][0]['deadline']+=2
        if case==4: reply['packet']['model_artifact_sha256']='0'*64
        if case==5: reply['live_registry_authorized']=True
        if case==6: reply['packet']['training_permission']=True
        if case==7: reply['packet']['evaluation']['accounting_permission']=True
        return reply
    monkeypatch.setattr(C,'exchange',exchange)
    path=tmp_path/'result.json'
    with pytest.raises(ValueError): C.run(endpoint.capture,endpoint.config,path,sample_count=32,seed=17)
    assert not path.exists()


def test_changed_live_object_never_saved(endpoint: Any,tmp_path: Path,monkeypatch: Any) -> None:
    monkeypatch.setattr(C,'exchange',endpoint.reply)
    before=endpoint.capture()
    values=iter((before,replace(before,values=(replace(before.values[0]),before.values[1]))))
    path=tmp_path/'result.json'
    with pytest.raises(ValueError,match='client_live_context_changed'):
        C.run(lambda:next(values),endpoint.config,path,sample_count=32,seed=17)
    assert not path.exists()


def test_success_exclusive_save(endpoint: Any,tmp_path: Path,monkeypatch: Any) -> None:
    monkeypatch.setattr(C,'exchange',endpoint.reply)
    path=tmp_path/'result.json'
    receipt=C.run(endpoint.capture,endpoint.config,path,sample_count=32,seed=17)
    raw=path.read_bytes()
    assert receipt['sha256']==hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError): C.run(endpoint.capture,endpoint.config,path,sample_count=32,seed=17)
    assert path.read_bytes()==raw


def test_real_child_to_parent_saved_packet(endpoint: Any,tmp_path: Path) -> None:
    path=tmp_path/'result.json'
    receipt=C.run(endpoint.capture,endpoint.config,path,sample_count=32,seed=17)
    packet=json.loads(path.read_bytes())
    assert receipt['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert packet['states']==[endpoint.config.serializer.encode(v) for v in endpoint.capture().values]
    assert packet['model_artifact_sha256']==C.MODEL_MANIFEST
