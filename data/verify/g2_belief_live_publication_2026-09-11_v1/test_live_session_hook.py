"""原update成功後だけ接続処理を呼び、元例外と所有updateを復元する。"""
from __future__ import annotations

from contextlib import ExitStack
import json
from types import SimpleNamespace as N
from typing import Any
import pytest
import check_saved_inputs
import live_session as L


@pytest.mark.parametrize('failure',[None,'original','callback'])
def test_original_then_callback_and_restore(failure: Any) -> None:
    calls=[]
    error=RuntimeError('same exception')
    class Pipe:
        def update(self,frame_idx: int,time_sec: float) -> Any:
            calls.append('original')
            if failure=='original': raise error
            return self
    pipe=Pipe()
    original=Pipe.update
    def completed(frame: int) -> None:
        assert frame==10 and calls==['original']
        calls.append('callback')
        if failure=='callback': raise error
    value=N(pipe=pipe,completed=completed,error=None,restored=False)
    with ExitStack() as stack:
        L.attach(stack,value)
        if failure:
            with pytest.raises(RuntimeError) as caught: pipe.update(10,10/60)
            assert caught.value is error
        else: assert pipe.update(10,10/60) is pipe
    assert Pipe.update is original and value.restored
    assert calls==(['original'] if failure=='original' else ['original','callback'])


def test_exclusive_saved_receipt(tmp_path: Any) -> None:
    path=tmp_path/'session.json'
    packet=dict(quality_gate_clear=False,retired=dict(reset_call_attribution='UNATTRIBUTED'))
    L.write(path,packet)
    assert json.loads(path.read_bytes())==packet
    with pytest.raises(FileExistsError): L.write(path,dict(replaced=True))
    assert json.loads(path.read_bytes())==packet
