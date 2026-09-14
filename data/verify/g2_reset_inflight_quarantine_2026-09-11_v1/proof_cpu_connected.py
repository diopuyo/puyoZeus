"""既存105更新・実保存資格callbackへ隔離hookを接続する正常回帰。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import torch
from inflight_loader import CONNECTION as C

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'g2_empty_tail_reset_live_adapter_2026-09-11_v1'
sys.path.insert(0,str(PREVIOUS))
import proof_cpu as P


class Context(P.P.Context):
    def perform(self,advisory: Any,frame: int,clock: float) -> None:
        super().perform(advisory,frame,clock)
        try:
            C.install(self.stack,self.recovery,self.state)
        except BaseException as error:
            self.error=repr(error)
            raise


def main() -> int:
    paths=[ROOT/name for name in ('quarantine.py','connection.py','inflight_loader.py','proof_cpu_connected.py')]
    pins={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    P.P.Context=Context
    P.R.A.A=P.arming_v2
    P.R.A.install=P.install
    P.R.E=N(**vars(P))
    code=P.R.main()
    after={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output=P.R.Q.ROOT/sys.argv[1]
    with (output/'INFLIGHT_SOURCE.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(source_sha=pins,unchanged=pins==after,cpu_only=not torch.cuda.is_initialized(),
                       quality_gate_clear=False,original_exit=code),stream,indent=2)
    return code or int(pins!=after or torch.cuda.is_initialized())


if __name__=='__main__': raise SystemExit(main())
