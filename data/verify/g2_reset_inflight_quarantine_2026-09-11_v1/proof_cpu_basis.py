"""元の整数正常回帰へ同call確率basis観測を並置する。元の採用判定は不変。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import proof_cpu_connected as R
from inflight_loader import basis_connection


class Context(R.Context):
    def perform(self,advisory: Any,frame: int,clock: float) -> None:
        super().perform(advisory,frame,clock)
        try:
            deadline=sys.modules[type(self.recovery).__module__].I.DEADLINE
            basis_connection().install(self.stack,self.recovery,self.state,frame,deadline)
        except BaseException as error:
            self.error=repr(error)
            raise


if __name__=='__main__':
    root=R.ROOT.parent/'g2_reset_settled_basis_gate_2026-09-11_v1'
    paths=[root/name for name in ('settled_basis_gate.py','gate_v2.py','actual_connection.py')]+[Path(__file__)]
    sha=lambda:{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    before=sha()
    R.Context=Context
    code=R.main()
    after=sha()
    with (R.P.R.Q.ROOT/sys.argv[1]/'SETTLED_BASIS_SOURCE.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(before=before,unchanged=before==after,original_exit=code,quality_gate_clear=False),stream,indent=2)
    raise SystemExit(code or int(before!=after))
