"""実NEXT emitから自動予約した入口で原資格/resetを105更新に通す。"""
from __future__ import annotations
from hashlib import sha256
import json
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any
import run_qualification as Q

R = Q.module('_entry_desync_original',Q.ROOT/'run_desync.py')
GUARD = Q.ROOT.parent/'g2_collector_continuous_guard_2026-09-11_v1'
AUTO = Q.ROOT.parent/'g2_empty_tail_auto_reset_2026-09-11_v1'
sys.path[:0] = [str(AUTO),str(GUARD)]
import integration as A
import integrated_connection as C
import raw_bound as B
import tail_auth as M
import entry_continuation_v2 as E

ORIGINAL_CONNECT = C.connect


def connect(stack: Any, state: Any) -> Any:
    return FunctionType(ORIGINAL_CONNECT.__code__,dict(vars(C),G=B,M=M))(stack,state)


def drive(original: Any, namespace: Any) -> Any:
    return C.drive(original,namespace,R.X.extend)


def main() -> int:
    files = sorted([*GUARD.glob('*.py'),*AUTO.glob('*.py')])
    pins = lambda: {str(p):sha256(p.read_bytes()).hexdigest() for p in files}
    before = pins()
    R.X.continuation = FunctionType(E.derived.__code__,dict(vars(E),E=N(install=A.register)))
    R.X.extend = A.extend(R.X.extend)
    R.R.drive,C.connect,C.RAW = drive,connect,N(install=B.install)
    try:
        code = R.main()
        with (Q.ROOT/sys.argv[1]/'COLLECTOR_GUARD_SOURCE.json').open('x',encoding='utf-8') as handle:
            json.dump(dict(before=before,unchanged=before==pins(),exit_code=code,quality_gate_clear=False),handle,indent=2)
        return code or int(before!=pins())
    finally:
        C.OLD.restore()


if __name__ == '__main__':
    raise SystemExit(main())
