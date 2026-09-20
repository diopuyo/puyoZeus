"""原raw caller認証を生成loop/実入口へ明示接続したCPU統合。"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import FunctionType
from typing import Any
import run_qualification as Q

R = Q.module('_entry_desync_original',Q.ROOT/'run_desync.py')
GUARD = Q.ROOT.parent/'g2_collector_continuous_guard_2026-09-11_v1'
sys.path.insert(0,str(GUARD))
import integrated_connection as C
import entry_continuation_v2 as E
import entry_boundary_v3 as V


def drive(original: Any, namespace: Any) -> Any:
    return C.drive(original,namespace,R.X.extend)


def main() -> int:
    files = sorted(GUARD.glob('*.py'))
    pins = lambda: {str(p):sha256(p.read_bytes()).hexdigest() for p in files}
    before = pins()
    R.X.continuation = FunctionType(E.derived.__code__,dict(vars(E),E=V))
    R.R.drive = drive
    try:
        code = R.main()
        after = pins()
        with (Q.ROOT/sys.argv[1]/'COLLECTOR_GUARD_SOURCE.json').open('x',encoding='utf-8') as handle:
            json.dump(dict(before=before,unchanged=before==after,exit_code=code,quality_gate_clear=False),handle,indent=2)
        return code or int(before!=after)
    finally:
        C.OLD.restore()


if __name__ == '__main__':
    raise SystemExit(main())
