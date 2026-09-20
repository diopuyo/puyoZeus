"""既存の関数複製に互換な原採録接続を105更新へ通す。"""
from __future__ import annotations
from types import FunctionType
from typing import Any
import run_qualification as Q

R = Q.module('_entry_desync_original',Q.ROOT/'run_desync.py')
import entry_continuation_v2 as E
import entry_boundary_v3 as V
import collector_connection_v2 as C


def drive(original: Any, namespace: Any) -> Any:
    return C.drive(original,namespace,R.X.extend)


if __name__ == '__main__':
    R.X.continuation = FunctionType(E.derived.__code__,dict(vars(E),E=V))
    R.R.drive = drive
    try:
        raise SystemExit(R.main())
    finally:
        C.C.restore()
