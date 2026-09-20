"""実保存ストリーム・原採録継続を含めた105更新入口の検証。"""
from __future__ import annotations
from types import FunctionType
from typing import Any
import run_qualification as Q

R = Q.module('_entry_desync_original',Q.ROOT/'run_desync.py')
import entry_continuation_v2 as E
import entry_boundary_v3 as V
import collector_connection as C

ORIGINAL_DRIVE = R.R.drive


def drive(original: Any, namespace: Any) -> Any:
    return C.wrap(ORIGINAL_DRIVE(original,namespace))


if __name__ == '__main__':
    R.X.continuation = FunctionType(E.derived.__code__,dict(vars(E),E=V))
    R.R.drive = drive
    try:
        raise SystemExit(R.main())
    finally:
        C.restore()
