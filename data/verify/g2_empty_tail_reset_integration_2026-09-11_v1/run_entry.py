"""NEXT観測から、両側採録後・次原update前のresetまでを原105/J210で検査。"""
from __future__ import annotations
import run_qualification as Q

R = Q.module('_entry_desync_original', Q.ROOT/'run_desync.py')
import entry_continuation as E

if __name__ == '__main__':
    R.X.continuation = E.derived
    raise SystemExit(R.main())
