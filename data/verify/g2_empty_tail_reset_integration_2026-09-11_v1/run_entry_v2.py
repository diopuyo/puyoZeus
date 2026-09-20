"""原署名と例外保持を修復した次update入口の原105更新検査。"""
from __future__ import annotations
import run_qualification as Q

R = Q.module('_entry_desync_original', Q.ROOT/'run_desync.py')
import entry_continuation_v2 as E

if __name__ == '__main__':
    R.X.continuation = E.derived
    raise SystemExit(R.main())
