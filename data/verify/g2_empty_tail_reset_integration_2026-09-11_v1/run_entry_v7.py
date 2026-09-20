"""同一予約の対象外side通知を冪等化した原105/J210自動復旧。"""
from __future__ import annotations
import run_entry_v6 as R
import arming_v2 as A

if __name__ == '__main__':
    R.A.A = A
    raise SystemExit(R.main())
