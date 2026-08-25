"""_diag_freeze_states_2026-08-15.py の単一イベント実行用ヘルパー
(モジュール名にハイフンを含むため `python -m` import が使えない事情への対処、
Bash経由でのクォート事故を避けるため独立ファイル化)。

使い方: PYTHONPATH=. ./venv/bin/python scripts/_run_diag_freeze_rank_2026-08-15.py <rank1-5>
"""
from __future__ import annotations

import importlib
import sys

m = importlib.import_module("scripts._diag_freeze_states_2026-08-15")

rank = int(sys.argv[1])
ev = next(e for e in m.FREEZE_EVENTS if e.rank == rank)
r = m.run_event(ev)
print(r)
