"""既存の最終NEXT変換認証を再用する固定入口。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'g2_firing_hand_connection_2026-09-10_v1'
sys.path[:0] = [str(PARENT), str(ROOT.parent/'g2_firing_policy_2026-09-10_v1')]
import firing_ticket_v4 as V

FIXED = {PARENT/'firing_ticket_v4.py': 'e86213693eff50808ffb9a08ba5f8682a7979797440c342f1c69ce77b24de1b7'}


def bind(pipe: Any, controller: Any) -> Any:
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p,h in FIXED.items()), 'scope_stop_source'
    assert V.__file__ and Path(V.__file__).resolve() == PARENT/'firing_ticket_v4.py'
    assert V.bound_update.__globals__ is vars(V), 'scope_stop_loader'
    actual = V.bound_update(pipe)
    assert actual.__globals__['__next_live'] is controller, 'scope_stop_controller'
    return actual
