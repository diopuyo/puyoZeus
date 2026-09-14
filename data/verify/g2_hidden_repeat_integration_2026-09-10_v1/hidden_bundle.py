"""既存部品の依存順を再用し、設置口だけを分離する。"""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
RETAINED = VERIFY/'g2_hidden_retained_integer_candidate_2026-09-10_v1'
REPEAT = VERIFY/'g2_repeated_firing_runtime_candidate_2026-09-10_v1'
SCOPE = VERIFY/'g2_same_scope_stop_guard_2026-09-10_v1'
sys.path.insert(0,str(RETAINED))
import run_retained as HR

CONT, LIFE, CURRENT, TAIL = HR.B,HR.B.B,HR.B.B.B,HR.B.B.B.B
PREFIX, K = HR.R,HR.R.K


def guards() -> dict[str,str]:
    paths = [*RETAINED.glob('*.py'),RETAINED/'PLAN.md',*ROOT.glob('*.py'),ROOT/'PLAN.md']
    return HR.guards() | {str(p):K.sha(p) for p in paths}


def extensions(stack: Any, factory: Any, configured: Any) -> None:
    control = factory.controller
    control.hidden_history_rows = []
    control.hidden_current_events,control.hidden_current_outputs = [],[]
    control.hidden_lifetime_rows = []
    PREFIX.C.install(stack,factory,configured,control.hidden_history_rows)
    TAIL.C.install(stack,factory,configured.patch,control.hidden_history_rows)
    provisional = K.load('_combined_existing_provisional',CURRENT.PROVISIONAL,stack)
    CURRENT.C.install(stack,factory,configured.patch,provisional,
        control.hidden_current_events,control.hidden_current_outputs)
    LIFE.L.install(stack,factory,configured.patch,control.hidden_lifetime_rows)
    CONT.C.install(stack,factory,configured.patch,control.hidden_history_rows)
    HR.W.install(stack,configured.patch)
