"""原cascadeローダーの生成型を保持し、到来/保存検査を同じ専用aliasで構築する。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent


def build(parts: Any, load: Callable) -> Any:
    base = parts.mode
    native = base.M.Mode.observe.__globals__['N']
    ledger = load('_g2_arrival_ledger', ROOT / 'ledger.py')
    math = load('_g2_arrival_math', ROOT / 'cascade_arrival.py',
                dict(belief=parts.belief, transition_candidates=base.V1.P.T, ledger=ledger))
    source = load('_g2_arrival_source', ROOT / 'enqueue_capture.py', dict(ledger=ledger))
    replay = load('_g2_arrival_source_replay', ROOT / 'source_replay.py',
                  dict(ledger=ledger, native_consumption=native))
    stable = load('_g2_arrival_stable', ROOT / 'stable_capture.py', dict(base_mode=base, source_replay=replay))
    mode = load('_g2_arrival_mode', ROOT / 'arrival_mode.py', dict(base_mode=base,
        cascade_arrival=math, enqueue_capture=source, ledger=ledger, stable_capture=stable))
    receipt = load('_g2_arrival_receipt_replay', ROOT / 'receipt_replay.py',
        dict(base_mode=base, cascade_arrival=math, ledger=ledger, source_replay=replay))
    saved = load('_g2_arrival_all_saved', ROOT / 'arrival_saved.py', dict(source_replay=replay, receipt_replay=receipt))
    old = load('_g2_arrival_old_saved', ROOT.parent / 'g2_live_probability_context_2026-09-12_v1/probability_saved.py')
    final = load('_g2_arrival_final_saved', ROOT / 'arrival_final_saved.py',
        dict(old_saved=old, arrival_saved=saved, stable_capture=stable, enqueue_capture=source))
    ledger.require(mode.BASE is base and math.B is parts.belief and math.T is base.V1.P.T
        and replay.N is native and stable.BASE is base and final.V is saved, 'loader_bound_modules')
    return SimpleNamespace(**(vars(parts) | dict(mode=mode, arrival_saved=final)))


def wrap(original: Callable, load: Callable) -> Callable:
    def modules() -> Any:
        return build(original(), load)
    return modules
