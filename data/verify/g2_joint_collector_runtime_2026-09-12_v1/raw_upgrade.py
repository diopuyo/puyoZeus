"""原raw guardを保持し、生成時の観測器と実採録設定だけを選ぶ。"""
from __future__ import annotations
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent
for folder in ('g2_collector_continuous_guard_2026-09-11_v1', 'g2_frozen_accounting_upgrade_2026-09-12_v1'):
    sys.path.insert(0, str(ROOT.parent / folder))
import raw_bound as B
import upgrade_v2 as U
import live_configuration as L


def build(collector: Any, source: Path, overrides: dict | None = None) -> Any:
    settings = dict(L.settings(), **(overrides or {}))
    B.C.require(settings.get('enable_event_accounting_sidecar') is True, 'joint_accounting_required')
    def configured(c: Any, path: Path, extra: Any = None) -> Any:
        return U.build(c, path, overrides=dict(settings, **(extra or {})))
    proxy = N(**(vars(B.C) | dict(build=configured)))
    method = FunctionType(B.build.__code__, dict(vars(B), C=proxy), argdefs=B.build.__defaults__)
    loop = method(collector, source, settings)
    B.R.validate(loop)
    B.C.require(loop.runtime_state['accounting_recorder']._observed_frame_count == 0,
                'joint_capture_must_precede_updates')
    return loop
