"""実PLAN設定との衝突を拒否し、生成後も設定一致を検査する。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
from typing import Any
import raw_upgrade as OLD

B, U, L = OLD.B, OLD.U, OLD.L


def merged(left: dict, right: dict) -> dict:
    B.C.require(all(left[k] == right[k] for k in left.keys() & right.keys()), 'live_setting_conflict')
    return dict(left, **right)


def build(collector: Any, source: Path, overrides: dict | None = None) -> Any:
    settings = merged(L.settings(), overrides or {})
    B.C.require(settings.get('enable_event_accounting_sidecar') is True, 'joint_accounting_required')
    def configured(c: Any, path: Path, extra: Any = None) -> Any:
        return U.build(c, path, overrides=merged(settings, extra or {}))
    proxy = N(**(vars(B.C) | dict(build=configured)))
    method = FunctionType(B.build.__code__, dict(vars(B), C=proxy), argdefs=B.build.__defaults__)
    loop = method(collector, source, settings)
    B.C.require(all(loop.configuration[k] == v for k, v in settings.items()), 'live_setting_changed')
    B.R.validate(loop)
    B.C.require(loop.runtime_state['accounting_recorder']._observed_frame_count == 0, 'joint_capture_must_precede_updates')
    return loop
