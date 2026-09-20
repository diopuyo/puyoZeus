"""v11を保持し、原whole driverに成功フレーム間休止だけを追加する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_m1_completion_runtime_2026-09-12_v11'
PACING_ROOT = ROOT.parent / 'g2_frame_pacing_candidate_2026-09-13_v1'
SPEC = importlib.util.spec_from_file_location('_g2_paced_prior', BASE / 'owned_adapter.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
PACE_SPEC = importlib.util.spec_from_file_location('_g2_owned_frame_pacing', PACING_ROOT / 'frame_pacing.py')
PACE = importlib.util.module_from_spec(PACE_SPEC)
PACE_SPEC.loader.exec_module(PACE)
PRIOR, START, SIDE, protect = A.PRIOR, A.START, A.SIDE, A.protect
CANDIDATE = A.CANDIDATE


def sources() -> tuple[Path, ...]:
    previous = {p for p in BASE.glob('*.py') if not p.name.startswith(('test_', 'probe_'))}
    return tuple(sorted(set(A.sources()) | previous | {ROOT / 'owned_adapter.py', PACING_ROOT / 'frame_pacing.py'}))


def configured(stack: Any) -> Any:
    selected = A.configured(stack)
    module = A.A.A.V4.A.DRIVER
    A.A.A.V4.replace_owned(stack, module, 'Driver', PACE.driver_class(module.Driver))
    return selected
