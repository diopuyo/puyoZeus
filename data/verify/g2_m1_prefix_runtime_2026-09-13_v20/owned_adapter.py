"""私有prefix接続のcold選択候補。実動画GOは別の全経路検収を必要とする。"""
from __future__ import annotations
import importlib.util
import inspect
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_m1_completion_runtime_2026-09-12_v11'
SAFETY = ROOT.parent / 'g2_m1_paced_runtime_2026-09-13_v12'
SPEC = importlib.util.spec_from_file_location('_g2_normal_v11', BASE / 'owned_adapter.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
PRIOR, START, SIDE, protect = A.PRIOR, A.START, A.SIDE, A.protect
CANDIDATE = A.CANDIDATE
LATE = ROOT.parent / 'g2_m1_loader_lifetime_2026-09-13_v1/late_loader.py'
UNBOUND = ROOT.parent / 'g2_m1_unbound_start_2026-09-13_v1'
LIFECYCLE = ROOT.parent / 'g2_prefix_lane_integration_2026-09-13_v1'
UNBOUND_SPEC = importlib.util.spec_from_file_location('_g2_v17_unbound_loader', LIFECYCLE / 'prefix_selection.py')
UNBOUND_MODULE = importlib.util.module_from_spec(UNBOUND_SPEC)
UNBOUND_SPEC.loader.exec_module(UNBOUND_MODULE)
PREPARE = ROOT.parent / 'g2_m1_prepare_owner_2026-09-13_v1'
PREPARE_SPEC = importlib.util.spec_from_file_location('_g2_v18_prepare_owner', PREPARE / 'owner_binding.py')
PREPARE_MODULE = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(PREPARE_MODULE)
LATE_SPEC = importlib.util.spec_from_file_location('_g2_v16_late_loader', LATE)
LATE_MODULE = importlib.util.module_from_spec(LATE_SPEC)
LATE_SPEC.loader.exec_module(LATE_MODULE)


def sources() -> tuple[Path, ...]:
    previous = {p for p in BASE.glob('*.py') if not p.name.startswith(('test_', 'probe_'))}
    safety = {SAFETY / name for name in ('resource_guard.py', 'supervise.py', 'review_resources.py')}
    back = ROOT.parent / 'g2_arrival_backlog_2026-09-13_v1'
    private = {back / name for name in ('split_landing.py', 'prefix_phase_math_v2.py', 'engine_identity.py', 'prefix_phase_saved_v4.py', 'identified_origin_candidate.py')}
    private |= {LIFECYCLE / 'stable_decision.py', LIFECYCLE / 'prefix_replay.py', LIFECYCLE / 'prefix_final_saved.py', LIFECYCLE / 'prefix_origin_initial.py'}
    return tuple(sorted(set(A.sources()) | private | previous | safety | {LIFECYCLE / 'prefix_selection.py', LIFECYCLE / 'prefix_live_adapter.py', LIFECYCLE / 'lane_state.py', LIFECYCLE / 'prefix_commit.py', LIFECYCLE / 'stable_evidence.py', PREPARE / 'owner_binding.py', UNBOUND / 'runtime_selection.py', UNBOUND / 'second_mode_binding.py', LATE, ROOT / 'owned_adapter.py',
        ROOT.parent / 'g2_bounded_publication_runtime_2026-09-09_v1/resource_guard.py'}))


def configured(stack: Any) -> Any:
    selected = A.configured(stack)
    dependencies = A.A.A.V4.A.D
    creator = dependencies.session_creator
    assert Path(creator.__code__.co_filename).resolve() == BASE / 'thermal_child.py', 'expected_thermal_wrapper'
    original = inspect.getclosurevars(creator).nonlocals['original']
    A.A.A.V4.replace_owned(stack, dependencies, 'session_creator', original)
    import sys
    owner = sys.modules[A.A.A.V4.OWNED_ALIAS]
    LATE_MODULE.install(stack, dependencies, owner, A.A.A.V4.replace_owned)
    UNBOUND_MODULE.install(stack, owner, A.A.A.V4.replace_owned)
    PREPARE_MODULE.install(stack, A.A.A.V4.A, owner, selected, A.A.A.V4.replace_owned)
    return selected
