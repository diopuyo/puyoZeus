"""凍結v10を保持し、新M1接続候補だけを私有選択する。実走GOはまだない。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_arrival_ack_runtime_2026-09-12_v10'
CANDIDATE = ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1'
SPEC = importlib.util.spec_from_file_location('_g2_completion_prior', BASE / 'owned_adapter.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
THERMAL_SPEC = importlib.util.spec_from_file_location('_g2_completion_thermal', ROOT / 'thermal_child.py')
THERMAL = importlib.util.module_from_spec(THERMAL_SPEC)
THERMAL_SPEC.loader.exec_module(THERMAL)
DRIVER_SPEC = importlib.util.spec_from_file_location('_g2_completion_driver_status', ROOT / 'driver_status.py')
DRIVER_STATUS = importlib.util.module_from_spec(DRIVER_SPEC)
DRIVER_SPEC.loader.exec_module(DRIVER_STATUS)
PRIOR, START, SIDE, protect = A.PRIOR, A.START, A.SIDE, A.protect


def sources() -> tuple[Path, ...]:
    added = (ROOT / 'owned_adapter.py', ROOT / 'thermal_child.py', ROOT / 'driver_status.py', BASE / 'owned_adapter.py') + tuple(CANDIDATE / n for n in
        ('runtime_patch.py', 'second_mode_binding.py', 'stable_capture_v2.py', 'capture_schedule.py',
         'evaluation_flags.py', 'capture_eligibility.py', 'scheduled_session.py', 'second_basis_boundary.py',
         'schedule_saved.py', 'qualification_saved.py', 'second_pending_replay.py', 'qualification_audit.py'))
    added += (ROOT.parent / 'g2_ui_hsv_runtime_2026-09-12_v9/probe_review_m1_saved.py',
              ROOT.parent / 'g2_joint_collector_runtime_2026-09-12_v1/verify_joint_saved.py')
    return tuple(sorted(set(A.sources()) | set(added)))


def configured(stack: Any) -> Any:
    selected = A.configured(stack)
    dependencies = A.A.V4.A.D
    A.A.V4.replace_owned(stack, dependencies, 'session_creator', THERMAL.creator(dependencies.session_creator))
    driver = A.A.V4.A.DRIVER
    A.A.V4.replace_owned(stack, driver, 'Driver', DRIVER_STATUS.driver_class(driver))
    previous = A.A.V5.wrap_modules
    installed: dict = {}
    def wrap_modules(original: Any, inner: Any, owned: dict) -> Any:
        owner = sys.modules[A.A.V4.OWNED_ALIAS]
        A.A.V4.require_owned(owner)
        bootstrap = owner.bootstrap()  # 原cold Board確定後の依存生成でのみ呼ぶ。
        assert sys.modules['_g2_live_probability_base_loader'] is bootstrap, 'completion_bootstrap_owner'
        if bootstrap not in installed:
            patch = bootstrap.load('_g2_completion_runtime_patch', CANDIDATE / 'runtime_patch.py')
            installed[bootstrap] = patch.install(inner, bootstrap, A.A.V4.replace_owned)
        return previous(original, inner, owned)
    A.A.V4.replace_owned(stack, A.A.V5, 'wrap_modules', wrap_modules)
    return selected
