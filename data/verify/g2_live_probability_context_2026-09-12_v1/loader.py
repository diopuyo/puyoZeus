"""冷起動後の原型を使い、予約名を占有せず既存確率部品を接続する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
REPO = ROOT.parents[2]
SNAPSHOT = REPO / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
QUARANTINE = VERIFY / 'g2_reset_inflight_quarantine_2026-09-11_v1'
HIDDEN = VERIFY / 'g2_hidden_basis_initialization_2026-09-11_v1'
CASCADE = VERIFY / 'g2_cascade_connection_plan_2026-09-11_v1'


def bootstrap() -> Any:
    board = sys.modules.get('src.board')
    if board is None or Path(board.__file__).resolve() != SNAPSHOT / 'src/board.py':
        raise RuntimeError('live_probability_requires_frozen_cold_board')
    alias, path = '_g2_live_probability_base_loader', QUARANTINE / 'inflight_loader.py'
    if alias in sys.modules:
        if Path(sys.modules[alias].__file__).resolve() != path:
            raise RuntimeError('live_probability_loader_alias')
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


def dependencies() -> Any:
    first = bootstrap()
    load = first.load
    second = load('_g2_live_probability_loader_v2', QUARANTINE / 'inflight_loader_v2.py', {'inflight_loader': first})
    binding = load('_g2_live_probability_binding_loader', QUARANTINE / 'basis_binding_loader.py', {'inflight_loader_v2': second})
    tracking = load('_g2_live_probability_tracking_loader', QUARANTINE / 'tracking_loader.py',
                    {'inflight_loader_v2': second, 'basis_binding_loader': binding})
    hidden = load('_g2_live_probability_hidden_loader', HIDDEN / 'hidden_loader.py', {'tracking_loader': tracking})
    cascade = load('_g2_live_probability_cascade_loader', CASCADE / 'cascade_loader.py')
    side = load('_g2_live_probability_side_connection', HIDDEN / 'side_actual_connection.py', {'inflight_loader': first})
    occurrence = load('_g2_live_probability_occurrence_retirement',
                      VERIFY / 'g2_side_occurrence_retirement_2026-09-11_v1/retirement.py')
    tracker = load('_g2_live_probability_tracker_type', CASCADE / 'tracker_input.py')
    baseline = load('_g2_live_probability_baseline_retirement', CASCADE / 'baseline_retirement.py', {'tracker_input': tracker})
    snapshot_root = VERIFY / 'g2_tracker_snapshot_repair_2026-09-11_v1'
    group = load('_g2_live_probability_group_snapshot', snapshot_root / 'group_snapshot.py')
    snapshot = load('_g2_live_probability_snapshot_connection', snapshot_root / 'connection.py', {'group_snapshot': group})
    archive_root = VERIFY / 'g2_archive_close_boundary_2026-09-11_v1'
    lifetime = load('_g2_live_probability_archive_lifetime', archive_root / 'archive_lifetime.py')
    anchor = load('_g2_live_probability_archive_anchor', archive_root / 'anchor_v2.py', {'archive_lifetime': lifetime})
    votes_root = VERIFY / 'g2_transition_endpoint_votes_2026-09-11_v1'
    endpoint = load('_g2_live_probability_votes_base', votes_root / 'endpoint_votes.py')
    prior = load('_g2_live_probability_votes_prior', votes_root / 'prior_votes_v2.py', {'endpoint_votes': endpoint})
    votes = load('_g2_live_probability_votes_qualified', votes_root / 'qualified_connection.py', {'prior_votes_v2': prior})
    core = load('_g2_live_probability_context_core', ROOT / 'context.py')
    return core.Dependencies(cascade.wrap(hidden.modules, load), tracking.CONNECTION, lambda: votes,
                             side, occurrence, baseline, snapshot, anchor)


def build(parent: type, *, end_frame: int) -> type:
    parts = dependencies()
    core = sys.modules['_g2_live_probability_context_core']
    return core.compose(parent, parts, end_frame=end_frame)
