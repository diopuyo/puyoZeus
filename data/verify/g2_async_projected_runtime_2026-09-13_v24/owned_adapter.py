"""A23を保持し、実prefix所有・予測入力保存の装着を別rootで選択する。"""
import importlib.util
from pathlib import Path
import sys
from typing import Any
import runtime_connection as C

ROOT = Path(__file__).resolve().parent
ASYNC = ROOT.parent / 'g2_async_projected_evaluation_2026-09-13_v1'
PREVIOUS_ROOT = ROOT.parent / 'g2_split_tail_runtime_2026-09-13_v23'


def module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    paths = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(value)
    finally:
        sys.path[:] = paths
    return value


PREVIOUS = module('_a24_previous', PREVIOUS_ROOT / 'owned_adapter.py')
B = module('_a24_session_binding', ASYNC / 'session_runtime_binding.py')
L = module('_a24_prefix_lease', ASYNC / 'prefix_owner_lease.py')
A, W = PREVIOUS.A, PREVIOUS.W
PRIOR, START, SIDE, protect = PREVIOUS.PRIOR, PREVIOUS.START, PREVIOUS.SIDE, PREVIOUS.protect
CANDIDATE, SAFETY, BASE = PREVIOUS.CANDIDATE, PREVIOUS.SAFETY, PREVIOUS.BASE
UNBOUND_MODULE, NOTICE = PREVIOUS.UNBOUND_MODULE, PREVIOUS.NOTICE
PREVIOUS_ADAPTER = PREVIOUS.PREVIOUS_ADAPTER
CURRENT: C.Connection | None = None


def sources() -> tuple[Path, ...]:
    names = ('session_runtime_binding', 'prefix_owner_lease', 'session_origin_binding',
             'journal_pair_reader', 'journal_origin_capture', 'projected_view',
             'prefix_projection', 'prefix_live_reader', 'prefix_projected_input')
    added = {ASYNC / (name + '.py') for name in names}
    added |= {ROOT / 'owned_adapter.py', ROOT / 'runtime_connection.py',
        B.REPO / 'src/chain_commit_candidate_v1.py', B.REPO / 'src/chain_prediction_ledger_v1.py',
        B.REPO / 'scripts/chain_end_epoch_shadow_v1.py', B.NOTICE / 'settled_notice.py'}
    return tuple(sorted(set(PREVIOUS.sources()) | added))


def configured(stack: Any) -> Any:
    global CURRENT
    if CURRENT is not None: raise ValueError('a24_overlapping_configuration')
    replace = A.A.A.V4.replace_owned
    lease = L.install(stack, UNBOUND_MODULE, replace)
    connection = C.Connection(stack, lease, B, replace)
    CURRENT = connection
    def release(kind: Any, body: Any, trace: Any) -> bool:
        global CURRENT
        if CURRENT is connection: CURRENT = None
        elif body is None: raise ValueError('a24_configuration_owner_changed')
        return False
    stack.push(release)
    selected = PREVIOUS.configured(stack)
    connection.install_environment(stack, selected)
    owner = sys.modules[A.A.A.V4.OWNED_ALIAS]
    connection.install_bootstrap(stack, owner)
    connection.install_creator(stack, A.A.A.V4.A.D)
    return selected
