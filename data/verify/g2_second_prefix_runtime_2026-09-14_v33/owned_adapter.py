"""A24の実所有接続を再用し、評価Sessionより前の原J履歴だけ追加する。"""
import importlib.util
from pathlib import Path
import sys
from typing import Any
import early_runtime as E

ROOT = Path(__file__).resolve().parent
PREVIOUS_ROOT = ROOT.parent / 'g2_async_projected_runtime_2026-09-13_v24'
spec = importlib.util.spec_from_file_location('_early_runtime_previous', PREVIOUS_ROOT / 'owned_adapter.py')
PREVIOUS = importlib.util.module_from_spec(spec)
paths = list(sys.path)
try:
    sys.path.insert(0, str(PREVIOUS_ROOT))
    spec.loader.exec_module(PREVIOUS)
finally:
    sys.path[:] = paths
A, W, protect = PREVIOUS.A, PREVIOUS.W, PREVIOUS.protect
REPAIR = ROOT.parent / 'g2_history_prediction_diagnosis_2026-09-13_v1'
B = PREVIOUS.module('_a29_session_binding', REPAIR / 'session_runtime_binding_candidate.py')
OC = PREVIOUS.module('_a30_observer_window', REPAIR / 'observer_window_candidate.py')
PREFIX = ROOT.parent / 'g2_second_pending_prefix_2026-09-14_v3'
PC = PREVIOUS.module('_a33_second_prefix_selection', PREFIX / 'selection.py')
CURRENT: E.Runtime | None = None


def sources() -> tuple[Path, ...]:
    return tuple(sorted(set(PREVIOUS.sources()) | {ROOT / 'owned_adapter.py', ROOT / 'early_runtime.py',
        E.ASYNC / 'early_origin_history.py', E.ASYNC / 'fixed_origin_reference.py', E.PUB / 'journal_context.py',
        ROOT / 'early_probability_capture.py', E.PUB / 'second_observation.py',
        E.ASYNC / 'fixed_root_probability.py',
        REPAIR / 'journal_origin_capture_candidate.py', REPAIR / 'session_runtime_binding_candidate.py',
        REPAIR / 'observer_window_candidate.py', *OC.PATHS.values(),
        *(PC.BASE / name for name in ('consumed_prefix.py','physical_adapter.py','saved_replay.py')),
        *(PREFIX / name for name in ('session_binding.py','mode_composition.py','selection.py','postrun.py')),
        E.WRITER / 'writer_contract.py', E.WRITER / 'writer_contract_v2.py', E.WRITER / 'stream_witness.py'}))


def configured(stack: Any) -> Any:
    global CURRENT
    if CURRENT is not None: raise ValueError('early_runtime_overlapping_configuration')
    replace = A.A.A.V4.replace_owned
    replace(stack, PREVIOUS, 'B', B)
    selected = PREVIOUS.configured(stack)
    OC.install(stack, selected, replace)
    runtime = E.Runtime(stack, PREVIOUS.CURRENT, replace, probability_enabled=True)
    CURRENT = runtime
    def release(kind: Any, body: Any, trace: Any) -> bool:
        global CURRENT
        if CURRENT is runtime: CURRENT = None
        elif body is None: raise ValueError('early_runtime_configuration_owner')
        return False
    stack.push(release)
    original_begin = runtime.begin_history
    def begin(bridge: Any, owner: Any, pipeline: Any) -> None:
        import json
        value = OC.verify_state(bridge.state)
        with (bridge.state['output'] / 'OBSERVER_WINDOW_CONNECTION.json').open('x') as stream:
            json.dump(value, stream)
        original_begin(bridge, owner, pipeline)
    replace(stack, runtime, 'begin_history', begin)
    runtime.install_binding(stack)
    runtime.install_creator_boundary(stack)
    runtime.install_bridge(stack, A.A.A.V4.A, sys.modules[A.A.A.V4.OWNED_ALIAS])
    def scope() -> Any:
        connection = runtime.connection
        return connection.active_binding_stack() if connection.environment_called else stack
    PC.install(stack, sys.modules[A.A.A.V4.OWNED_ALIAS], replace, scope_getter=scope)
    return selected
