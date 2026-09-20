"""正常側だけの明示module graph。reset Connection/arrival型を入力条件にしない。"""
from pathlib import Path
import sys
from typing import Any
import normal_dependencies as D
import normal_settings as K

ROOT = Path(__file__).resolve().parent
PUB = ROOT.parent / 'g2_belief_live_publication_2026-09-11_v1'
CAND = ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1'
MATH = ROOT.parent / 'g2_probabilistic_scope_candidate_2026-09-11_v1'
ORDER = ('journal_context', 'journal_witness', 'live_binding', 'reflection', 'live_binding_v2',
         'second_observation', 'second_basis', 'second_retirement', 'second_tracking', 'second_physical',
         'reflection_v3', 'live_binding_v3', 'sidecar', 'trained_connection', 'trained_sidecar')
NORMAL = frozenset(('second_observation', 'second_basis', 'second_retirement', 'second_tracking', 'second_physical'))


def replace(stack: Any, module: Any, name: str, value: Any) -> None:
    previous = getattr(module, name)
    setattr(module, name, value)
    def restore(kind: Any, body: Any, trace: Any) -> bool:
        if getattr(module, name) is value:
            setattr(module, name, previous)
        elif body is None:
            raise ValueError('normal_module_foreign:' + name)
        return False
    stack.push(restore)


def modules(state: Any, load: Any) -> dict:
    base = D.binding
    native = sys.modules['_g2_tracking_native']
    base.B.require(native.B is base.B is D.mode.B, 'normal_native_belief_identity')
    values = dict(belief=base.B, conditioning=base.C, serialization=base.S,
                  registry=base.OLD.R, native_consumption=native, normal_settings=K)
    values['joint_evaluation'] = load('_g2_pub_runtime_joint', MATH / 'joint_evaluation.py', values)
    for name in ORDER:
        source = (ROOT if name in NORMAL else PUB) / (name + '.py')
        values[name] = load('_normal_runtime_' + name, source, values)
    stack = state['joint_capture_stack']
    boundary = load('_normal_runtime_boundary', ROOT / 'second_basis_boundary.py')
    basis = values['second_basis']
    base.B.require(Path(basis.qualify.__code__.co_filename).resolve() == ROOT / 'second_basis.py'
                   and basis.qualify.__globals__ is vars(basis), 'normal_basis_source')
    replace(stack, basis, 'qualify', boundary.wrapped(basis.qualify, basis))
    schedule = load('_normal_runtime_schedule', CAND / 'capture_schedule.py')
    base.B.require(schedule.EARLIEST == (35370, 35410) and schedule.END == 36298
                   and schedule.STRIDE == K.STRIDE and schedule.MIN_GAP == K.MIN_GAP, 'normal_original_schedule')
    replace(stack, schedule, 'EARLIEST', K.EARLIEST)
    replace(stack, schedule, 'END', K.LAST)
    values['capture_schedule'] = schedule
    values['evaluation_flags'] = load('_normal_runtime_flags', CAND / 'evaluation_flags.py')
    values['original_eligibility'] = load('_normal_original_eligibility', CAND / 'capture_eligibility.py', values)
    for name in ('normal_owner', 'normal_eligibility', 'normal_session'):
        values[name] = load('_normal_runtime_' + name, ROOT / (name + '.py'), values)
    base.B.require(values['reflection_v3'].P is values['second_physical'], 'normal_reflection_type')
    values['live_session'] = values['normal_session']
    return values
