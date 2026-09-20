"""新runtimeの私有loadだけへ、段階検査と2P型束縛を接続する。"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
ARRIVAL = ROOT.parent / 'g2_arrival_ack_candidate_2026-09-12_v1'
PUBLICATION = ROOT.parent / 'g2_belief_live_publication_2026-09-11_v1'


def scheduled(original: Any, parent: type) -> type:
    policy = original('_g2_capture_schedule', ROOT / 'capture_schedule.py')
    flags = original('_g2_evaluation_flags', ROOT / 'evaluation_flags.py')
    eligible = original('_g2_capture_eligibility', ROOT / 'capture_eligibility.py', {'evaluation_flags': flags})
    module = original('_g2_scheduled_session', ROOT / 'scheduled_session.py',
        dict(old_session=SimpleNamespace(Session=parent), capture_schedule=policy,
             evaluation_flags=flags, capture_eligibility=eligible))
    return module.Session


def bind_basis(stack: Any, value: Any, original: Any, replace: Any, source: Path) -> tuple:
    assert value.qualify.__globals__ is vars(value), 'second_basis_original_globals'
    assert Path(value.qualify.__code__.co_filename).resolve() == source, 'second_basis_original_source'
    repair = original('_g2_second_basis_boundary', ROOT / 'second_basis_boundary.py')
    selected = repair.wrapped(value.qualify, value)
    replace(stack, value, 'qualify', selected)
    assert value.initialize.__globals__['qualify'] is selected, 'second_initialize_binding'
    return value, selected


def install(stack: Any, bootstrap: Any, replace: Any) -> dict:
    original = bootstrap.load
    state: dict = dict(arrival=None, stable=None, session=None, basis=None, calls=[])
    binding = original('_g2_second_mode_binding', ROOT / 'second_mode_binding.py')

    def load(alias: str, path: Any, injection: Any = None) -> Any:
        value = original(alias, path, injection)
        source = Path(path).resolve()
        if source == ARRIVAL / 'stable_capture.py' and state['stable'] is None:
            repaired = original('_g2_late_stable_verify', ROOT / 'stable_capture_v2.py', {'old_stable': value})
            replace(stack, value, 'verify', repaired.verify)
            state['stable'] = (value, repaired.verify)
        elif source == ARRIVAL / 'arrival_mode.py':
            assert state['arrival'] is None or state['arrival'] is value, 'arrival_type_replaced'
            assert value.BASE is sys.modules['_g2_real_basis_cascade_mode_v2'], 'second_original_module_anchor'
            state['arrival'] = value
        elif source == PUBLICATION / 'second_basis.py' and state['basis'] is None:
            state['basis'] = bind_basis(stack, value, original, replace, source)
        elif source == PUBLICATION / 'live_session.py' and state['session'] is None:
            assert state['arrival'] is not None, 'session_before_arrival_type'
            selected = binding.session_class(SimpleNamespace(Session=value.Session), state['arrival'])
            selected = scheduled(original, selected)
            replace(stack, value, 'Session', selected)
            state['session'] = (value, selected)
        for key, attribute in (('stable', 'verify'), ('session', 'Session'), ('basis', 'qualify')):
            pair = state[key]
            if pair is not None:
                assert getattr(pair[0], attribute) is pair[1], 'completion_private_hook_changed'
        return value

    replace(stack, bootstrap, 'load', load)
    return state
