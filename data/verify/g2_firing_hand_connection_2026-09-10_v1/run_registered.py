"""原39更新で前段資格→原J pop→起源登録まで。実精算はこの版では未到達。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import inspect
from types import FunctionType, SimpleNamespace
from typing import Any
import firing_connection as F
import run_front as FRONT

R, V2, ROOT = FRONT.R, FRONT.V2, FRONT.ROOT
DEFERRED = ROOT.parent/'g2_deferred_next_handoff_2026-09-10_v1'
POLICY = ROOT.parent/'g2_firing_policy_2026-09-10_v1'
sys.path[:0] = [str(DEFERRED), str(POLICY)]
import held_v2 as H2
import firing_policy as P


def references(factory: Any, pipe: Any) -> tuple[Any, ...]:
    cls = type(factory.controller)
    v1 = cls.prepared.__globals__['V1']
    accounting = inspect.getclosurevars(factory.controller.inventory.BoundPolicy.__init__).nonlocals['p']
    return (cls.prepared, cls.call, cls.consumed_history, v1.prepared,
            factory.provider.no_origin, pipe._apply_chain_formula_early_fire, accounting.BoundPolicy)


def drive(q: Any, target: Any, m: Any, state: Any, real: Any, fixture: Any,
          binding: Any, factory: Any) -> None:
    pipe, rows = real[0], []
    counter = {s:(getattr(pipe, '_tsumo_count_'+s), deepcopy(getattr(pipe, '_tsumo_count_'+s)))
               for s in ('1p', '2p')}
    data: dict[str, Any] = dict(trace=[], completed=[], primary=None)
    initial = references(factory, pipe)
    try:
        with ExitStack() as stack:
            base = R.K.load('_firing_deferred_base', DEFERRED/'connection.py', stack)
            connection = R.K.load('_firing_deferred_identity', DEFERRED/'connection_v2.py', stack)
            base.patch(stack, base.D, 'check', H2.check)
            stack.enter_context(connection.installed(base, factory.controller))
            stack.enter_context(F.installed(factory, pipe, base, P, rows))
            R.D.loop(q, target, m, state, real, fixture, binding, factory, data)
    finally:
        data['firing_references_restored'] = initial == references(factory, pipe)
        data['native_counter_unchanged'] = all(getattr(pipe, '_tsumo_count_'+s) is ref and ref == old
            for s,(ref,old) in counter.items())
        data['adoptions'] = factory.baseline_adoption.adoptions
        receiver = state['postcommit_current_receiver']
        data['publication'] = dict(issued=receiver.issued, released=receiver.released, errors=receiver.errors)
        data['first_move_final'] = pipe._first_move_sec_1p
        data['physical_certified'], data['collector_append_tested'] = False, False
        q.K.write(state['output']/'DIAGNOSTIC.json', fixture.subject.base.json_value(data))
        q.K.write(state['output']/'FIRING.json', fixture.subject.base.json_value(rows))


def execute(output: Path) -> Any:
    original = R.D
    try:
        R.D = SimpleNamespace(**(vars(original) | {'drive': drive}))
        return V2.execute(output)
    finally:
        R.D = original


def accepted(output: Path) -> Any:
    data, rows = R.K.read(output/'DIAGNOSTIC.json'), R.K.read(output/'FIRING.json')
    assert data['primary'] is None and len(data['completed']) == len(R.I.FRAMES)
    assert data['native_counter_unchanged'] and data['publication']['issued'] == 3
    assert data['raw_input_references_restored'] and data['score_input_references_restored']
    assert data['firing_references_restored']
    assert [r['stage'] for r in rows] == ['formula_deferred', 'firing_registered']
    last, registered = data['trace'][-1], rows[-1]
    assert last['history_count'] == 4 and sum(last['inventory']) == 8
    assert last['pending'] == [] and last['active']['mechanism'] == 'landing'
    assert sum(c != 0 for row in last['confirmed'] for c in row) == 6
    assert len(registered['private_state']['debts']) == 1
    return dict(completed_updates=len(data['completed']), original_J_firing_pop=True,
        origin_registered=True, predicted_final_published=False, native_counter_unchanged=True,
        actual_settlement_closed=False, next_hand_closed=False, next_long_repair_run='NO-GO')


def guards() -> dict[str, str]:
    paths = [*POLICY.glob('*.py'), *(DEFERRED/name for name in
        ('held.py', 'held_v2.py', 'connection.py', 'connection_v2.py'))]
    return R.K.guards() | {str(path): R.K.sha(path) for path in paths}


def main() -> int:
    common = SimpleNamespace(**(vars(R.K) | {'guards': guards}))
    return FunctionType(R.main.__code__, dict(vars(R), K=common, ROOT=ROOT, execute=execute, accepted=accepted))()


if __name__ == '__main__':
    raise SystemExit(main())
