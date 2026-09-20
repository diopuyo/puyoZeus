"""共通設置口を人工110更新で検収し、旧合成と時系列を照合する。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
from types import FunctionType, SimpleNamespace
from typing import Any
import importlib.util
from pathlib import Path
import sys

SPEC = importlib.util.spec_from_file_location('_repeat_shared_runtime_connection', Path(__file__).with_name('connection.py'))
C = importlib.util.module_from_spec(SPEC)
assert SPEC.name not in sys.modules
sys.modules[SPEC.name] = C
SPEC.loader.exec_module(C)
import repeat_inputs_v2 as I

ROOT, K, P, MAX_UPDATES = C.ROOT, C.K, C.P, 120


def drive(q: Any, target: Any, m: Any, state: Any, real: Any, fixture: Any,
          binding: Any, factory: Any) -> dict[str, Any]:
    pipe, rows = real[0], []
    counters = {s:(getattr(pipe, '_tsumo_count_'+s), deepcopy(getattr(pipe, '_tsumo_count_'+s)))
                for s in ('1p', '2p')}
    data: dict[str, Any] = dict(trace=[], completed=[], primary=None)
    before = C.references(factory, pipe, state)
    try:
        with ExitStack() as stack:
            C.install(stack, factory, pipe, state, rows)
            C.G.R.D.loop(q, target, m, state, real, fixture, binding, factory, data)
        return dict(firing_continuation_reached=True, physical_certified=False)
    finally:
        data['firing_references_restored'] = before == C.references(factory, pipe, state)
        data['native_counter_unchanged'] = all(getattr(pipe, '_tsumo_count_'+s) is ref and ref == old
            for s,(ref,old) in counters.items())
        receiver = state['postcommit_current_receiver']
        data['publication'] = dict(issued=receiver.issued, released=receiver.released, errors=receiver.errors)
        q.K.write(state['output']/'DIAGNOSTIC.json', fixture.subject.base.json_value(data))
        q.K.write(state['output']/'FIRING.json', fixture.subject.base.json_value(rows))


def accepted(output: Any) -> dict[str, Any]:
    value = P.accepted(output)
    prior = K.read(P.ROOT/'cpu_v2/RESULT.json')
    for name in ('registered_frames', 'settled_frames', 'current_frames', 'completed_updates', 'fires'):
        assert value[name] == prior[name], 'shared_installer_timeline_changed:'+name
    value.update(shared_installer=True, original_110_timeline_equal=True, live_constructor_verified=False)
    return value


def main() -> int:
    source = P.Q.R
    execute = FunctionType(source.execute.__code__, dict(vars(source), I=I, MAX_UPDATES=MAX_UPDATES, drive=drive))
    common = SimpleNamespace(**(vars(K) | {'guards': C.guards}))
    return FunctionType(C.G.R.main.__code__, dict(vars(C.G.R), K=common, ROOT=ROOT,
        execute=execute, accepted=accepted))()


if __name__ == '__main__':
    raise SystemExit(main())
