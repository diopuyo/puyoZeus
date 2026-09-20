"""実保持factoryのworld/確率保存を検査する。実constructor資格は作らない。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import probe_owner_finish_v2 as OLD
import probability_world as WORLD
import probability_saved as SAVED

ROOT = Path(__file__).resolve().parent
RESULT = 'PROBABILITY_LOWER_FINISH.json'
REQUIRED = ('probabilistic_tracking_mode', 'probabilistic_basis_connection',
            'repeat_scope_guard', 'conditional_full_rows', 'postcommit_publication_consumer')
LIVE_KEYS = ('private_suffix_factory', 'private_suffix_modules', 'repeated_firing_constructor',
             'combined_restored', 'conditional_runtime_factory', 'rolling_prefix_references')


def driver() -> Any:
    tree = ast.parse(inspect.getsource(OLD.OLD.V.main))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and n.value == 'POSITIVE_CORE_PROBE_v3.json']
    assert len(hits) == 1
    hits[0].value = 'POSITIVE_CORE_PROBE_v70.json'
    namespace = dict(vars(OLD.OLD.V))
    exec(compile(ast.fix_missing_locations(tree), OLD.OLD.V.__file__, 'exec'), namespace)
    return namespace['main']


def lower(kept: dict[str, Any]) -> dict[str, Any]:
    state, factory = kept['state'], kept['factory']
    assert all(key in state for key in REQUIRED), 'lower_missing_actual_state'
    before = dict(state)
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(OLD.OLD.BASE))
        original = importlib.import_module('runtime_world')
    finally:
        sys.path[:] = previous
    assert Path(original.__file__).resolve() == OLD.OLD.BASE / 'runtime_world.py'
    lease = state['repeat_scope_guard'].reset_lease
    world = WORLD.verify(original, factory, state, lease)
    connection, mode = state['probabilistic_basis_connection'], state['probabilistic_tracking_mode']
    serializer = type(connection).__init__.__globals__['S']
    native = N(extract=type(mode.native).observe.__globals__['extract'])
    probability = SAVED.verify(state, serializer, native)
    assert set(state) == set(before) and all(state[k] is v for k, v in before.items())
    return dict(world=world, probability=probability, actual_factory=True,
                live_constructor_keys={key: key in state for key in LIVE_KEYS},
                stage1_verified=False, full_probability_finalizer_verified=False,
                original_live_constructor=False, model_inference_performed=False,
                actual_video=False, quality_gate_clear=False)


def main() -> None:
    prior = OLD.OLD.closed_samecall
    def closed(kept: dict[str, Any]) -> Any:
        result = prior(kept)
        output = Path(kept['state']['output'])
        try:
            report = lower(kept)
        except BaseException as error:
            failure = dict(error=repr(error), owner_result=result, state_keys=sorted(kept['state']),
                           stage='lower_world_saved', quality_gate_clear=False)
            try:
                with (output / 'PROBABILITY_LOWER_FAILURE.json').open('x', encoding='utf-8') as stream:
                    json.dump(failure, stream, indent=2)
            except BaseException as save_error:
                print(json.dumps(dict(original_error=repr(error), save_error=repr(save_error))), file=sys.stderr)
            raise
        with (output / RESULT).open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        return result
    with ExitStack() as stack:
        for obj, name, value in ((OLD, 'driver', driver), (OLD.OLD, 'closed_samecall', closed),
                                (OLD, 'SOURCES', OLD.SOURCES + ('probe_lower_finish.py',
                                 'probability_world.py', 'probability_saved.py'))):
            stack.callback(setattr, obj, name, getattr(obj, name))
            setattr(obj, name, value)
        OLD.main()


if __name__ == '__main__':
    main()
