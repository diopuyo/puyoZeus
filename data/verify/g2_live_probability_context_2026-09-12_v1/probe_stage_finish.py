"""v70の原owner/world/保存を保持し、同じ実factoryの構造検査を後続実行する。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import sys
from typing import Any
import probe_lower_finish as OLD
import stage_probe_helpers as STAGE


def driver() -> Any:
    tree = ast.parse(inspect.getsource(OLD.OLD.OLD.V.main))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and n.value == 'POSITIVE_CORE_PROBE_v3.json']
    assert len(hits) == 1
    hits[0].value = 'POSITIVE_CORE_PROBE_v71.json'
    namespace = dict(vars(OLD.OLD.OLD.V))
    exec(compile(ast.fix_missing_locations(tree), OLD.OLD.OLD.V.__file__, 'exec'), namespace)
    return namespace['main']


def main() -> None:
    prior = OLD.lower
    def lower(kept: dict[str, Any]) -> Any:
        original = prior(kept)
        output = Path(kept['state']['output'])
        try:
            result = STAGE.verify(kept)
        except BaseException as error:
            failure = dict(error=repr(error), lower_result=original, state_keys=sorted(kept['state']),
                           stage='structure', quality_gate_clear=False)
            try:
                with (output / 'PROBABILITY_STAGE_FAILURE.json').open('x', encoding='utf-8') as stream:
                    json.dump(failure, stream, indent=2)
            except BaseException as save_error:
                print(json.dumps(dict(original_error=repr(error), save_error=repr(save_error))), file=sys.stderr)
            raise
        with (output / 'PROBABILITY_STAGE_FINISH.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
        return original
    with ExitStack() as stack:
        extra = ('probe_stage_finish.py', 'stage_probe_helpers.py', 'probability_stage.py', 'probability_boundary.py')
        for obj, key, value in ((OLD, 'driver', driver), (OLD, 'lower', lower),
                               (OLD.OLD, 'SOURCES', OLD.OLD.SOURCES + extra)):
            stack.callback(setattr, obj, key, getattr(obj, key))
            setattr(obj, key, value)
        OLD.main()


if __name__ == '__main__':
    main()
