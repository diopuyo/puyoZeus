"""構造identity修復を新runで検収し、次の外側終了用の実参照事実も保存する。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import inspect
import json
import sys
from typing import Any
import probe_stage_finish as OLD


def driver() -> Any:
    source = OLD.OLD.OLD.OLD.V
    tree = ast.parse(inspect.getsource(source.main))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and n.value == 'POSITIVE_CORE_PROBE_v3.json']
    assert len(hits) == 1
    hits[0].value = 'POSITIVE_CORE_PROBE_v72.json'
    namespace = dict(vars(source))
    exec(compile(ast.fix_missing_locations(tree), source.__file__, 'exec'), namespace)
    return namespace['main']


def main() -> None:
    prior = OLD.STAGE.verify
    def verify(kept: dict[str, Any]) -> Any:
        state = kept['state']
        guard = state['repeat_scope_guard']
        mode = state['probabilistic_tracking_mode']
        connection = state['probabilistic_basis_connection']
        facts = dict(guard_is_lease_guard=guard is guard.reset_lease.guard,
            state_is_mode_state=state is mode.state, guard_frame=guard.frame,
            native_frame=mode.native.last_frame, guard_error=repr(guard.error),
            guard_record=guard.record, binding_none=guard.binding is None,
            integer_1p_present='1P' in kept['factory'].controller.history,
            factory_is_connection=kept['factory'] is connection.recovery.factory,
            actual_video=False, quality_gate_clear=False)
        save_error = None
        try:
            with (state['output'] / 'OUTER_PROBABILITY_FACTS.json').open('x', encoding='utf-8') as stream:
                json.dump(facts, stream, indent=2)
        except (OSError, TypeError, ValueError) as error:
            save_error = repr(error)
            print(json.dumps(dict(outer_facts_save_error=save_error)), file=sys.stderr)
        result = prior(kept)
        return result | dict(outer_facts_saved=save_error is None, outer_facts_error=save_error)
    with ExitStack() as stack:
        target = OLD.OLD.OLD
        for obj, key, value in ((OLD, 'driver', driver), (OLD.STAGE, 'verify', verify),
            (target, 'SOURCES', target.SOURCES + ('probe_retired_stage.py', 'stage_retired_identity.py'))):
            stack.callback(setattr, obj, key, getattr(obj, key))
            setattr(obj, key, value)
        OLD.main()


if __name__ == '__main__':
    main()
