"""旧実入口の生成地点だけを観測し、最初の採録前に計画停止する。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import sys
from typing import Any
import probe_imports as I

ROOT = Path(__file__).resolve().parent
STOP = 'joint_constructor_observed_before_first_collect'


def main() -> None:
    sys.path.insert(0, str(ROOT.parent / 'g2_collector_continuous_guard_2026-09-11_v1'))
    import raw_bound as B
    result: dict[str, Any] = {}
    original = B.build
    def build(*args: Any, **kwargs: Any) -> Any:
        caller = inspect.currentframe().f_back
        state = caller.f_locals['state']
        observer = state.get('provisional_context_observer')
        result.update(state_keys=sorted(state), observer_present=observer is not None,
            source_id=getattr(observer, 'source_id', None), run_id=getattr(observer, 'run_id', None),
            metadata_count=state['collector_metadata_sink'].rows.count,
            frame_range=[caller.f_globals['FIRST'], caller.f_globals['LAST']],
            caller_file=caller.f_code.co_filename, actual_updates=0, planned_stop=STOP)
        with (ROOT / 'CONSTRUCTOR_PROBE_v1.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
        raise RuntimeError(STOP)
    with ExitStack() as stack:
        stack.callback(setattr, B, 'build', original)
        B.build = build
        I.R.V6.V4.start = I.R.start
        I.R.V6.V5.create = I.R.V6.create
        code = I.R.V6.V5.main()
    assert B.build is original and result and code == 1
    print(json.dumps(dict(original_exit=code, source_restored=True, **result)))


if __name__ == '__main__':
    main()
