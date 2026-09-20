"""原run_liveと全constructor_guardを実行し、原Bridge生成直後だけ計画停止する。"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import target_entry as T

STOP = 'planned_stop_after_actual_whole_Bridge_before_video_loop'


def main() -> None:
    output = T.ROOT / 'actual_prepare_v1'
    assert not output.exists()
    refs = {}
    with ExitStack() as stack:
        selected = T.A.configured(stack)
        common = selected.__globals__['K']
        pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in T.runtime_sources()}
        T.A.protect(stack, common, pins)
        if selected.__globals__['S'].K is not common:
            T.A.protect(stack, selected.__globals__['S'].K, pins)
        whole = T.A.A.A.A.V4.A
        original = whole.W.Bridge

        def bridge(*args: object, **kwargs: object) -> object:
            refs['bridge'] = original(*args, **kwargs)
            raise RuntimeError(STOP)

        T.A.A.A.A.V4.replace_owned(stack, whole.W, 'Bridge', bridge)
        code = T.observe(selected, output)
        entry = json.loads((output / 'ENTRY_RESULT.json').read_bytes())
        assert code == 1 and STOP in entry['error'] and 'bridge' in refs
        dependency = refs['bridge'].state['whole_dependency_scope']
        assert dependency.closed and dependency.error is None
    assert whole.W.Bridge is original
    assert T.A.UNBOUND_MODULE.ALIAS not in sys.modules
    result = dict(actual_run_live=True, original_constructor_guard=True, original_Scope_prepare=True,
        original_Bridge_constructed=True, planned_stop=STOP, actual_entry_exit=code,
        original_dependency_scope_closed=True, original_bridge_restored=True,
        private_alias_removed=True, actual_video_frames_processed=0,
        full_end=False, M1_captured=False, quality_gate_clear=False)
    with (T.ROOT / 'ACTUAL_PREPARE_REVIEW_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
