"""I/O失敗した原runの保存数学だけを再検収。正常exitへの書換えは行わない。"""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
DEST = Path('/mnt/d/puyo_analyzer/verify/g2_a40_recovery_2026-09-14_v1')
sys.path.insert(0, str(ROOT.parent/'g2_second_prefix_runtime_2026-09-14_v40'))
import postrun as P


def main() -> None:
    started = time.perf_counter()
    output = ROOT.parent/P.T.OUTPUT_NAME
    entry, waited = P.read(output/'ENTRY_RESULT.json'), P.read(output/'TARGET_PARENT_WAIT.json')
    assert entry['exit_code'] == 1 and 'OSError: [Errno 5] Input/output error' in entry['error']
    assert waited == dict(child_exit_code=1, resource_guard_exit=0, source='actual_wait')
    P.T.approved(output)
    result, error = None, None
    try:
        with ExitStack() as stack:
            P.T.A.configured(stack)
            sys.path.insert(0, str(ROOT.parent/'g2_settled_basis_actual_mismatch_2026-09-12_v1'))
            import probe_native_merge
            owner = sys.modules[P.T.A.A.A.A.V4.OWNED_ALIAS]
            parts = owner.dependencies().modules()
            result, _, timeline = P.inspect(output, parts, P.modules(stack, owner, parts))
            result['pending_timeline_frames'] = len(timeline)
    except BaseException as failure:
        error = repr(failure)
    P.T.approved(output)
    value = dict(result=result, error=error, seconds=time.perf_counter()-started,
        original_run=str(output), original_exit_code=1, original_error=entry['error'],
        saved_mathematics_only=True, original_exit_guard_unchanged=True,
        whole_run_success=False, quality_gate_clear=False)
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST/'SECOND_SAVED_REVIEW.json').open('x') as stream: json.dump(value,stream,indent=2)
    print(json.dumps(dict(error=error, result=result, seconds=value['seconds'])))
    if error: raise RuntimeError(error)


if __name__ == '__main__': main()
