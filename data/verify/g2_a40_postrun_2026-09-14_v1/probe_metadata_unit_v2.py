"""原metadata比較器だけを終了失敗runへ適用。全終了sinkの合格を発行しない。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'video38_second_prefix_candidate_v40'
RUNTIME = ROOT.parent / 'g2_second_prefix_runtime_2026-09-14_v40'
POSTRUN = ROOT.parent / 'g2_postrun_streamed_metadata_2026-09-10_v1'
sys.path.insert(0, str(POSTRUN))
import dependencies as D
import receipts_v2 as V
sys.path.insert(0, str(RUNTIME))
import target_entry as T
SCOPED_END = T.W.LAST + V.C.STRIDE


def inputs() -> tuple[Any, dict]:
    """欠測statusは補完せず、原J実完走区間だけ比較器へ渡す。"""
    T.approved(RUN)
    entry = V.C.read(RUN/'ENTRY_RESULT.json')
    waited = V.C.read(RUN/'TARGET_PARENT_WAIT.json')
    assert entry['exit_code']==1 and 'OSError: [Errno 5]' in entry['error']
    assert waited==dict(child_exit_code=1,resource_guard_exit=0,source='actual_wait')
    expected = [(f,side) for f in range(T.W.FIRST,T.W.LAST+T.W.STRIDE,T.W.STRIDE) for side in ('1P','2P')]
    with (RUN/'atomic_journal.jsonl').open() as stream:
        steps=[(r['frame_idx'],r['side']) for line in stream if (r:=json.loads(line))['kind']=='step']
    assert steps==expected
    status=dict(bounds=dict(first_frame=T.W.FIRST,end_exclusive=T.W.LAST+T.W.STRIDE,
        stride=T.W.STRIDE,expected_side_steps=len(steps)), original_exit_code=1,
        whole_run_verified=False, quality_gate_clear=False,
        closed_input_gate_passed=False, missing_original_status='POSTCOMMIT_CONSUMER_STATUS.json')
    paths=[RUN/name for name in ('ENTRY_RESULT.json','TARGET_PARENT_WAIT.json','atomic_journal.jsonl')]
    return status,{str(path):D.sha(path) for path in paths}


def main() -> None:
    started = time.perf_counter()
    status, guards = inputs()
    module = D.load()
    guards.update(D.guards(module))
    for name in (module.A.OUTER, module.A.META):
        path = RUN / name
        assert path.stat().st_size <= module.MAX_FILE_BYTES and not path.is_symlink()
        guards[str(path)] = D.sha(path)
    guards[str(Path(__file__))] = D.sha(Path(__file__))
    report = module.compare(RUN, status['bounds'])
    assert not report['join_differences'] and not report['replay_differences']
    assert all(D.sha(Path(path)) == digest for path, digest in guards.items())
    T.approved(RUN)
    value = dict(comparison=report, receipt=status, input_guards=guards,
        seconds=time.perf_counter() - started, original_comparator_reused=True, scoped_max_end=SCOPED_END,
        original_receipt_value_checks_reused=True, new_supervisor_actual_wait_checked=True,
        legacy_CLI_name_scope_not_reused=True, full_status_sink_audit_performed=False,
        whole_outer_materialized=False, whole_run_success=False, quality_gate_clear=False)
    with (Path('/mnt/d/puyo_analyzer/verify/g2_a40_recovery_2026-09-14_v1') / 'METADATA_UNIT_DIAGNOSIS_v2.json').open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    print(json.dumps({key: value[key] for key in ('seconds', 'whole_run_success', 'quality_gate_clear')}
        | {key: report[key] for key in ('joined_rows', 'actual_appends', 'join_differences', 'replay_differences')}), flush=True)


if __name__ == '__main__':
    main()
