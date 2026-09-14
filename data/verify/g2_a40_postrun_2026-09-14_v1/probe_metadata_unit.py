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
    assert RUN == T.ROOT.parent / T.OUTPUT_NAME and not RUN.is_symlink()
    T.approved(RUN)
    values = {name: V.C.read(RUN / name) for name in V.C.NAMES}
    complete = (RUN / 'COMPLETE').exists()
    code = values['ENTRY_RESULT.json']['exit_code']
    assert code == (0 if complete else 1)
    old_end = V.C.MAX_END
    try:
        V.C.MAX_END = SCOPED_END
        status = V.receipt_values(values, RUN, complete)
    finally:
        V.C.MAX_END = old_end
    supervisor_path = Path(str(RUN) + '.supervisor.json')
    supervisor = V.C.read(supervisor_path)
    waited = V.C.read(RUN / 'TARGET_PARENT_WAIT.json')
    assert supervisor['source'] == 'actual_Popen_wait' and waited == values['CHILD_EXIT.json']
    assert supervisor['child_pid'] == values['ENTRY_RESULT.json']['pid']
    assert supervisor['child_exit_code'] == waited['child_exit_code'] == code
    assert supervisor['resource_guard_exit'] == waited['resource_guard_exit'] == 0
    assert supervisor['supervisor_error'] is None and supervisor['forced_child_kill'] is False
    assert supervisor['guard_exited_first'] is False
    log = Path(str(RUN) + '.log')
    with log.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - V.C.LOG_TAIL_BYTES))
        tail = stream.read().splitlines()
    assert tail[-2:] == [b'ACTUAL_FINALIZE_EXIT=0', b'ACTUAL_SUPERVISOR_EXIT=0']
    paths = [RUN / name for name in V.C.NAMES] + [supervisor_path, RUN / 'TARGET_PARENT_WAIT.json', log]
    return status, {str(path): D.sha(path) for path in paths}


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
    with (Path('/mnt/d/puyo_analyzer/verify/g2_a40_recovery_2026-09-14_v1') / 'METADATA_UNIT_DIAGNOSIS_v1.json').open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    print(json.dumps({key: value[key] for key in ('seconds', 'whole_run_success', 'quality_gate_clear')}
        | {key: report[key] for key in ('joined_rows', 'actual_appends', 'join_differences', 'replay_differences')}), flush=True)


if __name__ == '__main__':
    main()
