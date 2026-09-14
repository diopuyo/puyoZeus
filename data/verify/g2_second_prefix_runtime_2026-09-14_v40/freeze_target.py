"""既存PASSを保持し、同世代終了の残件だけを検証するA40固定票。"""
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import target_entry as T

OLD = T.ROOT.parent/'g2_second_prefix_runtime_2026-09-14_v39'
POST = T.A.INACTIVE_ROOT
CURRENT = T.A.READER_ROOT
CURRENT_TESTS = ('terminal_prefix_reader_v1.xml', 'terminal_reader_limits_v1.xml',
                 'terminal_completed_v3.xml', 'actual_terminal_schedule_v1.xml')
TESTS = ('terminal_boundary_v3.xml','first_terminal_v5.xml','terminal_original_fifo_v2.xml',
    'terminal_actual_prefix_selection_v1.xml','terminal_publication_boundary_v1.xml',
    'terminal_outer_boundary_v1.xml','terminal_reentry_v1.xml','terminal_clock_guard_v1.xml')


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def evidence() -> list[str]:
    prior = read(OLD/'TARGET_GO.json')
    for path, expected in prior['frozen_files'].items():
        T.require(sha256(Path(path).read_bytes()).hexdigest() == expected, 'old_source_changed:'+path)
    tests = list(prior['test_receipts']) + [str(CURRENT/name) for name in CURRENT_TESTS]
    for path in tests:
        suites = list(ET.parse(path).getroot().iter('testsuite'))
        T.require(suites and sum(int(s.get('tests',0)) for s in suites)>0 and all(
            int(s.get('errors',0)) == int(s.get('failures',0)) == 0 for s in suites), 'CPU_failure:'+path)
    creator = read(CURRENT/'creator_terminal_v40a/RESULT.json')
    T.require(creator['error'] is None and creator['source_aliases_released'] and all(
        creator['result'][key] is True for key in ('actual_before_create_and_audit','session_saves_closed',
            'first_type_selected','first_capture_selected','second_type_selected','actual_prefix_saved_selected',
            'original_prefix_wrapper_preserved','terminal_loader_candidate_connected')), 'creator_unverified')
    T.require(len(creator['result']['reader_bind_records']) == 1, 'reader_bind_unverified')
    first = read(POST/'full_terminal_cpu_v2/RESULT.json')
    T.require(first['error'] is None and first['verified']['terminal']['terminal_end']==24
        and first['verified']['transitions']==first['verified']['consumptions']==2, 'first_saved_unverified')
    second = read(POST/'SECOND_FULL_SAVED_A39_CPU_v1.json')
    T.require(second['error'] is None and second['actual_new_runtime_postrun_modules']
        and second['original_exit_code']==1 and second['synthetic_future_after_36884']
        and second['result']['end']==T.W.LAST and second['result']['acknowledged']==10
        and len(second['result']['unacknowledged_terminal_tokens'])==1, 'second_saved_unverified')
    return tests


def main() -> None:
    T.require(not T.GO.exists() and not (T.ROOT.parent/T.OUTPUT_NAME).exists(), 'exclusive_target')
    parent = read(T.ROOT/'PARENT_PREFIX_DECISION.json')
    T.require(parent['decision']=='GO' and parent['known_connection_failures']==[]
        and parent['preserve_passed_conditions'] and parent['state_reconstruction_not_retest']
        and parent['independent_unread_supplement_closed'], 'parent_not_GO')
    tests = evidence()
    pins = {str(path):sha256(path.read_bytes()).hexdigest() for path in T.runtime_sources()}
    value = dict(decision='GO_fixed_whole_target_observation',output=str(T.ROOT.parent/T.OUTPUT_NAME),
        quality_gate_clear=False,production_permission=False,frozen_files=pins,
        known_connection_failures=[],parent_decision=parent,test_receipts=tests,
        frame_range=[T.W.FIRST,T.W.LAST],stride=T.W.STRIDE,expected_updates=len(T.A.OC.FRAMES),
        stop_conditions=['original_error','RAM_VRAM_guard','source_or_scope_changed',
            'unqualified_terminal_or_arrival','unfinished_or_unrestored_close'],
        prior_pass_not_reopened=True,state_reconstruction_not_retest=True,
        original_A38_exit_preserved=True,G3_run_authorized=False)
    with T.GO.open('x') as stream:
        json.dump(value,stream,indent=2)
    T.approved(T.ROOT.parent/T.OUTPUT_NAME)
    print(json.dumps(dict(frozen_files=len(pins),output=value['output'],quality_gate_clear=False)))


if __name__ == '__main__':
    main()
