"""合格を再利用し、旧M1互換出口と残終了だけを閉じる固定入力票。"""
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import target_entry as T

OLD = T.ROOT.parent/'g2_second_prefix_runtime_2026-09-14_v37'
COMPAT = T.A.COMPAT_ROOT


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def evidence() -> list[str]:
    old = read(OLD/'TARGET_GO.json')
    for path, expected in old['frozen_files'].items():
        T.require(sha256(Path(path).read_bytes()).hexdigest() == expected, 'old_source_changed')
    tests = list(old['test_receipts']) + [str(COMPAT/'scope_after_v1.xml'),
        str(COMPAT/'saved_compatibility_v1.xml'), str(T.A.WARNING_SOURCE.parent/'saved_raw_tail_v1.xml')]
    for path in tests:
        suites = ET.parse(path).getroot()
        T.require(all(int(s.get('errors', 0)) == int(s.get('failures', 0)) == 0
            for s in suites.iter('testsuite')), 'CPU_failure:'+path)
    creator = read(COMPAT/'CREATOR_v38_v2.json')
    T.require(creator['error'] is None and creator['result']['actual_before_create_and_audit']
        and creator['result']['session_saves_closed'] and creator['source_aliases_released'], 'creator_unverified')
    child = read(COMPAT/'EMPTY_CLIENT_CLOSE_v1.json')
    T.require(child['real_start_worker'] and child['child_exit'] == child['accepted_count'] == 0
        and child['closed'], 'empty_child_unclosed')
    loader = read(T.ROOT/'SAVED_MODULES_v1.json')
    T.require(loader['actual_A38_saved_loader'] and loader['selected_raw_consumer']
        and loader['shared_types'] and loader['private_aliases_released'], 'saved_loader_unverified')
    return tests


def main() -> None:
    T.require(not T.GO.exists() and not (T.ROOT.parent/T.OUTPUT_NAME).exists(), 'exclusive_target')
    parent = read(T.ROOT/'PARENT_PREFIX_DECISION.json')
    T.require(parent['decision'] == 'GO' and parent['known_connection_failures'] == []
        and parent['preserve_passed_conditions'] and parent['state_reconstruction_not_retest'], 'parent_not_GO')
    tests = evidence()
    pins = {str(path):sha256(path.read_bytes()).hexdigest() for path in T.runtime_sources()}
    value = dict(decision='GO_fixed_whole_target_observation', output=str(T.ROOT.parent/T.OUTPUT_NAME),
        quality_gate_clear=False, production_permission=False, frozen_files=pins,
        known_connection_failures=[], parent_decision=parent, test_receipts=tests,
        frame_range=[T.W.FIRST,T.W.LAST], stride=T.W.STRIDE, expected_updates=len(T.A.OC.FRAMES),
        stop_conditions=['original_error_except_explicit_legacy_coverage_observation', 'RAM_VRAM_guard',
            'source_or_scope_changed', 'unqualified_arrival', 'unfinished_or_unrestored_close'],
        original_M1_two_sample_gate_unchanged=False, legacy_M1_coverage_observational=True,
        prior_pass_not_reopened=True, state_reconstruction_not_retest=True,
        G3_run_authorized=False, original_A37_exit_preserved=True)
    with T.GO.open('x') as stream:
        json.dump(value, stream, indent=2)
    T.approved(T.ROOT.parent/T.OUTPUT_NAME)
    print(json.dumps(dict(frozen_files=len(pins), output=value['output'], quality_gate_clear=False)))


if __name__ == '__main__':
    main()
