"""終端着弾のCPU閉鎖後だけ、同一動画の有限実観測を固定する。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import target_entry as T

OLD = T.ROOT.parent / 'g2_second_prefix_runtime_2026-09-14_v35'
TERM = T.A.TERMINAL
REVIEW = T.ROOT.parent / 'g2_fable_upstream_preflight_2026-09-11_v1'
TESTS = tuple(TERM / name for name in (
    'original_witness_tap_v2.xml', 'binding_native_v1.xml', 'original_application_v1.xml',
    'session_order_v2.xml', 'warning_source_v1.xml', 'warning_original_local_v1.xml',
    'drop_order_domain_v2.xml', 'terminal_drop_candidate_v1.xml',
    'external_handoff_contract_v1.xml', 'terminal_registry_v5.xml',
    'recovery_dispatch_v3.xml', 'terminal_saved_cpu_v3.xml', 'terminal_saved_tail_v3.xml',
    'terminal_notice_after_v3.xml')) + (OLD / 'm1_timeline_v1.xml', T.ROOT.parent / 'g2_second_terminal_arrival_2026-09-14_v2/final_identity_v1.xml')
UNCHANGED = ('early_runtime.py', 'early_probability_capture.py', 'handoff_gate.py',
             'review_resources.py', 'target_entry.py', 'run_whole_target.sh')


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def inherited() -> None:
    for path, digest in read(OLD / 'TARGET_GO.json')['frozen_files'].items():
        T.require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest,
                  'A35_source_changed:' + path)
    for name in UNCHANGED:
        expected = (OLD / name).read_text().replace(OLD.name, T.ROOT.name).replace(
            'video38_second_prefix_candidate_v35', T.OUTPUT_NAME)
        T.require(expected.rstrip('\n') == (T.ROOT / name).read_text().rstrip('\n'),
                  'unreviewed_inherited_delta:' + name)


def evidence() -> None:
    for path in TESTS:
        suites = ET.parse(path).getroot().findall('.//testsuite')
        T.require(bool(suites) and all(int(s.get('tests', '0')) > int(s.get('skipped', '0'))
            and int(s.get('failures', '0')) == int(s.get('errors', '0')) == 0 for s in suites),
            'terminal_CPU_not_passed:' + str(path))
    for review in (315, 316, 317):
        wait = read(REVIEW / ('WAIT_v' + str(review) + '.json'))
        T.require(wait['actual_exit_code'] == 0 and not wait['timed_out'], 'terminal_review_incomplete')
    repair = T.ROOT.parent / 'g2_second_terminal_arrival_2026-09-14_v2'
    before = read(repair / 'CREATOR_v35_before_v2.json')
    after = read(repair / 'CREATOR_v36_after_v1.json')
    T.require(before['error'] == "ValueError('origin_runtime_final_class_changed')",
              'old_actual_creator_not_reproduced')
    T.require(after['error'] is None and after['source_aliases_released']
        and all(after['result'][key] for key in ('actual_creator_return_used',
            'actual_before_create_and_audit', 'session_saves_closed')), 'actual_creator_audit_unchecked')
    saved = read(T.ROOT / 'SAVED_MODULES_v1.json')
    T.require(all(saved[key] for key in ('actual_A36_saved_loader', 'shared_prefix_family_type',
        'native_module_same', 'private_aliases_released')), 'terminal_saved_loader_unchecked')


def main() -> None:
    T.require(not T.GO.exists() and not (T.ROOT.parent / T.OUTPUT_NAME).exists(), 'target_already_exists')
    parent = read(T.ROOT / 'PARENT_PREFIX_DECISION.json')
    T.require(parent['decision'] == 'GO' and parent['quality_gate_clear'] is False
              and parent['known_connection_failures'] == [], 'terminal_parent_not_GO')
    inherited()
    evidence()
    pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in T.runtime_sources()}
    value = dict(decision='GO_fixed_whole_target_observation', output=str(T.ROOT.parent / T.OUTPUT_NAME),
        quality_gate_clear=False, production_permission=False, frozen_files=pins,
        known_connection_failures=[], parent_decision=parent, independent_reviews=[315, 316, 317],
        partial_design_review=314, frame_range=[T.W.FIRST, T.W.LAST], stride=T.W.STRIDE,
        expected_updates=len(T.A.OC.FRAMES), stop_conditions=['original_strict_failure', 'RAM_guard',
            'source_or_owner_changed', 'unqualified_or_unsupported_arrival', 'original_completion_rejection'],
        original_M1_two_sample_gate_unchanged=True, observation_only=True, G3_run_authorized=False,
        obsolete_M1_window_not_new_mechanism_quality_requirement=True,
        multiple_video_generalization_is_G3=True, missing_real_evidence=parent['actual_video_remaining'],
        test_receipts=[str(path) for path in TESTS])
    with T.GO.open('x') as stream:
        json.dump(value, stream, indent=2)
    T.approved(T.ROOT.parent / T.OUTPUT_NAME)
    print(json.dumps(dict(frozen_files=len(pins), output=value['output'], quality_gate_clear=False)))


if __name__ == '__main__':
    main()
