"""2P複数消費修復の独立差分検収後に有限実観測を固定する。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import target_entry as T

OLD = T.ROOT.parent/'g2_second_prefix_runtime_2026-09-14_v32'
REPAIR = T.A.PREFIX
REVIEW = T.ROOT.parent/'g2_fable_upstream_preflight_2026-09-11_v1'
TESTS = tuple(REPAIR/name for name in ('composed_session_v1.xml','composition_ownership_v1.xml','notice_prefix_error_v2.xml','composed_positive_origin_v1.xml'))
UNCHANGED = ('early_runtime.py','early_probability_capture.py','handoff_gate.py',
    'review_m1.py','review_resources.py','target_entry.py','run_whole_target.sh')


def inputs() -> dict:
    decision = json.loads((T.ROOT/'PARENT_PREFIX_DECISION.json').read_bytes())
    if decision['decision']!='GO' or decision['quality_gate_clear'] is not False:
        raise ValueError('prefix_parent_not_GO')
    old = json.loads((OLD/'TARGET_GO.json').read_bytes())
    for path,digest in old['frozen_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('old_source_changed:'+path)
    for name in UNCHANGED:
        source = (OLD/name).read_text().replace(OLD.name,T.ROOT.name).replace(
            'video38_second_prefix_candidate_v32',T.OUTPUT_NAME)
        if source.rstrip('\n')!=(T.ROOT/name).read_text().rstrip('\n'):
            raise ValueError('unreviewed_runtime_delta:'+name)
    wait = json.loads((REVIEW/'WAIT_v297.json').read_bytes())
    if wait['actual_exit_code']!=0 or wait['timed_out']:
        raise ValueError('prefix_review_incomplete')
    for path in TESTS:
        suites = ET.parse(path).getroot().findall('.//testsuite')
        if not suites or any(int(s.get('failures','0')) or int(s.get('errors','0'))
            or int(s.get('tests','0'))<=int(s.get('skipped','0')) for s in suites):
            raise ValueError('prefix_CPU_not_passed:'+str(path))
    cold = json.loads((REPAIR/'FULL_CONSTRUCTOR_v1.json').read_bytes())
    saved = json.loads((REPAIR/'POSTRUN_SELECTION_v1.json').read_bytes())
    if not (cold['actual_full_loader'] and cold['actual_session_instance'] and cold['actual_origin_binding'] and cold['aliases_released']
            and saved['corrected_consumer_loaded'] and saved['aliases_released']):
        raise ValueError('prefix_cold_selection_unchecked')
    normal_notice()
    return decision


def normal_notice() -> None:
    # v1の正常caseだけ再用。失敗caseは別v2で所有期待を是正済み。
    root = ET.parse(REPAIR/'notice_prefix_flow_v1.xml').getroot()
    cases = [case for case in root.iter('testcase') if case.get('name')=='test_notice_non_consumption_and_original_error[False]']
    if len(cases)!=1 or any(cases[0].find(tag) is not None for tag in ('failure','error','skipped')):
        raise ValueError('normal_notice_case_not_passed')


def main() -> None:
    if T.GO.exists() or (T.ROOT.parent/T.OUTPUT_NAME).exists():
        raise FileExistsError('target_already_exists')
    decision = inputs()
    pins = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in T.runtime_sources()}
    value = dict(decision='GO_fixed_whole_target_observation',output=str(T.ROOT.parent/T.OUTPUT_NAME),
        quality_gate_clear=False,production_permission=False,frozen_files=pins,
        known_connection_failures=[],parent_decision=decision,independent_reviews=[285,286,287,288,291,292,293,294,296,297],
        frame_range=[T.W.FIRST,T.W.LAST],stride=T.W.STRIDE,expected_updates=len(T.A.OC.FRAMES),
        stop_conditions=['original_strict_failure','RAM_guard','source_or_owner_changed',
            'qualified_zero_support','original_tail_or_completion_rejection'],
        original_M1_two_sample_gate_unchanged=True,G3_run_authorized=False,observation_only=True,
        obsolete_M1_window_not_new_mechanism_quality_requirement=True,multiple_video_generalization_is_G3=True,
        missing_real_evidence=['same_run_2P_qualified_recovery','continuous_observer_to_36900',
            'tail_accounting_and_end','death_and_end_timing_comparison','full_exit'],
        test_receipts=[str(path) for path in TESTS])
    with T.GO.open('x') as stream:
        json.dump(value,stream,indent=2)
    T.approved(T.ROOT.parent/T.OUTPUT_NAME)
    print(json.dumps(dict(frozen_files=len(pins),output=value['output'],quality_gate_clear=False)))


if __name__=='__main__':
    main()
