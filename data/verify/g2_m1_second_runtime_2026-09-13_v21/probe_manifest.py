"""v21の限定観測検収票を固定し、既存凍結器を再用する。"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
import target_entry as T


def evidence() -> dict:
    lane = T.A.PREVIOUS_ADAPTER.LIFECYCLE
    checks = [(T.ROOT / 'final_v1.xml', 17),
              (T.A.SAFETY / 'final_v2.xml', 43),
              (T.A.BASE / 'candidate_final_v1.xml', 145)]
    for name, count in (('full_prefix_path_v2', 1), ('full_firing_branches_v1', 1),
                        ('review202_origin_repair_v2', 1), ('review202_error_counter_v3', 1),
                        ('failure_save_v1', 2), ('saved_full_negative_v2', 11),
                        ('private_lifetime_v2', 5)):
        checks.append((lane / (name + '.xml'), count))
    checks += [(T.A.NOTICE / 'notice_v1.xml', 4), (T.A.NOTICE / 'notice_saved_v2.xml', 8),
               (T.A.NOTICE / 'notice_sm_v1.xml', 2), (T.ROOT / 'selection_v2.xml', 1),
               (T.ROOT / 'saved_v1.xml', 1), (T.ROOT / 'review206_v1.xml', 2)]
    receipts = {}
    for path, count in checks:
        cases = ET.parse(path).getroot().findall('.//testcase')
        T.require(len(cases) == count and all(len(case) == 0 for case in cases), 'tests')
        receipts[str(path)] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tests=count)
    path = T.ROOT / 'ACTUAL_PREPARE_REVIEW_v1.json'
    prepare = json.loads(path.read_bytes())
    T.require(all(prepare[k] is True for k in ('actual_run_live', 'original_constructor_guard',
              'original_Bridge_constructed', 'original_dependency_scope_closed',
              'original_bridge_restored', 'private_alias_removed')), 'actual_prepare')
    T.require(prepare['actual_entry_exit'] == 1 and prepare['actual_video_frames_processed'] == 0
              and prepare['quality_gate_clear'] is False, 'planned_stop_not_quality')
    receipts[str(path)] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for name in ('full_cpu_d8z_493x', 'firing_cpu_q5bp6cbh'):
        path = lane / name / 'RESULT.json'
        result = json.loads(path.read_bytes())
        T.require(result['quality_gate_clear'] is False and result['actual_video'] is False
                  and result['verified']['saved_distribution_and_consumptions_verified'], 'cpu_saved')
        receipts[str(path)] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return receipts


def main() -> None:
    output = T.ROOT.parent / T.OUTPUT_NAME
    T.require(not T.GO.exists() and not output.exists(), 'exclusive_run')
    review = json.loads((T.ROOT / 'PARENT_REVIEW.json').read_bytes())
    T.require(review['decision'] == 'GO_fixed_whole_target_observation'
              and review['known_open_connection_failures'] == [], 'parent_review')
    receipts = evidence()
    source = T.ROOT.parent / 'g2_arrival_ack_runtime_2026-09-12_v10/probe_manifest.py'
    spec = importlib.util.spec_from_file_location('_v21_original_freeze', source)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    T.require(previous.T is T, 'freeze_target')
    pins, snapshot = previous.freeze()
    value = dict(decision=review['decision'], output=str(output), frozen_files=pins,
        quality_gate_clear=False, production_permission=False, repair_accepted=False,
        known_connection_failures=[], known_quality_failures=review['remaining_video_conditions'],
        stop_conditions=review['stop_conditions'], bounds=[29052, 36298], diagnostic_bounds=[35172, 36298],
        test_receipts=receipts, source_snapshot=str(snapshot), previous_actual='video38_m1_prefix_candidate_v20',
        purpose='2P終了通知分類/1Pprefixの連続反映・有資格M1二採録・全終了の限定観測')
    with T.GO.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    T.approved(output)
    print(json.dumps(dict(sources=len(pins), quality_gate_clear=False)), flush=True)


if __name__ == '__main__':
    main()
