"""既存凍結器を再用するv12限定観測GO。品質合格ではない。"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
import target_entry as T


def evidence() -> dict:
    receipts = {}
    checks = ((T.ROOT / 'final_v2.xml', 43), (T.A.PACING_ROOT / 'connected_v1.xml', 15),
              (T.A.BASE / 'candidate_final_v1.xml', 145))
    for path, count in checks:
        cases = ET.parse(path).getroot().findall('.//testcase')
        T.require(len(cases) == count and all(len(case) == 0 for case in cases), 'tests')
        receipts[str(path)] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tests=count)
    return receipts


def main() -> None:
    output = T.ROOT.parent / T.OUTPUT_NAME
    T.require(not T.GO.exists() and not output.exists(), 'exclusive_run')
    review = json.loads((T.ROOT / 'PARENT_REVIEW.json').read_bytes())
    T.require(review['decision'] == 'GO_fixed_whole_target_observation'
              and review['known_open_connection_failures'] == [], 'parent_review')
    receipts = evidence()
    source = T.ROOT.parent / 'g2_arrival_ack_runtime_2026-09-12_v10/probe_manifest.py'
    spec = importlib.util.spec_from_file_location('_v12_original_freeze', source)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    T.require(previous.T is T, 'freeze_target')
    pins, snapshot = previous.freeze()
    value = dict(decision=review['decision'], output=str(output), frozen_files=pins,
        quality_gate_clear=False, production_permission=False, repair_accepted=False,
        known_connection_failures=[], known_quality_failures=review['remaining_video_conditions'],
        stop_conditions=review['stop_conditions'], bounds=[29052, 36298], diagnostic_bounds=[35172, 36298],
        test_receipts=receipts, source_snapshot=str(snapshot), previous_actual='video38_m1_completion_candidate_v11',
        purpose='有資格M1二採録/第三到来/全終了を熱監視付き減速版で同run検証する')
    with T.GO.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    T.approved(output)
    print(json.dumps(dict(sources=len(pins), quality_gate_clear=False)), flush=True)


if __name__ == '__main__':
    main()
