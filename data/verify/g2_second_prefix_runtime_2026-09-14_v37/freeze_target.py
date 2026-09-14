"""承認済み状態再構築の固定票。旧成果物と元品質guardを保持する。"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import target_entry as T

OLD = T.ROOT.parent / 'g2_second_prefix_runtime_2026-09-14_v36'
REPAIR = T.A.WARNING_SOURCE.parent


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def evidence() -> list[str]:
    old = read(OLD / 'TARGET_GO.json')
    for path, digest in old['frozen_files'].items():
        T.require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'old_source_changed:' + path)
    tests = old['test_receipts'] + [str(REPAIR / name) for name in ('warning_score_v2.xml', 'math_receipt_v1.xml')]
    for path in tests:
        suites = ET.parse(path).getroot().findall('.//testsuite')
        T.require(bool(suites) and all(int(s.get('tests', '0')) > int(s.get('skipped', '0'))
            and int(s.get('failures', '0')) == int(s.get('errors', '0')) == 0 for s in suites), 'CPU_not_passed')
    saved = read(T.ROOT / 'SAVED_MODULES_v2.json')
    T.require(all(saved[key] for key in ('actual_A37_saved_loader', 'shared_prefix_family_type',
        'native_module_same', 'private_aliases_released', 'live_and_saved_warning_source_same',
        'actual_A36_rows_accepted')), 'final_warning_connection_unchecked')
    for name in ('early_runtime.py', 'early_probability_capture.py', 'handoff_gate.py',
                 'target_entry.py', 'review_resources.py', 'run_whole_target.sh', 'review_m1.py'):
        expected = (OLD / name).read_text().replace('v36', 'v37').replace('a36', 'a37').replace('A36', 'A37')
        T.require(expected.rstrip() == (T.ROOT / name).read_text().rstrip(), 'unexpected_runtime_change:' + name)
    before = (T.A.TERMINAL / 'observed_terminal_drop.py').read_text()
    T.require(before.replace('observed_warning_lower_bound/v1', 'observed_warning_lower_bound/v2').rstrip()
        == T.A.TERMINAL_MATH.read_text().rstrip(), 'terminal_math_changed')
    return tests


def main() -> None:
    T.require(not T.GO.exists() and not (T.ROOT.parent / T.OUTPUT_NAME).exists(), 'target_exists')
    parent = read(T.ROOT / 'PARENT_PREFIX_DECISION.json')
    T.require(parent['decision'] == 'GO' and parent['known_connection_failures'] == []
        and parent['user_authorized_prefix_reconstruction'], 'parent_not_GO')
    tests = evidence()
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in T.runtime_sources()}
    value = dict(decision='GO_fixed_whole_target_observation', output=str(T.ROOT.parent / T.OUTPUT_NAME),
        quality_gate_clear=False, production_permission=False, frozen_files=pins, known_connection_failures=[],
        parent_decision=parent, frame_range=[T.W.FIRST, T.W.LAST], stride=T.W.STRIDE,
        expected_updates=len(T.A.OC.FRAMES), test_receipts=tests,
        stop_conditions=['original_strict_failure', 'RAM_guard', 'source_or_owner_changed',
            'unqualified_or_unsupported_arrival', 'original_completion_rejection'],
        original_M1_two_sample_gate_unchanged=True, observation_only=True, G3_run_authorized=False,
        obsolete_M1_window_not_new_mechanism_quality_requirement=True, multiple_video_generalization_is_G3=True)
    with T.GO.open('x') as stream:
        json.dump(value, stream, indent=2)
    T.approved(T.ROOT.parent / T.OUTPUT_NAME)
    print(json.dumps(dict(frozen_files=len(pins), quality_gate_clear=False, output=value['output'])))


if __name__ == '__main__':
    main()
