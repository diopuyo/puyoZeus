"""人工保存票で原親finalizeの実呼出を検査する。動画や実waitの証明ではない。"""
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'g2_early_origin_runtime_2026-09-13_v30'
sys.path.insert(0, str(RUNTIME))
import target_entry as T


def write(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream)


def fixtures(output: Path, common: Any) -> None:
    write(output / 'FIXTURE_ONLY.json', dict(artificial=True, actual_wait=False, quality_gate_clear=False))
    write(output / 'ENTRY_RESULT.json', dict(pid=12345, exit_code=0))
    write(output / T.H.NAME, dict(frame=36898, root_probability_counts=dict(last_frame=36898,
        READY=1, HOLD=0, calls=1), reference_counts=dict(REFERENCE_ONLY=1), history_transferred=True,
        probability_candidates=1, history_first=29052, history_last=35160,
        original_M1_two_sample_gate_unchanged=True, quality_gate_clear=False))
    lifetime = dict.fromkeys(('sampling_closed', 'hook_closed', 'complete_restored',
        'capture_restored', 'references_released'), True)
    write(output / 'EARLY_PROBABILITY_LIFETIME.json', lifetime | dict(hook_error=None,
        original_error=None, quality_gate_clear=False))
    write(Path(str(output) + '.supervisor.json'), dict(source='actual_Popen_wait', child_pid=12345,
        child_exit_code=0, resource_guard_exit=0, supervisor_error=None, guard_exited_first=False))
    write(Path(str(output) + '.resources.jsonl'), dict(pid=12345, safety_stop=False,
        rss_limit_kib=8*1024*1024, minimum_available_kib=2*1024*1024,
        rss_kib=0, available_kib=4*1024*1024))
    closure = importlib.import_module('closure')
    with ExitStack() as scope:
        T.W.common(scope, common, T.A.A.A.A.V4.replace_owned)
        bounds = common.bounds()
    write(output / closure.ENGINE, dict(computation_closed=True, bounds=bounds,
        summary=dict(references_restored=True, guards_unchanged=True,
            goal=dict(current_event_observed=False, outer_publication_observed=False)),
        sha256={}, guard_sha256={}, **closure.PERMISSIONS))


def main() -> None:
    started = time.perf_counter()
    directory = ROOT / 'finalizer_dispatch_v1'
    directory.mkdir(exist_ok=False)
    output = directory / T.OUTPUT_NAME
    output.mkdir()
    sys.path.insert(0, str(T.FINALIZER_ROOT))
    common = importlib.import_module('common')
    fixtures(output, common)
    calls = set()
    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == 'call':
            calls.add((frame.f_code.co_filename, frame.f_code.co_name))
    old_root, old_profile = T.ROOT, sys.getprofile()
    try:
        T.ROOT = directory / 'runtime'
        sys.setprofile(profile)
        result = T.finalize(output, 0, 0)
    finally:
        sys.setprofile(old_profile)
        T.ROOT = old_root
    forbidden = [(path, name) for path, name in calls if name in ('checks', 'verify')
        and any(part in path for part in ('g2_full_observation_scope_',
            'g2_hidden_probability_capture_', 'g2_provisional_context_capture_'))]
    assert not forbidden and result['computation_closed'] and (output / 'COMPLETE').exists()
    value = dict(seconds=time.perf_counter()-started, actual_target_finalize_called=True,
        artificial_receipts=True, real_process_wait=False, video_updates=0,
        forbidden_observer_verifier_calls=forbidden, original_common_restored=common.END == 36300,
        calls=sorted(calls), quality_gate_clear=False)
    write(directory / 'RESULT.json', value)
    print(json.dumps({key: item for key, item in value.items() if key != 'calls'}))


if __name__ == '__main__':
    main()
