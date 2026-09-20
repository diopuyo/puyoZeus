"""新規排他出力だけへ観測値を保存するCPU入口。"""
from __future__ import annotations

import ast
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import observables as O

UNIT = Path(__file__).resolve().parent
OWN = ('observables.py', 'test_observables.py', 'run_cpu.py', 'PLAN.md')
MAX_FUNCTION_LINES = 50
SELECTED = (
    ('1P', 34888, 34892, 'parent_selected_progress'),
    ('1P', 34930, 34934, 'parent_selected_progress'),
    ('2P', 35784, 35786, 'parent_selected_progress'),
    ('2P', 35786, 35790, 'parent_selected_progress'),
    ('2P', 34914, 34918, 'parent_selected_crossing'),
    ('2P', 34918, 34922, 'parent_selected_crossing'),
    ('2P', 34910, 34914, 'parent_selected_stationary_neighbor'),
    ('2P', 34922, 34926, 'parent_selected_stationary_neighbor'),
    ('2P', 35780, 35784, 'parent_selected_stationary_neighbor'),
    ('1P', 34884, 34888, 'neighbor_control_not_new_truth_label'),
)


def save(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def guards() -> dict[str, str]:
    return {name: O.sha((UNIT / name).read_bytes()) for name in OWN}


def function_limits() -> list[dict[str, Any]]:
    result = []
    for name in OWN:
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((UNIT / name).read_bytes())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                size = node.end_lineno - node.lineno + 1
                if size > MAX_FUNCTION_LINES or node.returns is None:
                    result.append({'file': name, 'name': node.name, 'lines': size})
    return result


def test_artifacts(output: Path) -> int:
    import pytest
    with (output / 'PYTEST.log').open('x', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = pytest.main([str(UNIT / 'test_observables.py'), '-q',
                '-p', 'no:cacheprovider', '--basetemp', str(output / 'synthetic_tmp'),
                '--junitxml', str(output / 'PYTEST.xml')])
    return int(result)


def selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for side, before, after, role in SELECTED:
        matching = [row for row in rows if row['side'] == side
                    and row['frame_before'] == before and row['frame_after'] == after]
        O.require(len(matching) == len(O.SLOTS), 'selected_pair_missing')
        result.append({'side': side, 'before': before, 'after': after,
            'selection_origin': role, 'selection_is_measurement_classification': False,
            'slots': {row['slot']: row['summary'] for row in matching}})
    return result


def actual_measurement(output: Path) -> dict[str, Any]:
    O.configure()
    rois = O.read_rois()
    images, inputs = O.load_images()
    rows = O.measure_all(images, rois)
    save(output / 'INPUTS.json', inputs)
    save(output / 'OBSERVATIONS.json', {'format': 'next-motion-observables/v1',
        'environment': O.environment(), 'rois': rois, 'rows': rows,
        'quality_gate_clear': False, 'physical_progress_certified': False,
        'production_permission': False, 'video_sha_recomputed': False})
    save(output / 'SELECTED.json', selected_rows(rows))
    return {'png_count': sum(len(items) for items in images.values()),
            'record_count': len(rows), 'adjacent_pair_count': len(rows) // 8,
            'point_count': sum(row['summary']['seed_count'] for row in rows),
            'input_count': len(inputs), 'environment': O.environment()}


def execute(output: Path) -> int:
    before, started = guards(), time.perf_counter()
    result: dict[str, Any] = {'pid': os.getpid(), 'status': 'failed',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'quality_gate_clear': False, 'physical_progress_certified': False,
        'production_permission': False, 'new_detector_or_threshold': False}
    code = 1
    try:
        O.require(not function_limits(), 'function_limits')
        result['pytest_exit'] = test_artifacts(output)
        O.require(result['pytest_exit'] == 0, 'synthetic_test_failed')
        result['measurement'] = actual_measurement(output)
        O.require('torch' not in sys.modules, 'unexpected_model_runtime')
        O.require(guards() == before, 'own_source_changed')
        result.update(status='measured_not_physical_certification', source_guards=before,
                      own_source_unchanged=True, torch_imported=False, function_violations=[])
        code = 0
    except BaseException as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc),
                           'traceback': traceback.format_exc()}
    result.update(actual_exit=code, elapsed_sec=time.perf_counter() - started)
    save(output / 'RESULT.json', result)
    names = ('INPUTS.json', 'OBSERVATIONS.json', 'SELECTED.json', 'RESULT.json', 'PYTEST.log', 'PYTEST.xml')
    index = {name: O.sha((output / name).read_bytes()) for name in names if (output / name).exists()}
    save(output / 'INDEX.json', index)
    if code == 0:
        save(output / 'CPU_COMPLETE.json', {'status': 'diagnostic_measurement_only',
            'index_sha256': O.sha((output / 'INDEX.json').read_bytes()),
            'physical_progress_certified': False, 'quality_gate_clear': False,
            'production_permission': False})
    print(json.dumps(result, ensure_ascii=False))
    return code


def main() -> int:
    O.require(len(sys.argv) == 2, 'usage: run_cpu.py new_output_name')
    output = (UNIT / sys.argv[1]).resolve()
    O.require(output.parent == UNIT and output.name.startswith('cpu_v'), 'output_scope')
    output.mkdir(exist_ok=False)
    return execute(output)


if __name__ == '__main__':
    raise SystemExit(main())
