"""原動画のメタデータだけで実load_default/候補設置/解除を検査する限定入口。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import live_adapter as A
import constructor_observation as C

ROOT = Path(__file__).resolve().parent
SOURCES = ('probe_actual_constructor.py', 'constructor_observation.py', 'live_adapter.py', 'loader.py',
           'context.py', 'probability_outer_scope.py', 'probability_owner.py', 'probability_finish.py',
           'probability_boundary.py', 'probability_world.py', 'probability_saved.py', 'probability_stage.py',
           'stage_retired_identity.py', 'run_actual_constructor_v2.sh')


def collect(main: Any, env: Any, output: Path, receipt: Any, config: Any) -> dict[str, Any]:
    base = env['runtime'].M.base
    before, paths = Path.cwd(), list(sys.path)
    kept: dict[str, Any] = {}
    try:
        os.chdir(base.SNAPSHOT)
        collector = base.load_collector()
        kwargs = main.__globals__['K'].actual_kwargs(base.collection_arguments(collector, config))
        assert kwargs['precise_seek'] is False, 'constructor_probe_must_not_decode_prefix'
        expected = C.expected_kwargs(collector, Path(receipt['video_path']), output, kwargs)
        main.__globals__['K'].write(output / 'CONSTRUCTOR_INPUT.json', dict(
            collector_kwargs=base.json_value(kwargs), expected_load_kwargs=base.json_value(expected)))
        try:
            C.intercepted(main.__globals__['collect'], kept)(env, output, receipt, collector, kwargs)
        except C.ConstructorObserved as stop:
            assert str(stop) == 'actual_constructor_return_observed'
        else:
            raise AssertionError('actual_constructor_stop_not_reached')
        finally:
            C.release(kept)
        return C.inspect_closed(main, kept, expected, output)
    finally:
        os.chdir(before)
        sys.path[:] = paths


def execute(output: Path) -> dict[str, Any]:
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'constructor_probe_original_cuda_device'
    assert not any(name == 'src' or name.startswith('src.') for name in sys.modules), 'constructor_probe_cold_src'
    with ExitStack() as stack:
        main = A.configured(stack)
        session = main.__globals__['S']
        with session.configured() as env, session.live_scopes(env):
            owned: list[Path] = []
            receipt, config = session.prepare_output(env, output, owned)
            main.__globals__['K'].write(output / 'PLAN.json', receipt)
            result = collect(main, env, output, receipt, config)
            env['runtime'].M.base.assert_unchanged(receipt['input_and_code_sha256'])
            result.update(C.gpu_receipt())
            main.__globals__['K'].write(output / 'CONSTRUCTOR_CLOSED.json', result)
    assert '_g2_live_probability_outer' not in sys.modules
    assert not any(name in sys.modules for name in A.FINISH_MODULES)
    return result | dict(outer_configuration_restored=True, original_sources_unchanged=True)


def main() -> int:
    name = sys.argv[1]
    assert Path(name).name == name and name.startswith('constructor_observation_'), 'constructor_probe_output'
    output = ROOT / name
    assert not output.exists() and not output.is_symlink(), 'constructor_probe_exclusive'
    raw = {str(ROOT / name): (ROOT / name).read_bytes() for name in SOURCES}
    started, code = time.perf_counter(), 0
    try:
        result = execute(output)
    except BaseException:
        result, code = dict(error=traceback.format_exc(), quality_gate_clear=False), 1
        if not output.exists():
            output.mkdir(exist_ok=False)
    unchanged = all(Path(path).read_bytes() == value for path, value in raw.items())
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(exist_ok=False)
    for path, value in raw.items():
        with (snapshot / Path(path).name).open('xb') as stream:
            stream.write(value)
    result.update(exit_code=code, seconds=time.perf_counter()-started, source_unchanged=unchanged,
                  source_sha256={path: hashlib.sha256(value).hexdigest() for path, value in raw.items()})
    with (output / 'RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
    return code if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
