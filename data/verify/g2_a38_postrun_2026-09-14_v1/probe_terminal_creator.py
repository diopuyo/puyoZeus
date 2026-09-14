"""A38実creator/最終型監査へ新終了loaderを接続。人工履歴CPU、動画なし。"""
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import terminal_selection as T
import creator_terminal_checks as CHECKS

ROOT = Path(__file__).resolve().parent


def main() -> None:
    attempt = sys.argv[1] if len(sys.argv) > 1 else 'v1'
    if not attempt.isalnum():
        raise ValueError('invalid CPU attempt')
    version = sys.argv[2] if len(sys.argv) > 2 else 'v38'
    if version not in ('v38', 'v39'):
        raise ValueError('invalid runtime version')
    output = ROOT/('creator_terminal_' + attempt)
    output.mkdir(exist_ok=False)
    (output/'START.json').write_text(json.dumps({'pid':os.getpid(), 'actual_video':False}))
    previous = ROOT.parent/'g2_legacy_m1_compatibility_2026-09-14_v1/probe_creator_audit.py'
    spec = importlib.util.spec_from_file_location('_post38_original_creator_probe', previous)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    sys.path.insert(0, str(ROOT.parent/('g2_second_prefix_runtime_2026-09-14_' + version)))
    import owned_adapter as A
    started, error, result = time.perf_counter(), None, {}
    try:
        with ExitStack() as stack:
            selected = A.configured(stack)
            owner = sys.modules[A.A.A.A.V4.OWNED_ALIAS]
            replace = A.A.A.A.V4.replace_owned
            def scope() -> object:
                connection = A.CURRENT.connection
                return connection.active_binding_stack() if connection.environment_called else stack
            if version == 'v38':
                T.install_owner(stack, owner, replace, scope)
            sys.path.insert(0, str(ROOT.parent/'g2_settled_basis_actual_mismatch_2026-09-12_v1'))
            import probe_native_merge
            with selected.__globals__['S'].configured():
                result = CHECKS.execute(probe, A, output)
                records = list((A.INACTIVE if version == 'v39' else T).TRACE)
                result['terminal_bound_records'] = records
                result['terminal_loader_candidate_connected'] = {row['kind'] for row in records} == {
                    'first', 'second', 'saved', 'boundary'}
                assert result['terminal_loader_candidate_connected'], 'missing_terminal_actual_load_binding'
    except BaseException as failure:
        error = repr(failure)
    value = dict(result=result, error=error, seconds=time.perf_counter()-started,
        runtime_version=version,
        artificial_initial_history_and_producer=True, actual_video=False, quality_gate_clear=False,
        source_aliases_released=not any(name in sys.modules for name in A.SC.NAMES))
    (output/'RESULT.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value))
    if error is not None:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
