"""既存専用testの保存helperを再用。人工rec/外側ENGINEと実保存を区別する。"""
from __future__ import annotations
import contextlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import pytest
import adapter as Q
import cpu_fixture as C


def load(path: Path, name: str) -> Any:
    alias = '_palette_proof_history_' + name
    Q.require(alias not in sys.modules, 'history_alias_collision')
    before = list(sys.path)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before
    return module


def components(proof: Any) -> dict[str, Any]:
    d = load(Q.HISTORY_TEST.parent / 'adapter.py', 'downstream')
    a, r = load(d.TORCH, 'torch'), load(d.RUNTIME, 'runtime')
    r.M = load(d.RUNTIME.parent / 'adapter.py', 'evidence')
    latest = load(r.LATEST / 'adapter.py', 'latest')
    r.M.bind(latest)
    f = a.OLD.F
    namespace = {'Any': Any, 'Path': Path, 'SimpleNamespace': SimpleNamespace, 'sys': sys,
        'subprocess': subprocess, 'pytest': pytest, 'D': d, 'A': a, 'R': r, 'P': proof,
        'F': f, 'LATEST': latest, 'N': f.next_runner, 'write': C.write}
    return Q.source_functions(Q.HISTORY_TEST, ('artificial_rec', 'saved_history', 'finalization_of_saved_history'), namespace)


def run_saved(case: Any, monkeypatch: Any) -> None:
    n = components(case.proof)
    d, a, r, f = (n[k] for k in ('D', 'A', 'R', 'F'))
    output = case.output
    receipt = Q.read(output / 'PLAN.json')
    original = (f.validate_comparison, f.diagnostic_reports, n['LATEST'].entry.ORIGINAL_FINISH)
    with contextlib.ExitStack() as stack:
        d.install(stack, r, a, case.proof, enabled=True)
        a.install(stack, r)
        result, calls = n['saved_history'](output, receipt)
        with pytest.raises(FileExistsError):
            n['saved_history'](output, receipt)
    assert (f.validate_comparison, f.diagnostic_reports, n['LATEST'].entry.ORIGINAL_FINISH) == original
    assert calls == 1 and result['quality_gate_clear'] is False
    n['finalization_of_saved_history'](output, monkeypatch)
    case.proof.verify(output)
    Q.verify(output)
    C.write(output.parent / 'HISTORY_SAVE.json', {'original_history_calls': calls,
        'original_finalizer_called': True, 'actual_proof_receipt_reused': True,
        'artificial_rec_and_outer_engine': True, 'real_collector_run_executed': False,
        'old_failed_run_changed': False, 'quality_gate_clear': False})


def execute(case: Any, monkeypatch: Any) -> None:
    # frozen collector の namespace は消さず、保存専用の新processへ分離する。
    C.write(case.output / 'SPLIT_COMPARISON.json', case.report)
    command = [sys.executable, str(Path(__file__).resolve()), str(case.output)]
    with (case.output.parent / 'HISTORY_CHILD.log').open('x') as log:
        child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    C.write(case.output.parent / 'HISTORY_CHILD_EXIT.json', {'actual_exit': child.returncode,
        'command': command, 'fresh_process_no_namespace_replacement': True})
    assert child.returncode == 0


def main() -> int:
    if not __debug__:
        raise RuntimeError('optimized_python_unsupported')
    case = SimpleNamespace(output=Path(sys.argv[1]), proof=Q.load(Q.PROOF, 'history_proof'))
    with pytest.MonkeyPatch.context() as patch:
        run_saved(case, patch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
