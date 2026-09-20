"""専用CPUだけ実行し、own/入力guardと実child終了・排他索引を保存する。"""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))
import loader as L


def write(path: Path, data: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)


def imported_sources() -> dict[str, str]:
    return {str(Path(m.__file__).resolve()): L.sha(Path(m.__file__))
        for n, m in tuple(sys.modules.items()) if n.startswith(('src.', 'scripts.'))
        and getattr(m, '__file__', None) and str(m.__file__).endswith('.py')}


def function_violations() -> list[Any]:
    return [(name, node.name, node.end_lineno - node.lineno + 1)
        for name in L.OWN for node in ast.walk(ast.parse((ROOT / name).read_text(encoding='utf-8')))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno - node.lineno + 1 > 50]


def main() -> int:
    output = ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    api = L.backend()
    own, inputs = L.guards(), L.input_guards() | imported_sources()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', LOADER_OUTPUT=str(output),
        PYTHONPATH=os.pathsep.join((str(ROOT), str(PROJECT))))
    command = [sys.executable, '-m', 'pytest', '-q', str(ROOT / 'test_loader.py'), '-p', 'no:cacheprovider']
    with (output / 'PYTEST.txt').open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        print({'actual_child_pid': child.pid}, flush=True)
        exit_code = child.wait()
    after_own = {p: L.sha(Path(p)) for p in own}
    after_inputs = {p: L.sha(Path(p)) for p in inputs}
    violations = function_violations()
    ok = exit_code == 0 and own == after_own and inputs == after_inputs and not violations
    write(output / 'RESULT.json', {'pid': os.getpid(), 'child_pid': child.pid, 'actual_child_exit': exit_code,
        'elapsed_sec': time.perf_counter() - started, 'own_before': own, 'own_after': after_own,
        'inputs_before': inputs, 'inputs_after': after_inputs, 'function_violations': violations,
        'cuda_initialized': api.torch.cuda.is_initialized(), 'model_inference_executed': False,
        'quality_gate_clear': False, 'status': 'PASS' if ok else 'FAIL'})
    if ok:
        index = {p.name: L.sha(p) for p in output.iterdir() if p.is_file()}
        write(output / 'COMPLETE.json', {'actual_child_exit': exit_code, 'sha256': index})
    print({'actual_child_exit': exit_code, 'status': 'PASS' if ok else 'FAIL'}, flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
