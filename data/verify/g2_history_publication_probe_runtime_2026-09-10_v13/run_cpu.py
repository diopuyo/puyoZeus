"""少数境界→同一版full/原loop/rawを一回。モデル/GPUは使わない。"""
from __future__ import annotations
import ast
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import common as K


def style() -> None:
    for name in K.OWN:
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((K.ROOT/name).read_bytes())):
            if isinstance(node, ast.FunctionDef):
                K.require(node.end_lineno-node.lineno+1 <= 50 and node.returns is not None, 'style:' + node.name)
                K.require(all(a.annotation is not None or a.arg in ('self','cls') for a in
                    node.args.posonlyargs+node.args.args+node.args.kwonlyargs), 'types:' + node.name)


def call(output: Path, name: str, argv: list[str], env: Any) -> dict[str, Any]:
    started = time.perf_counter()
    with (output/(name+'.log')).open('x') as stream:
        result = subprocess.run([sys.executable, *argv], env=env, stdout=stream, stderr=subprocess.STDOUT)
    return dict(exit_code=result.returncode, seconds=time.perf_counter()-started)


def main() -> int:
    output = (K.ROOT/sys.argv[1]).resolve()
    K.require(output.parent == K.ROOT and output.name.startswith('cpu_v'), 'CPU_output')
    output.mkdir(exist_ok=False)
    before, started = K.guards(), time.perf_counter()
    style()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
    tests = call(output, 'PYTEST', ['-m','pytest',str(K.ROOT/'test_probe.py'),str(K.ROOT/'test_ast_connection.py'),
        str(K.ROOT/'test_initial_hand_connection.py'),str(K.ROOT/'test_continuation_scope.py'),str(K.ROOT/'test_missing_connection.py'),
        str(K.ROOT/'test_scope_connection.py'),str(K.ROOT/'test_finalizer_connection.py'),'-q','-p','no:cacheprovider',
        '--basetemp='+str(output/'tmp'),'--junitxml='+str(output/'JUNIT.xml')], env)
    full = {'exit_code': None}
    if tests['exit_code'] == 0:
        (output/'full').mkdir(exist_ok=False)
        full = call(output, 'FULL', [str(K.ROOT/'cpu_smoke.py'), output.name+'/full'], env)
    after = K.guards()
    code = tests['exit_code'] or full['exit_code'] or int(before != after)
    result = dict(pid=os.getpid(), exit_code=code, seconds=time.perf_counter()-started,
        tests=tests, full=full, guards_before=before, guards_after=after, guard_equal=before == after,
        model_load=False, video_decode=False, gpu=False, quality_gate_clear=False, actual_current_event=False)
    K.write(output/'RESULT.json', result)
    if code == 0:
        K.write(output/'CPU_COMPLETE.json', result)
    K.write(output/'INDEX.json', {p.relative_to(output).as_posix(): K.sha(p) for p in output.rglob('*')
        if p.is_file() and 'tmp' not in p.relative_to(output).parts})
    print({k:v for k,v in result.items() if not k.startswith('guards_')}, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
