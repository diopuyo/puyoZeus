"""人工輸送の単体検収を排他保存。実動画/GPU成立と区別する。"""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
OWN = ('witness.py','adoption.py','fixture.py','test_witness.py','run_cpu.py')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)


def main() -> int:
    output = ROOT/sys.argv[1]
    assert output.parent==ROOT and output.name.startswith('cpu_v')
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    guards = {name:sha(ROOT/name) for name in OWN}
    with (output/'pytest.log').open('x',encoding='utf-8') as stream:
        child = subprocess.run([sys.executable,'-m','pytest',str(ROOT/'test_witness.py'),'-q'],
            stdout=stream,stderr=subprocess.STDOUT,check=False)
    functions = []
    for name in OWN:
        for node in ast.walk(ast.parse((ROOT/name).read_bytes())):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                functions.append(dict(file=name,name=node.name,lines=node.end_lineno-node.lineno+1,
                    typed=node.returns is not None))
    unchanged = all(sha(ROOT/name)==digest for name,digest in guards.items())
    style = all(row['lines']<=50 and row['typed'] for row in functions)
    code = 0 if child.returncode==0 and unchanged and style else 1
    save(output/'RESULT.json',dict(pid=os.getpid(),exit_code=code,pytest_exit=child.returncode,
        seconds=time.perf_counter()-started,source_sha256=guards,source_unchanged=unchanged,
        style_pass=style,functions=functions,artificial_transport=True,gpu=False,
        original_full_update=False,quality_gate_clear=False))
    save(output/'INDEX.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(dict(output=str(output),exit_code=code)),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
