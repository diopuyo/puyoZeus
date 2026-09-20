"""保存worldの正例と反例を凍結ソース・実wait終了票へまとめる。"""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import empty_world as W
import world_inputs as I

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]


def write(path: Path, value: Any) -> None:
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main() -> int:
    output=ROOT/sys.argv[1]
    assert output.parent==ROOT and output.name.startswith('world_cpu_v')
    output.mkdir(exist_ok=False)
    paths=[ROOT/name for name in ('empty_world.py','world_inputs.py','test_world.py','run_world.py')]
    paths += [I.RUN/name for name in (*I.NAMES,'INDEX.json')]
    for folder in (W.SOURCE.parent,ROOT.parent/'g2_conditional_finalizer_world_2026-09-10_v1'):
        paths.extend(folder.glob('*.py'))
    pins=lambda:{str(p):sha256(p.read_bytes()).hexdigest() for p in paths}
    before,started=pins(),time.monotonic()
    write(output/'INPUTS.json',before)
    (output/'source_snapshot').mkdir()
    for p in paths:
        if p.parent==ROOT: shutil.copy2(p,output/'source_snapshot'/p.name)
    with (output/'stdout.log').open('x',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,'-m','pytest',str(ROOT/'test_world.py'),'-q','--tb=short',
            '--junitxml='+str(output/'tests.xml')],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
        code=child.wait()
    result=W.verify_world(**I.inputs()) if code==0 else {}
    unchanged=pins()==before
    write(output/'RESULT.json',dict(pid=child.pid,launcher_pid=os.getpid(),actual_exit=code,
        seconds=time.monotonic()-started,source_unchanged=unchanged,world=result,
        conditional_registration_coverage_verified=False,actual_factory=False,quality_gate_clear=False))
    return code or int(not unchanged)


if __name__=='__main__':
    raise SystemExit(main())
