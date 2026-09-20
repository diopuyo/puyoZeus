"""子の実終了コードをwaitして保存する。原RESULTだけで融合完了としない。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main() -> int:
    name = sys.argv[1]
    assert name.startswith('prefix_cpu_v') and Path(name).name == name
    output = ROOT / name
    assert not output.exists()
    started = time.monotonic()
    child = subprocess.Popen([sys.executable, str(ROOT / 'run_runtime.py'), name], env=os.environ.copy())
    code = child.wait()
    result = dict(child_pid=child.pid, parent_pid=os.getpid(), exit_code=code,
        seconds=time.monotonic()-started, actual_wait=True, fused_runtime_exists=(output/'FUSED_RUNTIME.json').is_file(),
        quality_gate_clear=False, physical_certified=False)
    with (ROOT / (name + '_CHILD_EXIT.json')).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
