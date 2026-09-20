"""原sealの実guard集合を別親で検証する。動画品質は人工票で代替しない。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import target_entry as T

ROOT = T.ROOT / 'seal_parent_v2'
OUTPUT = ROOT / T.OUTPUT_NAME


def child() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        selected = T.A.configured(stack)
        common, closure = selected.__globals__['K'], selected.__globals__['Q']
        pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in T.runtime_sources()}
        T.A.protect(stack, common, pins)
        other = selected.__globals__['S'].K
        if other is not common:
            T.A.protect(stack, other, pins)
        receipt = dict(input_and_code_sha256=common.guards() | other.guards())
    # 入力/終了状態は人工。ENGINEは自作せず、実入口と同じ原sealへ渡す。
    summary = dict(references_restored=True, guards_unchanged=True,
                   goal=dict(current_event_observed=False, outer_publication_observed=False), artificial=True)
    common.write(OUTPUT / 'PLAN.json', receipt)
    common.write(OUTPUT / 'SUMMARY.json', summary)
    common.write(OUTPUT / 'ENTRY_RESULT.json', dict(pid=os.getpid(), exit_code=0, artificial=True))
    closure.seal(OUTPUT, summary, receipt)


def parent() -> None:
    started = time.perf_counter()
    process = subprocess.Popen([sys.executable, __file__, 'child'], cwd=Path.cwd())
    code = process.wait(timeout=120)
    T.require(code == 0, 'seal_child_exit')
    resources = Path(str(OUTPUT) + '.resources.jsonl')
    resources.write_text(json.dumps(dict(pid=process.pid, safety_stop=False, artificial=True)) + '\n')
    # 本番入口のscopeを人工出力先だけへ変更。別processの原finalizerを実行する。
    script = ('from pathlib import Path; import target_entry as T; '
              'T.ROOT=Path(__import__("sys").argv[1]); '
              'print(T.finalize(T.ROOT.parent/T.OUTPUT_NAME,0,0))')
    finalized = subprocess.run([sys.executable, '-c', script, str(ROOT / 'runner')],
        cwd=Path.cwd(), env=dict(os.environ, PYTHONPATH=str(T.ROOT) + os.pathsep + '.'),
        capture_output=True, text=True, timeout=120)
    T.require(finalized.returncode == 0, 'parent_finalize:' + finalized.stderr)
    result = json.loads((OUTPUT / 'COMPLETE').read_bytes())
    engine = json.loads((OUTPUT / 'PUBLICATION_ENGINE.json').read_bytes())
    T.require(result['computation_closed'] and not result['quality_gate_clear'], 'seal_complete')
    value = dict(child_exit=code, parent_exit=finalized.returncode, source_count=len(engine['guard_sha256']),
                 seconds=time.perf_counter()-started, actual_video=False, quality_gate_clear=False,
                 original_seal_to_separate_parent=True)
    (ROOT / 'RESULT.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    child() if sys.argv[1:] == ['child'] else parent()
