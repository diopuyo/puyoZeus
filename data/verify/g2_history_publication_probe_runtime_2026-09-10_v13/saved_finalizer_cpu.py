"""最新scope110保存票を実走用wrapperから一回再集計。動画は再走しない。"""
from __future__ import annotations
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType, ModuleType, SimpleNamespace
from typing import Any
import common as K
import goals as G
import finalizer_connection as F

SOURCE = K.VERIFY/'g2_repeated_firing_runtime_candidate_2026-09-10_v1/scope_composition_v1/cpu_v1'
SNAPSHOT = K.PROJECT/'.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
PROBE = K.VERIFY/'g2_inventory_producer_diagnosis_2026-09-09_v1/probe.py'
CPU_FIRST, CPU_END, CPU_HISTORY_FIRST = 34772,34992,34796


def bounded_goals() -> Any:
    bounds = SimpleNamespace(**(vars(K) | dict(FIRST=CPU_FIRST,END=CPU_END,HISTORY_FIRST=CPU_HISTORY_FIRST,
        FRAMES=tuple(range(CPU_FIRST,CPU_END,K.STRIDE)))))
    module = ModuleType('_CPU_saved_goals')
    module.__dict__.update(vars(G) | {'K':bounds})
    for name in ('history','current_proof','publication','evaluate'):
        original = getattr(G,name)
        setattr(module,name,FunctionType(original.__code__,vars(module)))
    return module


def main() -> int:
    output = K.ROOT/sys.argv[1]
    K.require(output.parent == K.ROOT and output.name.startswith('saved_cpu_v'),'saved_CPU_output')
    output.mkdir(exist_ok=False)
    index = K.read(SOURCE/'INDEX.json')
    before = K.guards() | {str(SOURCE/n):h for n,h in index.items()} | {str(PROBE):K.sha(PROBE)}
    K.require(all(K.sha(Path(p)) == h for p,h in before.items()),'saved_CPU_input_changed')
    started, result, code = time.perf_counter(), {}, 0
    try:
        with ExitStack() as stack:
            paths = list(sys.path)
            stack.callback(setattr,sys,'path',paths)
            K.require(not any(n == 'src' or n.startswith('src.') for n in sys.modules),'cold_snapshot_required')
            sys.path.insert(0,str(SNAPSHOT))
            probe = K.load('_saved_scope_finalizer_legal',PROBE,stack)
            for name,module in tuple(sys.modules.items()):
                if name.startswith('src.') and getattr(module,'__file__',None):
                    path = Path(module.__file__).resolve()
                    K.require(path.is_relative_to(SNAPSHOT),'foreign_snapshot_module')
                    before[str(path)] = K.sha(path)
            rows = [json.loads(line) for line in (SOURCE/'directional_history.jsonl').read_text().splitlines()]
            state = {'repeated_firing_constructor':dict(installed=True,closed=True,references_restored=True,
                rows=K.read(SOURCE/'FIRING.json'))}
            goal = F.evaluate(bounded_goals(),rows,probe.placement_matches,SOURCE,state)
            K.require(goal['firing_placements']==2 and len(goal['current_proofs'])==7
                and goal['issued']==7 and goal['released']==0,'saved_actual_goals')
            K.write(output/'GOAL.json',goal)
            result = dict(goal_verified=True,original_updates_rerun=0,CPU_only_bounds=[CPU_FIRST,CPU_END,CPU_HISTORY_FIRST])
    except BaseException as error:
        import traceback
        result,code = dict(error=repr(error),traceback=traceback.format_exc()),1
    equal = all(K.sha(Path(p)) == h for p,h in before.items())
    code = code or int(not equal)
    result.update(pid=os.getpid(),seconds=time.perf_counter()-started,exit_code=code,guards=before,
        guards_unchanged=equal,GPU_model_video=False,quality_gate_clear=False)
    K.write(output/'RESULT.json',result)
    K.write(output/'INDEX.json',{p.name:K.sha(p) for p in output.iterdir() if p.is_file()})
    print({k:v for k,v in result.items() if k not in ('guards','traceback')},flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
