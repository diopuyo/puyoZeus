"""A33終了済み原票に2P新consumerを適用。G2全体の合格は発行しない。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any, Iterator
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent/'g2_second_prefix_runtime_2026-09-14_v33'
sys.path.insert(0,str(RUNTIME))
import target_entry as T
RUN = ROOT.parent/T.OUTPUT_NAME


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def lines(path: Path) -> Iterator[dict]:
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def inspect(saved: Any, parts: Any, original: Any) -> list[dict]:
    modes = read(RUN/'BELIEF_M1_SESSION.json')['modes']
    first = min(mode['initial']['state']['frame'] for mode in modes)
    steps = {r['frame_idx']:r for r in lines(RUN/'atomic_journal.jsonl')
             if r.get('kind')=='step' and r['side']=='2P' and r['frame_idx']>=first}
    contexts = {r['frame_idx']:r for r in lines(RUN/'provisional_context.jsonl') if r['frame_idx']>=first}
    paths,results = set(),[]
    for mode in modes:
        initial = mode['initial']['source_call_token'].replace(':','_')
        path = RUN/('SECOND_PREFIX_'+initial+'.jsonl')
        T.require(path.is_file() and not path.is_symlink(),'second_postrun_sidecar_missing')
        paths.add(path)
        end = mode['retired']['frame']-2 if mode['retired'] else max(steps)
        replay = saved.Replay(parts.mode.C.T,parts.binding.S,parts.mode.CORE.V1.N,
                              original,mode,steps,contexts)
        results.append(replay.run(list(lines(path)),end))
    T.require(paths==set(RUN.glob('SECOND_PREFIX_*.jsonl')),'second_postrun_extra_sidecar')
    return results


def saved_module(parts: Any) -> Any:
    """Sessionを生成しない保存検査でも、元の遅延factoryを通す。"""
    selected = [module for module in tuple(sys.modules.values())
        if getattr(module,'__file__',None) and Path(module.__file__).resolve()==T.A.PC.SECOND]
    T.require(len(selected)==1,'second_postrun_binding_selection')
    selected[0].session_class(SimpleNamespace(Session=object),parts.mode)
    return sys.modules[T.A.PC.NAMES[3]]


def main() -> None:
    T.approved(RUN)
    entry = read(RUN/'ENTRY_RESULT.json')
    wait = read(RUN/'TARGET_PARENT_WAIT.json')
    supervisor = read(Path(str(RUN)+'.supervisor.json'))
    T.require(wait['source']=='actual_wait' and supervisor['source']=='actual_Popen_wait'
        and wait['child_exit_code']==entry['exit_code']==supervisor['child_exit_code']
        and wait['resource_guard_exit']==supervisor['resource_guard_exit']==0
        and supervisor['child_pid']==entry['pid'] and not supervisor['forced_child_kill']
        and supervisor['supervisor_error'] is None,'second_postrun_actual_wait')
    with ExitStack() as stack:
        T.A.configured(stack)
        sys.path.insert(0,str(ROOT.parent/'g2_settled_basis_actual_mismatch_2026-09-12_v1'))
        import probe_native_merge
        owner = sys.modules[T.A.A.A.A.V4.OWNED_ALIAS]
        parts = owner.dependencies().modules()
        alias = '_a33_original_second_pending_replay'
        original = owner.bootstrap().load(alias,ROOT.parent/
            'g2_m1_completion_candidate_2026-09-12_v1/second_pending_replay.py',
            {'native_consumption':parts.mode.CORE.V1.N})
        stack.callback(sys.modules.pop,alias,None)
        results = inspect(saved_module(parts),parts,original)
    T.approved(RUN)
    result = dict(results=results,actual_entry_exit=entry['exit_code'],
        all_pending_empty=all(r['pending']==0 for r in results),quality_gate_clear=False,
        actual_J_and_context_replayed=True,physical_ground_truth=False)
    with (ROOT/'POSTRUN_v1.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
