"""原Qualification生成/原callerを保持して証拠取得器を装着する。私有実走用候補。"""
from __future__ import annotations
import inspect
from pathlib import Path
from types import FunctionType
from typing import Any
import prior_adapter as A
import exit_next_evidence as E

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT.parents[2]/'.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/recognition_pipeline.py'
MAX_FUNCTIONS = 64
REQUIRED_LOCALS = frozenset(('p1_pair','both','slide_check_1p','frame','frame_idx','next_pair_1p'))


def select_update(value: Any) -> Any:
    """外側計装を変更せず、NEXT処理を持つ実生成codeをclosureから一意に選ぶ。"""
    pending,seen,candidates = [value],set(),[]
    while pending:
        fn = pending.pop()
        if not isinstance(fn,FunctionType) or id(fn) in seen:
            continue
        seen.add(id(fn))
        E.require(len(seen) <= MAX_FUNCTIONS,'update_graph_bound')
        if (Path(fn.__code__.co_filename).resolve() == PIPELINE
                and REQUIRED_LOCALS <= set(fn.__code__.co_varnames)):
            candidates.append(fn)
        pending.extend(inspect.getclosurevars(fn).nonlocals.values())
        pending.append(getattr(fn,'__wrapped__',None))
    generated = [fn for fn in candidates if '__next_live' in fn.__code__.co_names]
    chosen = generated or candidates
    E.require(len(chosen) == 1,'original_update_selection')
    return chosen[0]


def install(stack: Any, qualified: Any, update: Any, slide: type, patch: Any, owned: dict) -> None:
    module = qualified.V
    if module in owned:
        E.require(module.install is owned[module],'install_owner_changed')
        return
    update = select_update(update)
    E.require(isinstance(update,FunctionType) and Path(inspect.getsourcefile(update)).resolve() == PIPELINE
              and update.__qualname__ == 'RecognitionPipeline.update','original_update_source')
    E.require(update.__globals__['NextSlideDetector'] is slide,'original_slide_class')
    original = module.install
    E.require(isinstance(original,FunctionType) and original.__globals__ is vars(module),'original_prior_install')
    def connected(inner: Any, machine: Any, qualification: Any) -> Any:
        E.require(module.install is connected and type(qualification) is qualified.Qualification,'qualification_type')
        r = qualification.recovery
        E.require(machine is r.pipe._sm_1p,'machine_owner')
        provider = E.Provider(qualification,update.__code__,slide,r.state['output'])
        inner.push(provider.close)
        return A.install(module,provider,inner,machine,qualification,original_install=original)
    patch(stack,module,'install',connected)
    owned[module] = connected
