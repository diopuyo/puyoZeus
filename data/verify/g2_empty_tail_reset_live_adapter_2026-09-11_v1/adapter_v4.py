"""実保存資格/失敗保存を接続し、reset時だけ実goalsを多scope検収へ渡す。"""
from __future__ import annotations
from types import FunctionType, SimpleNamespace as N
import sys
from typing import Any
import adapter_v3 as A
import proof_status as STATUS

ROOT = A.ROOT


def dependencies(stack: Any, old: Any) -> Any:
    modules = A.dependencies(stack,old)
    modules.proof = N(**(vars(modules.proof)|dict(Context=STATUS.derived(modules.proof.Context))))
    return modules


def finalize(stack: Any, main: Any) -> None:
    import live_finish as FINISH
    target = main.__globals__['Q'].FINAL
    original = target.evaluate
    def evaluate(goals: Any, rows: Any, legal: Any, output: Any, state: Any) -> Any:
        context = state['live_empty_reset_context']
        if context.proof is None:
            assert context.error is None
            return original(goals,rows,legal,output,state)
        assert context.error is None and context.recovery.error is None and context.recovery.failure is None
        assert context.recovery.pending is None and context.recovery.baseline_count==1
        factory = sys.modules['_private_live_finalizer'].context(state)
        assert factory is context.factory and factory.controller.legal is legal and output==state['output']
        for refs in (state['rolling_prefix_references'],factory.controller.empty_runtime_references):
            assert refs['installed'] and refs['closed'] and refs['restored']
        saved_status = FINISH.read(output,'LIVE_EMPTY_RESET.json')
        assert saved_status['error'] is None and saved_status['baseline_recovered']
        prepared = FINISH.prepared(factory,state,state['repeat_scope_guard'].reset_lease,
                                   state['private_suffix_modules'].evidence)
        result = FINISH.stage(goals,rows,legal,output,state,prepared,
                              firing_rows=state['repeated_firing_constructor']['rows'])
        main.__globals__['K'].write(output/'LIVE_EMPTY_RESET_FINAL.json',result)
        return result
    stack.callback(setattr,target,'evaluate',original)
    target.evaluate = evaluate


def configured(stack: Any) -> Any:
    function = FunctionType(A.A.configured.__code__,dict(vars(A.A),dependencies=dependencies))
    main = function(stack)
    finalize(stack,main)
    return main
