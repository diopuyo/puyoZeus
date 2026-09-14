"""原load_default返却後・初回update前だけ共通候補を設置する。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
from typing import Any


def attach(stack: Any, candidate: Any, factory: Any, pipe: Any,
           state: Any, evidence: dict[str, Any]) -> None:
    owned = ExitStack()
    before = candidate.references(factory, pipe, state)
    try:
        candidate.install(owned, factory, pipe, state, evidence['rows'])
        owned.enter_context(candidate.P.Q.R.Q.installed(pipe, evidence['qualification']))
    except BaseException:
        owned.close()
        evidence['references_restored'] = before == candidate.references(factory, pipe, state)
        evidence['closed'] = True
        raise
    def close() -> None:
        try:
            owned.close()
        finally:
            evidence['references_restored'] = before == candidate.references(factory, pipe, state)
            evidence['closed'] = True
    stack.callback(close)
    evidence.update(installed=True, pipe_identity=id(pipe))


def install(stack: Any, collector: Any, factory: Any, state: Any, candidate: Any) -> dict[str, Any]:
    cls = collector.RecognitionPipeline
    original = inspect.getattr_static(cls, 'load_default')
    assert isinstance(original, classmethod), 'repeat_original_classmethod'
    assert 'repeated_firing_constructor' not in state, 'repeat_constructor_already_installed'
    assert factory.controller is not None and 'postcommit_current_receiver' in state
    evidence: dict[str, Any] = dict(attempts=0, installed=False, closed=False,
        references_restored=None, qualification={}, rows=[], live_constructor_verified=False)
    def load(inner: Any, *args: Any, **kwargs: Any) -> Any:
        assert inner is cls and evidence['attempts'] == 0, 'repeat_duplicate_or_foreign_constructor'
        evidence['attempts'] += 1
        pipe = original.__func__(inner, *args, **kwargs)
        assert type(pipe) is cls, 'repeat_foreign_constructor_result'
        assert not factory.controller.history, 'repeat_constructor_after_baseline'
        attach(stack, candidate, factory, pipe, state, evidence)
        return pipe
    state['repeated_firing_constructor'] = evidence
    stack.callback(setattr, cls, 'load_default', original)
    setattr(cls, 'load_default', classmethod(load))
    return evidence
