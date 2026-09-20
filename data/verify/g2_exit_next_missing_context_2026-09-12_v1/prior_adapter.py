"""原update/transitionのcodeとcaller契約を保持するCPU接続候補。証拠取得器は未実接続。"""
from __future__ import annotations
import sys
from types import FunctionType
from typing import Any, Callable
import next_vote_policy as P


def cloned(original: Any, **bindings: Any) -> Any:
    value = FunctionType(original.__code__,dict(original.__globals__,**bindings),
                         original.__name__,original.__defaults__,original.__closure__)
    value.__kwdefaults__ = original.__kwdefaults__
    return value


def install(module: Any, provider: Callable[..., Any], *args: Any,
            original_install: Any = None, **kwargs: Any) -> Any:
    original = module.PriorVotes
    def tick(*values: Any, **options: Any) -> Any:
        caller = sys._getframe(1)
        assert caller.f_code is original.update.__code__, 'original_prior_update_caller'
        value = module.Tick(*values,**options)
        value.exit_evidence = provider(caller,value)
        return value
    class Candidate(original):
        pass
    Candidate.update = cloned(original.update,Tick=tick)
    Candidate.transition = cloned(original.transition,past_history=P.past_history)
    assert Candidate.update.__code__ is original.update.__code__
    assert Candidate.transition.__code__ is original.transition.__code__
    return cloned(module.install if original_install is None else original_install,
                  PriorVotes=Candidate)(*args,**kwargs)
