"""保存実 PLAN の式 flag 3 項目だけを原 constructor に供給する。"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

NAMES = ('enable_chain_formula_read_verify', 'enable_formula_chain_count_update', 'enable_formula_step_interlude')


def transport(original: Any, kwargs: dict[str, Any], evidence: dict[str, Any]) -> Any:
    def real(frozen: Any, monkeypatch: Any) -> Iterator[Any]:
        cls, calls = frozen.RecognitionPipeline, []
        constructor = cls.__init__
        def init(pipe: Any, *args: Any, **supplied: Any) -> None:
            assert not calls and not set(kwargs).intersection(supplied)
            calls.append(dict(kwargs))
            constructor(pipe, *args, **(supplied | kwargs))
        cls.__init__ = init
        try:
            yield from original(frozen, monkeypatch)
        finally:
            cls.__init__ = constructor
            evidence.update(calls=calls, constructor_restored=cls.__init__ is constructor)
    return real


@contextmanager
def installed(stack: Any, common: Any, evidence: dict[str, Any]) -> Iterator[None]:
    plan = common.VERIFY/'video38_history_publication_probe_live_2026-09-10_v6/PLAN.json'
    source = common.read(plan)['actual_collector_kwargs']
    kwargs = {name: source[name] for name in NAMES}
    assert all(type(v) is bool and v for v in kwargs.values())
    evidence.update(plan=str(plan), sha256=common.sha(plan), supplied=kwargs, all_live_flags_equal=False)
    original = common.load
    def load(alias: str, path: Path, inner: Any) -> Any:
        result = original(alias, path, inner)
        if path != common.CURRENT/'run_cpu.py':
            return result
        old = result.load
        def member(name: str, owned: Any) -> Any:
            module = old(name, owned)
            if name == 'fixture_full':
                original_real = module.original_real
                owned.callback(setattr, module, 'original_real', original_real)
                module.original_real = lambda real, record: original_real(transport(real, kwargs, evidence), record)
            return module
        inner.callback(setattr, result, 'load', old)
        result.load = member
        return result
    common.load = load
    stack.callback(setattr, common, 'load', original)
    yield
    assert evidence['constructor_restored'] and len(evidence['calls']) == 1
