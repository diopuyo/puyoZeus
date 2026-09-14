"""実v7と同じ既存starvation flagを人工CPU constructorへ渡す。原本体不変。"""
from __future__ import annotations
from types import FunctionType, SimpleNamespace
from typing import Any, Iterator

FLAG = 'enable_next_history_starvation_fix'


def configured(fixture: Any) -> Any:
    """既存flag注入器を二重に使い、原ojama観測flagも保持する。"""
    extra = FunctionType(fixture.original_real.__code__,dict(vars(fixture),FLAG=FLAG))
    def original_real(original: Any, evidence: Any) -> Any:
        palette: dict[str,Any] = {}
        merged = fixture.original_real(extra(original,palette),evidence)
        def real(frozen: Any, monkeypatch: Any) -> Iterator[Any]:
            try:
                yield from merged(frozen,monkeypatch)
            finally:
                evidence['palette_constructor_fixture'] = palette
        return real
    install = FunctionType(fixture.install.__code__,dict(vars(fixture),original_real=original_real))
    return SimpleNamespace(**(vars(fixture)|dict(install=install)))


def execute(output: Any) -> Any:
    """原reset CPU組成と同じ順序。人工constructor引数だけ実走へ整合する。"""
    with ExitStack() as stack:
        q = K.load('_reset_target_helpers',BASE,stack)
        parent = K.load('_reset_current_runner',K.CURRENT/'run_cpu.py',stack)
        old = {name:parent.load(name,stack) for name in parent.FIXED}
        a,full = old['assembly_history'],old['run_cpu']
        m = a.clone(S.loaded,FRAMES=I.FRAMES)(stack)
        facade = SimpleNamespace(**(vars(a)|{'session':lambda c,modules:S.A.session(stack,a,c,modules)}))
        def runner(parent: Any, active: Any, binding: Any, factory: Any) -> Any:
            def run(c: Any, state: Any, real: Any, fixture: Any, inner: Any) -> Any:
                return drive(q,m,state,real,fixture,binding,factory)
            return run
        result = a.clone(full.execute,A=facade,FRAMES=I.FRAMES,runner=runner,
            F=palette_fixture.configured(full.F))(output)
        assert result['builder_restored']
        return result


def loader(original: Any) -> Any:
    def load(alias: str, path: Any, stack: Any) -> Any:
        module = original(alias,path,stack)
        if alias=='_hidden_prefix_full_parent':
            import palette_fixture
            module.palette_fixture = palette_fixture
            module.execute = FunctionType(execute.__code__,dict(vars(module)))
        return module
    return load
